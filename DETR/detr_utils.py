import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

from DETR.utils import box_cxcywh_to_xyxy, generalized_box_iou

""" This file contains the main components of the DETR architecture, including:
- HungarianMatcher: computes an assignment between the targets and the predictions of the network using the Hungarian algorithm.
- SetCriterion: computes the loss for DETR, including classification loss, cardinality error, and bounding box losses.
- TransformerDetectionHead: defines the detection head for the DETR model.
"""

# =============================================================================
# Loss Functions
# =============================================================================

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# =============================================================================
# Hungarian Matcher
# =============================================================================

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple]:
        """
        Performs the matching : given the outputs and targets, computes a matching between them
        using the Hungarian algorithm.
        The matching is based on the costs of the predicted class, the L1 box distance and the giou distance.

        Args:
            outputs: dict containing at least 'pred_logits' and 'pred_boxes'
            targets: list of dicts, each containing 'labels' and 'boxes'
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        device = outputs["pred_logits"].device
        if len(tgt_ids) == 0:
            return [(torch.tensor([], dtype=torch.int64, device=device), torch.tensor([], dtype=torch.int64, device=device)) for _ in range(bs)]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64, device=device), torch.as_tensor(j, dtype=torch.int64, device=device))
            for i, j in indices
        ]


# =============================================================================
# SetCriterion
# =============================================================================

class SetCriterion(nn.Module):
    """ 
    This class computes the loss for DETR. The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    The losses are weighted and summed up in the final loss, which is returned.
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: Dict, eos_coef: float, losses: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the classification loss (NLL)."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device,
        )
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        if log and len(target_classes_o) > 0:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, which is the absolute error in the number of predicted non-empty boxes."""
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        
        if len(target_boxes) == 0:
            return {
                "loss_bbox": torch.tensor(0.0, device=outputs["pred_boxes"].device),
                "loss_giou": torch.tensor(0.0, device=outputs["pred_boxes"].device),
            }
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following indices."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following indices."""
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Calculate the following losses:
           - 'labels': CrossEntropy loss for classification
            - 'cardinality': Absolute error in the number of predicted non-empty boxes
            - 'boxes': L1 regression loss for the bounding boxes and GIoU loss for the bounding boxes
        """
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        First, we compute the hungarian assignment between outputs and targets.
        Then we supervise each pair of matched ground-truth / prediction (supervise class and box)
        
        Args:
            outputs: dict of tensors, see the output specification of the model for more details
            targets: list of dicts, such that len(targets) == batch_size.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, aux_indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


# =============================================================================
# DETR Detection Head
# =============================================================================

class TransformerDetectionHead(nn.Module):
    """
    DETR-style detection head.
    """

    def __init__(
        self,
        num_features: int,
        num_queries: int = 100,
        num_decoder_heads: int = 8,
        num_decoder_layers: int = 6,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim) if num_features != hidden_dim else nn.Identity()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_decoder_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=False,
            ),
            num_layers=num_decoder_layers,
        )
        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            memory: (seq_len, batch_size, num_features)
        Returns:
            dict with 'pred_logits' and 'pred_boxes'
        """
        memory = self.input_proj(memory)
        bs = memory.shape[1]
        queries = repeat(self.query_embed.weight, "q f -> q b f", b=bs)
        decoded = self.decoder(queries, memory)
        decoded = decoded.permute(1, 0, 2)
        pred_logits = self.class_predictor(decoded)
        pred_boxes = self.bbox_predictor(decoded)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
