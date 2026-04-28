
'''This file implements the Deformable DETR detection head, which operates on the patch tokens produced by DINO.
It includes the DeformableCrossAttention module, which performs multi-scale deformable attention, and the SetCriterion class, which computes the loss for training the model.
It also includes utility functions for building the position encodings and the detection head.'''

def build_position_encoding(hidden_dim: int, height: int = 32, width: int = 32, device=None, dtype=None):
    """
    Generate 2D sinusoidal position encodings for a given height and width, with the specified hidden dimension.
    By default, it generates encodings for a 32x32 feature map, which can be used for the deformable attention in Deformable DETR.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return build_2d_sincos_position_embedding(height, width, hidden_dim // 2, device, dtype)

def build_deformable_detection_head(
    num_features: int,
    num_queries: int = 100,
    num_decoder_heads: int = 8,
    num_decoder_layers: int = 6,
    num_classes: int = 10,
    hidden_dim: int = 256,
    num_feature_levels: int = 3,
    num_points: int = 4,
):
    """
    Utility function to instantiate the DeformableDetectionHead with all the correct arguments.
    """
    return DeformableDetectionHead(
        num_features=num_features,
        num_queries=num_queries,
        num_decoder_heads=num_decoder_heads,
        num_decoder_layers=num_decoder_layers,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_feature_levels=num_feature_levels,
        num_points=num_points,
    )
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

from detection_lora_deformable.utils import box_cxcywh_to_xyxy, generalized_box_iou



# Loss Functions

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


# Hungarian Matcher

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


# SetCriterion

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
        """Compute the classification loss."""
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


#  Deformable DETR Detection Head


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def build_2d_sincos_position_embedding(height: int, width: int, num_pos_feats: int, device, dtype) -> torch.Tensor:
    y_embed, x_embed = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, device=device, dtype=dtype),
        torch.linspace(0.5, width - 0.5, width, device=device, dtype=dtype),
        indexing="ij",
    )

    eps = 1e-6
    y_embed = y_embed / (height + eps) * 2 * math.pi
    x_embed = x_embed / (width + eps) * 2 * math.pi

    dim_t = torch.arange(num_pos_feats, device=device, dtype=dtype)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos.reshape(1, height * width, -1)


class MLP(nn.Module):
    '''Simple multi-layer perceptron (MLP) with ReLU activations between layers.'''
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MultiScaleDeformableAttention(nn.Module):
    '''Implements the multi-scale deformable attention mechanism as described in the Deformable DETR paper.'''
    def forward(
        self,
        value: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_heads, head_dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
        value_list = value.split([height * width for height, width in spatial_shapes_list], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []

        for level_id, (height, width) in enumerate(spatial_shapes_list):
            value_l = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch_size * num_heads, head_dim, height, width)
            )
            sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l)

        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, num_queries, num_levels * num_points
        )
        output = (
            (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * head_dim, num_queries)
        )
        return output.transpose(1, 2).contiguous()


class DeformableCrossAttention(nn.Module):
    '''Implements the cross-attention mechanism for the Deformable DETR decoder, which uses multi-scale deformable attention.'''
    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads

        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = MultiScaleDeformableAttention()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        with torch.no_grad():
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.num_heads, 1, 1, 2)
                .repeat(1, self.num_levels, self.num_points, 1)
            )
            for point in range(self.num_points):
                grid_init[:, :, point, :] *= point + 1
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
    ) -> torch.Tensor:
        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        value = self.value_proj(encoder_hidden_states)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        elif reference_points.shape[-1] == 2:
            offset_normalizer = torch.as_tensor(
                [[width, height] for height, width in spatial_shapes_list],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.attn(value, spatial_shapes_list, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableDecoderLayer(nn.Module):
    """Implements a single layer of the Deformable DETR decoder."""

    def __init__(self, hidden_dim: int, num_heads: int, num_levels: int, num_points: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = DeformableCrossAttention(hidden_dim, num_heads, num_levels, num_points)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * 4, hidden_dim, 2)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes_list: List[Tuple[int, int]],
    ) -> torch.Tensor:
        residual = hidden_states
        q = hidden_states + query_pos
        attn_output, _ = self.self_attn(q, q, hidden_states)
        hidden_states = self.self_attn_norm(residual + F.dropout(attn_output, p=self.dropout, training=self.training))

        residual = hidden_states
        repeated_reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes_list), 1)
        cross_output = self.cross_attn(hidden_states, encoder_hidden_states, repeated_reference_points, spatial_shapes_list)
        hidden_states = self.cross_attn_norm(residual + F.dropout(cross_output, p=self.dropout, training=self.training))

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.final_norm(residual + F.dropout(hidden_states, p=self.dropout, training=self.training))
        return hidden_states


class DeformableDetectionHead(nn.Module):
    """Deformable DETR-style detection head operating on DINO patch tokens."""

    def __init__(
        self,
        num_features: int,
        num_queries: int = 100,
        num_decoder_heads: int = 8,
        num_decoder_layers: int = 6,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_feature_levels: int = 3,
        num_points: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_feature_levels = num_feature_levels
        self.num_points = num_points

        self.input_proj = nn.Linear(num_features, hidden_dim) if num_features != hidden_dim else nn.Identity()
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, hidden_dim) * 0.01)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.reference_points = nn.Linear(hidden_dim, 4)

        self.decoder_layers = nn.ModuleList(
            [
                DeformableDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_decoder_heads,
                    num_levels=num_feature_levels,
                    num_points=num_points,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.class_predictors = nn.ModuleList([nn.Linear(hidden_dim, num_classes + 1) for _ in range(num_decoder_layers)])
        self.bbox_predictors = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)])

    def _build_pyramid(self, memory: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        batch_size, sequence_length, hidden_dim = memory.shape
        side = int(math.sqrt(sequence_length))
        if side * side != sequence_length:
            raise ValueError(
                "DeformableDetectionHead expects square patch tokens so it can build a multi-scale feature pyramid."
            )

        feature_map = memory.transpose(1, 2).reshape(batch_size, hidden_dim, side, side)
        sources = [feature_map]
        for _ in range(1, self.num_feature_levels):
            sources.append(F.avg_pool2d(sources[-1], kernel_size=2, stride=2))

        flattened_sources = []
        spatial_shapes_list: List[Tuple[int, int]] = []
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes_list.append((height, width))
            source = source.flatten(2).transpose(1, 2)
            position_embeddings = build_2d_sincos_position_embedding(
                height=height,
                width=width,
                num_pos_feats=self.hidden_dim // 2,
                device=memory.device,
                dtype=memory.dtype,
            )
            source = source + position_embeddings + self.level_embed[level].view(1, 1, -1)
            flattened_sources.append(source)

        return flattened_sources, spatial_shapes_list

    def forward(self, memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            memory: (seq_len, batch_size, num_features)
        Returns:
            dict with 'pred_logits' and 'pred_boxes'
        """
        memory = self.input_proj(memory).permute(1, 0, 2)
        flattened_sources, spatial_shapes_list = self._build_pyramid(memory)
        encoder_memory = torch.cat(flattened_sources, dim=1)

        batch_size = encoder_memory.shape[0]
        query_pos = self.query_position.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = self.query_content.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        outputs_class = []
        outputs_coord = []

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                query_pos=query_pos,
                encoder_hidden_states=encoder_memory,
                reference_points=reference_points,
                spatial_shapes_list=spatial_shapes_list,
            )

            class_logits = self.class_predictors[layer_index](hidden_states)
            bbox_deltas = self.bbox_predictors[layer_index](hidden_states)
            pred_boxes = (bbox_deltas + inverse_sigmoid(reference_points)).sigmoid()

            outputs_class.append(class_logits)
            outputs_coord.append(pred_boxes)

            if layer_index < len(self.decoder_layers) - 1:
                reference_points = pred_boxes.detach()

        predictions: Dict[str, torch.Tensor] = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

        if len(outputs_class) > 1:
            predictions["aux_outputs"] = [
                {"pred_logits": class_logits, "pred_boxes": box_predictions}
                for class_logits, box_predictions in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return predictions


TransformerDetectionHead = DeformableDetectionHead
