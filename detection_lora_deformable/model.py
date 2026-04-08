import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from typing import List, Dict, Tuple, Optional

from detection_lora.lora import create_backbone_with_lora
from detection_lora_deformable.deformable_detr import SetCriterion, HungarianMatcher, DeformableDetectionHead, build_position_encoding
from detection_lora.utils import box_cxcywh_to_xyxy, generalized_box_iou
from detection_lora.utils import extract_patch_targets, get_boundaries
import copy
import torch.nn.functional as F
from deformable_detr_repo.util.misc import NestedTensor, nested_tensor_from_tensor_list  # Peut rester si NestedTensor n'est pas redéfini



class LoRADeformableDetectionModel(L.LightningModule):
    def __init__(
        self,
        backbone_name: str = "dinov2",
        backbone_size: str = "small",
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 6,
        pre_norm: bool = False,
        num_feature_levels: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        eos_coef: float = 0.1,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        use_patching: bool = False,
        img_size: int = 1024,
        num_patches: int | None = None,
        patch_size: int = 224,
        overlap_size: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = not use_patching
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.use_patching = use_patching
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.val_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox")

        if use_patching:
            self.boundaries, self.img_size = get_boundaries(num_patches, overlap_size, patch_size, img_size=self.img_size)
        else:
            self.img_size = patch_size
            self.boundaries = None

        self.backbone = create_backbone_with_lora(
            name=backbone_name,
            model_size=backbone_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        
        # On utilise la tête DeformableDetectionHead custom (pure PyTorch)
        from detection_lora_deformable.deformable_detr import build_deformable_detection_head
        # Synchronise hidden_dim avec la sortie du backbone
        hidden_dim = self.backbone.num_features
        self.deformable_detr_head = build_deformable_detection_head(
            num_features=hidden_dim,
            num_queries=num_queries,
            num_decoder_heads=nheads,
            num_decoder_layers=dec_layers,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels,
            num_points=4,
        )
        # Projection inutile si hidden_dim == num_features, mais on garde pour compatibilité
        self.input_proj = nn.ModuleList()
        self.input_proj.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.hidden_dim = hidden_dim  # Pour accès dans forward_patch


        if not use_patching:
            self.resize_full = torchvision.transforms.Resize((self.img_size, self.img_size), antialias=True)

        matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={"loss_ce": cost_class, "loss_bbox": cost_bbox, "loss_giou": cost_giou},
            eos_coef=eos_coef,
            losses=["labels", "boxes", "cardinality"],
        )

    def forward_patch(self, patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Extract features with our LoRA-DINO backbone
        dino_features = self.backbone.extract_features(patch)
        
        # DINOv2 returns features of shape [B, N, C] where N is num_patches*num_patches
        # We need to reshape it to [B, C, H, W]
        bs, n_patches, feat_dim = dino_features.shape
        h_patches = w_patches = int(n_patches**0.5)
        dino_features_2d = dino_features.permute(0, 2, 1).reshape(bs, feat_dim, h_patches, w_patches)

        # 2. Create a NestedTensor for the features
        mask = torch.zeros(patch.shape[0], patch.shape[2], patch.shape[3], dtype=torch.bool, device=patch.device)
        nested_features = NestedTensor(dino_features_2d, mask)
        
        # 3. Generate positional embeddings dynamiquement selon la taille du patch
        _, _, h, w = dino_features_2d.shape
        pos_embed = build_position_encoding(self.hidden_dim, height=h, width=w, device=patch.device, dtype=patch.dtype)

        # 4. Project features to the transformer's hidden dimension
        src = self.input_proj[0](nested_features.tensors)
        
        # Deformable DETR expects multi-scale features. We only have one scale from DINO.
        # We will pass this single scale feature map.
        srcs = [src]
        masks = [mask]
        pos_embeds = [pos_embed]

        # 5. Pass to the Deformable Transformer
        # Passage à la tête custom : on utilise directement la méthode forward de la tête
        # On concatène les features multi-échelles si besoin (ici, un seul niveau)
        # La tête custom attend un tensor [seq_len, batch, features]
        # On adapte la forme si besoin
        # Ici, on suppose que srcs[0] est [B, C, H, W] -> on le met sous forme [H*W, B, C]
        b, c, h, w = srcs[0].shape
        memory = srcs[0].reshape(b, c, h * w).permute(2, 0, 1)  # [H*W, B, C]
        predictions = self.deformable_detr_head(memory)
        return predictions

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_patching:
            raise ValueError("Use forward_patch for patched training")
        x = self.resize_full(images)
        return self.forward_patch(x)

    def _compute_loss(self, predictions: Dict, targets: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        loss_dict = self.criterion(predictions, targets)
        weighted_losses = sum(
            self.criterion.weight_dict.get(k, 1.0) * v
            for k, v in loss_dict.items()
            if k in self.criterion.weight_dict
        )
        return weighted_losses, loss_dict

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        if not self.use_patching:
            predictions = self(images)
            device = predictions["pred_logits"].device
            targets = [{"labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]
            total_loss, loss_dict = self._compute_loss(predictions, targets)
            
            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_epoch=True, on_step=False, prog_bar=(k in ["loss_ce", "loss_bbox", "loss_giou"]), batch_size=len(images))
            self.log("train/loss", total_loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=len(images))
            return total_loss
        
        else:
            opt = self.optimizers()
            device = images.device
            losses = []
            
            for start_y, end_y in self.boundaries:
                for start_x, end_x in self.boundaries:
                    opt.zero_grad()
                    
                    patch = images[..., start_y:end_y, start_x:end_x]
                    patch_bounds = (start_y, end_y, start_x, end_x)
                    patch_targets = extract_patch_targets(targets, patch_bounds, self.img_size, self.patch_size)
                    patch_targets = [{"labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in patch_targets]
                    
                    predictions = self.forward_patch(patch)
                    total_loss, loss_dict = self._compute_loss(predictions, patch_targets)
                    
                    self.manual_backward(total_loss)
                    self.clip_gradients(opt, gradient_clip_val=0.1, gradient_clip_algorithm="norm")
                    opt.step()
                    
                    losses.extend([total_loss.detach()] * len(images))
            
            avg_loss = sum(losses) / len(losses)
            self.log("train/loss", avg_loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=len(images))
            return avg_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        device = images.device
        
        if not self.use_patching:
            predictions = self(images)
            targets = [{"labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]
            total_loss, _ = self._compute_loss(predictions, targets)
            self.log("val/loss", total_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(images))
            self._update_metrics(predictions, targets)
            return total_loss
        
        else:
            losses = []
            
            for start_y, end_y in self.boundaries:
                for start_x, end_x in self.boundaries:
                    patch = images[..., start_y:end_y, start_x:end_x]
                    patch_bounds = (start_y, end_y, start_x, end_x)
                    patch_targets = extract_patch_targets(targets, patch_bounds, self.img_size, self.patch_size)
                    patch_targets = [{"labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in patch_targets]
                    
                    predictions = self.forward_patch(patch)
                    total_loss, _ = self._compute_loss(predictions, patch_targets)
                    losses.extend([total_loss] * len(images))
                    self._update_metrics(predictions, patch_targets)
            
            avg_loss = sum(losses) / len(losses)
            self.log("val/loss", avg_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(images))
            return avg_loss

    def _update_metrics(self, predictions: Dict, targets: List[Dict]):
        pred_logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]

        for i in range(len(targets)):
            scores, labels = pred_logits[i].softmax(-1)[:, :-1].max(-1)
            keep = scores > 0.001
            
            if keep.sum() > 0:
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i][keep])
                pred_boxes_xyxy = pred_boxes_xyxy.clamp(0, 1)
                pred_entry = {
                    "boxes": pred_boxes_xyxy * self.patch_size,
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            else:
                device = pred_logits.device
                pred_entry = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                }

            gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[i]["boxes"])
            gt_boxes_xyxy = gt_boxes_xyxy.clamp(0, 1)
            gt_entry = {
                "boxes": gt_boxes_xyxy * self.patch_size,
                "labels": targets[i]["labels"],
            }
            self.val_metric.update([pred_entry], [gt_entry])

    def on_validation_epoch_end(self):
        metrics = self.val_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        lora_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                lora_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch]).float()
        targets = [b[1] for b in batch]
        return images, targets

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
