
import lightning as L
import torch
import torchmetrics
import torchvision
from typing import Dict, Tuple, List

from detection.lora import create_frozen_backbone
from detection.detr import SetCriterion, HungarianMatcher, TransformerDetectionHead
from detection.utils import box_cxcywh_to_xyxy, extract_patch_targets, get_boundaries


# =============================================================================
# 8. Detection Model with Frozen Backbone (Lightning Module)
# =============================================================================

class DetectionModel(L.LightningModule):
    """
    Object Detection model with frozen backbone and trainable DETR Detection Head.
    Only the detection head parameters are trained.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2",
        backbone_size: str = "small",
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_decoder_heads: int = 8,
        num_decoder_layers: int = 6,
        # Loss parameters
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        eos_coef: float = 0.1,
        # Optimizer parameters
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        # Patching parameters
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
        self.weight_decay = weight_decay
        self.use_patching = use_patching
        self.img_size = img_size
        self.patch_size = patch_size
        self.val_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox")

        if use_patching:
            self.boundaries, self.img_size = get_boundaries(num_patches, overlap_size, patch_size, img_size=self.img_size)
        else:
            self.img_size = patch_size
            self.boundaries = None

        # Frozen backbone (no LoRA, no gradient updates)
        self.backbone = create_frozen_backbone(
            name=backbone_name,
            model_size=backbone_size,
        )
        num_features = self.backbone.num_features

        # Detection head (fully trainable)
        self.detection_head = TransformerDetectionHead(
            num_features=num_features,
            num_queries=num_queries,
            num_decoder_heads=num_decoder_heads,
            num_decoder_layers=num_decoder_layers,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

        # Resize transforms for full images
        if not use_patching:
            self.resize_full = torchvision.transforms.Resize(
                (self.img_size, self.img_size), antialias=True
            )

        # Loss criterion for DETR
        matcher = HungarianMatcher(
            cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={
                "loss_ce": cost_class,
                "loss_bbox": cost_bbox,
                "loss_giou": cost_giou,
            },
            eos_coef=eos_coef,
            losses=["labels", "boxes", "cardinality"],
        )

    def forward_patch(self, patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a single patch."""
        features = self.backbone.extract_features(patch)
        memory = features.permute(1, 0, 2)
        predictions = self.detection_head(memory)
        return predictions

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for full images (when not using patching)."""
        if self.use_patching:
            raise ValueError("Use forward_patch for patched training")
        x = self.resize_full(images)
        return self.forward_patch(x)

    def _compute_loss(self, predictions: Dict, targets: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Compute the weighted sum of all losses."""
        loss_dict = self.criterion(predictions, targets)
        weighted_losses = sum(
            self.criterion.weight_dict.get(k, 1.0) * v
            for k, v in loss_dict.items()
            if k in self.criterion.weight_dict
        )
        return weighted_losses, loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step with optional patching. 
        This is custom implemented to handle patch-based training when use_patching=True.
        """
        images, targets = batch
        
        if not self.use_patching:
            predictions = self(images)
            device = predictions["pred_logits"].device
            targets = [
                {
                    "labels": t["labels"].to(device),
                    "boxes": t["boxes"].to(device),
                }
                for t in targets
            ]
            
            total_loss, loss_dict = self._compute_loss(predictions, targets)
            
            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_epoch=True, on_step=False,
                         prog_bar=(k in ["loss_ce", "loss_bbox", "loss_giou"]),
                         batch_size=len(images))
            self.log("train/loss", total_loss, on_epoch=True, on_step=True,
                     prog_bar=True, batch_size=len(images))
            return total_loss
        
        else:
            opt = self.optimizers()
            device = images.device
            losses = []
            
            for start_y, end_y in self.boundaries:
                for start_x, end_x in self.boundaries:
                    opt.zero_grad()
                    
                    patch = images[..., start_y:end_y, start_x:end_x]
                    #patch = images[..., start_x:end_x, start_y:end_y]
                    
                    patch_bounds = (start_y, end_y, start_x, end_x)
                    patch_targets = extract_patch_targets(
                        targets, patch_bounds, self.img_size, self.patch_size
                    )
                    
                    patch_targets = [
                        {
                            "labels": t["labels"].to(device),
                            "boxes": t["boxes"].to(device),
                        }
                        for t in patch_targets
                    ]
                    
                    predictions = self.forward_patch(patch)
                    total_loss, loss_dict = self._compute_loss(predictions, patch_targets)
                    
                    self.manual_backward(total_loss)
                    # manually clip gradients to prevent exploding gradients in patch-based training
                    self.clip_gradients(opt, gradient_clip_val=0.1, gradient_clip_algorithm="norm")
                    opt.step()
                    
                    losses.extend([total_loss.detach()] * len(images))
            
            avg_loss = sum(losses) / len(losses)
            self.log("train/loss", avg_loss, on_epoch=True, on_step=True,
                     prog_bar=True, batch_size=len(images))
            return avg_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with optional patching.
        This is custom implemented to handle patch-based validation when use_patching=True.
        It also updates mAP metrics at the end of each epoch.
        """
        images, targets = batch
        device = images.device
        
        if not self.use_patching:
            predictions = self(images)
            targets = [
                {
                    "labels": t["labels"].to(device),
                    "boxes": t["boxes"].to(device),
                }
                for t in targets
            ]

            total_loss, _ = self._compute_loss(predictions, targets)
            self.log("val/loss", total_loss, on_epoch=True, on_step=False,
                     prog_bar=True, batch_size=len(images))
            
            self._update_metrics(predictions, targets)
            return total_loss
        
        else:
            losses = []
            
            for start_y, end_y in self.boundaries:
                for start_x, end_x in self.boundaries:
                    patch = images[..., start_y:end_y, start_x:end_x]
                    #patch = images[..., start_x:end_x, start_y:end_y]
                    
                    patch_bounds = (start_y, end_y, start_x, end_x)
                    patch_targets = extract_patch_targets(
                        targets, patch_bounds, self.img_size, self.patch_size
                    )
                    
                    patch_targets = [
                        {
                            "labels": t["labels"].to(device),
                            "boxes": t["boxes"].to(device),
                        }
                        for t in patch_targets
                    ]
                    
                    predictions = self.forward_patch(patch)
                    total_loss, _ = self._compute_loss(predictions, patch_targets)
                    losses.extend([total_loss] * len(images))
                    
                    self._update_metrics(predictions, patch_targets)
            
            avg_loss = sum(losses) / len(losses)
            self.log("val/loss", avg_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(images))
            
            return avg_loss

    def _update_metrics(self, predictions: Dict, targets: List[Dict]):
        """Update mAP metrics."""
        pred_logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]

        for i in range(len(targets)):
            scores, labels = pred_logits[i].softmax(-1)[:, :-1].max(-1)
            keep = scores > 0.001  # Filter out low-confidence predictions
            
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
        """Compute and log mAP metrics at the end of validation epoch."""
        metrics = self.val_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        """Test step simply calls validation step to compute metrics on test set."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Compute and log mAP metrics at the end of test epoch."""
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        """Configure optimizer for detection head parameters (backbone is frozen)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch]).float()
        targets = [b[1] for b in batch]
        return images, targets
