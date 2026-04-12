import lightning as L
from peft import LoraConfig,  get_peft_model

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from CenterNet.utils_centernet import decode_centernet

""" This file contains the model definitions for the CenterNet-based organoid detection.
- CenterNetHead : Detection head with heatmap, width-height and regression branches.
- DinoCenterNet : Backbone DINOv2/DINOv3 + FPN + CenterNetHead
- get_model : Factory function to instantiate the model with optional LoRA/DoRA applied to the backbone.
To use Dinov3, download the weights from the official repository and place them in the 'weights' directory. 
Then clone the distant repository locally in 'dinov3' to load the architecture. 
The code will automatically load the correct weights based on the specified model size.
"""

DINOV2_SIZE_MAP = {
    "small": ("dinov2_vits14", 384),
    "base": ("dinov2_vitb14", 768),
    "large": ("dinov2_vitl14", 1024),
    "giant": ("dinov2_vitg14", 1536),
}
DINOV3_SIZE_MAP = {
    "small": ("dinov3_vits16", 384),
    "smallplus": ("dinov3_vits16plus", 384),
    "base": ("dinov3_vitb16", 768),
    "large": ("dinov3_vitl16", 1024),
    "giant": ("dinov3_vit7b16", 1536),
}

REPO_DIR = 'dinov3'
WEIGHTS_DIR = 'weights'

# =============================================================================
# CenterNet Detection Head
# =============================================================================

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        # Heatmap : organoid's center probability
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, num_classes, 1) 
        )
        # WH : Width and Height of the object from the center
        self.wh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, 1)
        )
        # Offset : Correction for stride's resolution loss
        self.reg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, 1)
        )

        self.heatmap[-1].bias.data.fill_(-2.19) ## initialize to black

    def forward(self, x):
        return {
            "hm": self.heatmap(x).sigmoid(),
            "wh": self.wh(x),
            "reg": self.reg(x)
        }
    
# =============================================================================
#  DINOv2/DINOv3 + Neck + Head
# =============================================================================
#     

class DinoCenterNet(nn.Module):
    def __init__(self, name: str = "dinov2",model_size="base", out_channels=256, num_classes=1):
        super().__init__()
        # Backbone DINOv2 
        if name == "dinov2":
            size_map = DINOV2_SIZE_MAP
            if model_size not in size_map:
                raise ValueError(f"Invalid model size for DINOv2: {model_size}.")
            
            hub_name, num_features = size_map[model_size]
            self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        elif name == "dinov3":
            size_map = DINOV3_SIZE_MAP
            if model_size not in size_map:
                raise ValueError(f"Invalid model size for DINOv3: {model_size}.")
            
            hub_name, num_features = size_map[model_size]
            search_pattern = os.path.join(WEIGHTS_DIR, f"{hub_name}*.pth")
            files = glob.glob(search_pattern)
            
            if not files:
                raise ValueError(f"No weights file found for pattern '{hub_name}' in {WEIGHTS_DIR}.")
                
            checkpoint_path = files[0] # First file found
            self.model = torch.hub.load(
                REPO_DIR, 
                hub_name, 
                source='local', 
                pretrained = False
            )
            print(f"📦 Loading local weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict,strict=True)

        self.embed_dim = self.model.embed_dim 

        # we implement FPN for better performances
        if model_size == "base":
            self.target_layers = [3, 6, 9, 11]
        elif model_size == "large":
            self.target_layers = [7, 15, 20, 23]
        elif model_size == "giant":
            self.target_layers = [9, 19, 29, 39]

        self.fusion_weights = nn.Parameter(torch.ones(len(self.target_layers)))

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, out_channels, 1) for _ in range(len(self.target_layers))
        ])

        self.upsample = nn.Sequential(
            # 16x16 -> 32x32 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),

            # 32x32 -> 64x64 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        
        # Head
        self.head = CenterNetHead(in_channels=out_channels, num_classes=num_classes)

    def forward(self, x):
        layers = self.model.get_intermediate_layers(x, n=self.target_layers)
        
        B = x.shape[0]
        features = []

        for i, f in enumerate(layers):
            
            if f.dim() == 3:
                num_tokens = f.shape[1]
                h = w = int(num_tokens**0.5)
                patch_size = self.model.patch_size if hasattr(self.model, 'patch_size') else 16
                h, w = x.shape[2] // patch_size, x.shape[3] // patch_size

                expected_patches = h * w
                f = f[:, -expected_patches:, :]
                f = f.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)

            features.append(self.lateral_convs[i](f))

        weights = torch.relu(self.fusion_weights)
        weights = weights / (weights.sum() + 1e-4)

        combined_features = 0
        for i in range(len(features)):
            combined_features += features[i] * weights[i]

        f_up = self.upsample(combined_features)
        
        return self.head(f_up)
    
# =============================================================================
# Fonction Factory : get_model
# =============================================================================

def get_model(method: str = "lora", num_classes: int = 1, r: int = 16, name: str = "dinov2", model_size="base"):
    """
    Creates the DinoCenterNet model and applies PEFT if specified.
    """
    model = DinoCenterNet(name = name, model_size = model_size, out_channels=256, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False

    if method in ["lora", "dora"]:
        print(f"[PEFT] Applying {method.upper()} (r={r}) to {name}")
        target_modules = ["qkv", "proj", "fc1", "fc2"]
        config = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            use_dora=(method == "dora"),
            modules_to_save=["head", "upsample"],
        )
        model.model = get_peft_model(model.model, config)
    
    elif method == "frozen":
        print("[Info] Training only Head and Neck.")
        for param in model.head.parameters(): param.requires_grad = True
        for param in model.upsample.parameters(): param.requires_grad = True
        
    return model    

# =============================================================================
# LightningModule
# =============================================================================

class OrganoidDetectionModule(L.LightningModule):
    def __init__(self, name = "dinov2", model_size = "base", method="lora", num_classes=1, r=16, lr=1e-4, lr_backbone = 1e-5, patch_size = 224, stride = 3.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(method, num_classes, r, name=name,model_size=model_size)
        self.lr = lr
        self.lr_backbone = lr_backbone,
        self.val_metric = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",class_metrics=True)
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, batch_dict = batch 
        targets = batch_dict["targets"]
        output = self(images)
        mask = targets["reg_mask"].bool().squeeze(1) 
        num_objects = mask.sum()

        # We compute the different losses

        loss_hm = self._neg_loss(output["hm"], targets["hm"])

        if num_objects > 0:
            pred_wh = output["wh"].permute(0, 2, 3, 1)[mask] 
            target_wh = targets["wh"].permute(0, 2, 3, 1)[mask]
            
            diff = torch.abs(pred_wh - target_wh)
            loss_wh = (diff / (target_wh.detach() + 1e-6)).mean()
            
            pred_reg = output["reg"].permute(0, 2, 3, 1)[mask]
            target_reg = targets["reg"].permute(0, 2, 3, 1)[mask]
            loss_reg = F.l1_loss(pred_reg, target_reg, reduction='mean')
        else:
            loss_wh = 0.0
            loss_reg = 0.0
            
        total_loss = loss_hm + 0.5 * loss_wh + loss_reg
        
        self.log_dict({"train/loss": total_loss, "train/hm": loss_hm, "train/wh": loss_wh, "train/reg": loss_reg}, prog_bar=True)
        return total_loss
    
    def save_heatmap_debug(self, image, hm_pred, hm_gt, batch_idx, save_dir = "debug_plots"):
        """ Save Heatmap to understand what the model is learning during the first epochs.
        This function also serves as a performance indicator.
        """
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)

        pred = hm_pred[0, 0].detach().cpu().numpy()
        gt = hm_gt[0, 0].cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        
        ax[1].imshow(gt, cmap='jet')
        ax[1].set_title("Ground Truth HM")
        
        ax[2].imshow(pred, cmap='jet')
        ax[2].set_title(f"HM Prediction (Max: {pred.max():.2f})")

        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}.png")
        plt.savefig(file_path)
        plt.close(fig)
        print(f"Heatmap saved : {file_path}")

    def _calculate_iou(self, boxA, boxB):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.
        boxA : [x1, y1, x2, y2]
        boxB : [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # we prevent division by zero
        return iou
    
    def validation_step(self, batch, batch_idx):
        images, targets_dict = batch
        outputs = self(images)
        if batch_idx == 0: 
            self.save_heatmap_debug(images, outputs["hm"], targets_dict["targets"]["hm"], batch_idx)
        # We use decode_centernet to have predictions in the same format as the GT (list of boxes and labels) to compute mAP with torchmetrics.
        preds = decode_centernet(outputs, threshold=0.3, stride=self.stride, patch_size = self.patch_size)
        
        gts = [
            {"boxes": b, "labels": l} 
            for b, l in zip(targets_dict["gt_boxes"], targets_dict["gt_labels"])
        ]
        
        self.val_metric.update(preds, gts)
    
    def on_validation_epoch_end(self):
        mAP_metrics = self.val_metric.compute()
        self.log("val/mAP_50", mAP_metrics["map_50"], prog_bar=True)
        self.log("val/mAP", mAP_metrics["map"], prog_bar=True)
        self.val_metric.reset()

    def on_test_start(self):
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        images, targets_dict = batch
        outputs = self(images)
        
        preds = decode_centernet(outputs, threshold=0.3, stride=self.stride, patch_size = self.patch_size)
        gt_boxes = targets_dict["gt_boxes"]
        formatted_targets = []
        batch_stats = []
        if batch_idx % 10 == 0:
            self.save_heatmap_debug(
                images, 
                outputs["hm"], 
                targets_dict["targets"]["hm"], 
                batch_idx,
                save_dir = "test_plots"
            )

        for i, (p, g) in enumerate(zip(preds, gt_boxes)):
            if len(p["boxes"]) > 0 and len(g) > 0:
                p_area = (p["boxes"][:, 2] - p["boxes"][:, 0]) * (p["boxes"][:, 3] - p["boxes"][:, 1])
                g_area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
                
                batch_stats.append({
                    "mean_pred_area": p_area.mean().item(),
                    "mean_gt_area": g_area.mean().item(),
                    "num_preds": len(p["boxes"]),
                    "num_gts": len(g)
                })
            formatted_targets.append({
                "boxes": g,
                "labels": torch.zeros(len(g), dtype=torch.long, device=g.device) 
            })
        res = {
            "preds": preds,
            "gts": gt_boxes,
            "stats": batch_stats
        }
        
        self.test_step_outputs.append(res) 
        self.val_metric.update(preds, formatted_targets)
        return res

    def on_test_epoch_end(self): 
        """ 
        This step is useful to compute additional metrics that are not directly supported by torchmetrics, such as the average distance error between predicted and ground truth centers.
        We also save the predictions to a .pt file for further analysis and understanding of the errors.
        """
        mAP_metrics = self.val_metric.compute()
        self.log("test/mAP_50", mAP_metrics["map_50"])
        self.log("test/mAP", mAP_metrics["map"])

        all_pred_wh = []
        all_gt_wh = []
        all_dist_errors = [] 
        
        for batch_out in self.test_step_outputs:
            preds = batch_out["preds"]
            gts = batch_out["gts"]
            
            for img_idx in range(len(preds)):
                p_boxes = preds[img_idx]["boxes"] # [N, 4]
                g_boxes = gts[img_idx]           # [M, 4]
                
                if len(p_boxes) == 0 or len(g_boxes) == 0:
                    continue

                for gt_box in g_boxes:
                    gx, gy = (gt_box[0] + gt_box[2])/2, (gt_box[1] + gt_box[3])/2
                    gw, gh = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
                    
                    best_iou = 0
                    best_pred = None
                    
                    for p_box in p_boxes:
                        iou = self._calculate_iou(gt_box, p_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = p_box
                    
                    if best_pred is not None and best_iou > 0.3:
                        px, py = (best_pred[0] + best_pred[2])/2, (best_pred[1] + best_pred[3])/2
                        pw, ph = best_pred[2] - best_pred[0], best_pred[3] - best_pred[1]
                        
                        all_gt_wh.append([gw, gh])
                        all_pred_wh.append([pw, ph])
                        
                        dist = torch.sqrt(torch.tensor((gx-px)**2 + (gy-py)**2))
                        all_dist_errors.append(dist.item())

        avg_dist = sum(all_dist_errors) / len(all_dist_errors) if all_dist_errors else 0
        print(f"\n Average Center Error = {avg_dist:.2f} pixels")
        
        torch.save({
            "gt_wh": all_gt_wh,
            "pred_wh": all_pred_wh,
            "dist_errors": all_dist_errors
        }, "diagnostic_results.pt")
        
        self.val_metric.reset()
        
    def _neg_loss(self, pred, gt):
        """Focal Loss for Heatmaps"""
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    def configure_optimizers(self):
        """
        We create two parameter groups with different learning rates: one for the LoRA parameters (backbone) and one for the head and neck.
        This allows us to fine-tune the backbone with a smaller learning rate while training the head more aggressively.
        We also use a Cosine Annealing scheduler to reduce the learning rates over time, which often leads to better convergence in fine-tuning scenarios.
        """
        lora_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "dinov2" in name:
                lora_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_params, 
                    "lr": self.hparams.lr_backbone, # usually smaller (ex: 1e-5)
                    "name": "lora"
                },
                {
                    "params": head_params, 
                    "lr": self.hparams.lr,          # usually larger (ex: 1e-4)
                    "name": "head"
                },
            ],
            weight_decay=self.hparams.weight_decay if hasattr(self.hparams, 'weight_decay') else 1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Update at the end of each epoch
                "frequency": 1,
            },
        }