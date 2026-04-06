import lightning as L
from peft import LoraConfig,  get_peft_model

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def get_model(method: str = "lora", num_classes: int = 1, r: int = 16):
    """
    Model Factory for Faster R-CNN with various PEFT methods.
    Args:
        method: One of ['full', 'lora', 'dora']
        num_classes: Number of foreground classes (background is auto-added)
        r: Rank for LoRA/DoRA
    """
    
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Torchvision Faster R-CNN uses (num_classes + 1) to account for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1
    )

    if method == "full":
        print("[Info] Method: Full Fine-Tuning. All parameters trainable.")
        return model

    # We focus on the 1x1 convolutions (conv1, conv3) and 3x3 (conv2)
    target_modules = ["conv1", "conv2", "conv3", "downsample.0"]

    if method in ["lora", "dora"]:
        print(f"[PEFT] Applying {method.upper()} (r={r})")
        config = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            use_dora=(method == "dora"),
            # We must keep the detection heads trainable
            modules_to_save=["roi_heads", "rpn"],
        )
        model = get_peft_model(model, config)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['full', 'lora', 'dora']")

    # Display trainable parameters for verification
    model.print_trainable_parameters()
    
    return model

class OrganoidDetectionModule(L.LightningModule):

    def __init__(self, method="lora", num_classes=1, r=16, lr=1e-4):

        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(method, num_classes, r)
        self.val_metric = MeanAveragePrecision(iou_type="bbox")

    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):

        images, targets = batch
        # Faster R-CNN returns a dict of losses in training mode
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        images, targets = batch
        outputs = self.model(images)
        self.val_metric.update(outputs, targets)

    def on_validation_epoch_end(self):

        mAP = self.val_metric.compute()
        self.log("val/mAP_50", mAP["map_50"], prog_bar=True)
        self.log("val/mAP_75", mAP["map_75"], prog_bar=True)
        self.log("val/mAP", mAP["map"], prog_bar=True)
        self.val_metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)