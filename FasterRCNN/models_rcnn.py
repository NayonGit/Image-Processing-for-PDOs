import lightning as L
from peft import LoraConfig,  get_peft_model

import timm

import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN, _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

""" This file contains the model definitions based on Faster R-CNN with different backbones and PEFT methods.
- FasterRCNN with ResNet50-FPN backbone (torchvision) + optional LoRA/DoRA applied to the backbone
- FasterRCNN with DINOv2-Large backbone + FPN + optional LoRA/DoRA applied to the backbone
- FasterRCNN with Swinv2-B backbone + FPN + optional LoRA/DoRA applied to the backbone
- FasterRCNN with ConvNeXtV2-Base backbone + FPN + optional LoRA/DoRA applied to the backbone
- FasterRCNN with ConvNeXtV2-Huge backbone + FPN + optional LoRA/DoRA applied to the backbone (commenté pour l'instant car très lourd)
All of these models come with their own get_model_xxx() factory function that takes care of loading the pretrained weights, modifying the detection heads, and applying PEFT if specified.
The OrganoidDetectionModule class is a LightningModule wrapper that can be instantiated with any of these backbones and PEFT methods, and implements the training/validation/test steps and optimizer configuration.
"""
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

class DinoViT_FPN_Backbone(nn.Module):
    def __init__(self, model_type="dinov2_vitl14", out_channels=256):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_type)
        self.embed_dim = self.dinov2.embed_dim
        
        self.target_layers = [4, 10, 16, 22] 
        
        
        # FPN to convert DINOv2 features to a format compatible with Faster R-CNN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[self.embed_dim] * 4,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        
        self.out_channels = out_channels

    def forward(self, x):
        # x shape: [B, 3, H, W]
        B, _, H, W = x.shape
        h, w = H // 14, W // 14
        num_patches = h * w

        # Intermediate features extraction
        layers = self.dinov2.get_intermediate_layers(x, n=self.target_layers)
        
        # Reshape Tokens -> 2D Feature Maps
        features = {}
        for i, f in enumerate(layers):
            # ignore the cls token and reshape to [B, C, H', W']
            f = f[:, -num_patches:, :].permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
            features[str(i)] = f
            
        return self.fpn(features)

def get_model_dinov2(method: str = "lora", num_classes: int = 1, r: int = 16):
    """
    Faster R-CNN with a DINOv2-Large Backbone and LoRA/DoRA applied to the backbone.
    """
    backbone = DinoViT_FPN_Backbone(model_type="dinov2_vitl14", out_channels=256)
    
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1, # +1 for background
        rpn_anchor_generator=anchor_generator,
        min_size=224, max_size=224 
    )

    if method in ["lora", "dora"]:
        print(f"[PEFT] Applying {method.upper()} to DINOv2-L Backbone")
        config = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules=["qkv", "proj"], 
            lora_dropout=0.1,
            use_dora=(method == "dora"),
            modules_to_save=["roi_heads", "rpn", "fpn"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
    return model

class SwinBackbone(nn.Module):
    def __init__(self, model, in_channels_list, out_channels):
        super().__init__()
        self.body = nn.ModuleDict({
            "0": model.features[0:2],
            "1": model.features[2:4],
            "2": model.features[4:6],
            "3": model.features[6:8],
        })
        self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        res = {}
        for name, module in self.body.items():
            x = module(x)
            res[name] = x.permute(0, 3, 1, 2)
        return self.fpn(res)

def get_model_swin(num_classes: int = 1, method: str = "lora", r: int = 16):
    backbone_model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
    
    in_channels_list = [128, 256, 512, 1024]
    out_channels = 256
    
    backbone = SwinBackbone(backbone_model, in_channels_list, out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "pool"], 
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(backbone, num_classes=num_classes+1, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    if method in ["lora", "dora"]:
        print(f"[PEFT] Applying {method.upper()} to Swin-B")
        config = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules=["qkv", "proj"], 
            lora_dropout=0.1,
            modules_to_save=["roi_heads", "rpn", "fpn"],
        )
        model = get_peft_model(model, config)
        
    return model

class ConvNeXtBackbone(nn.Module):
        def __init__(self, model, in_channels_list, out_channels):
            super().__init__()
            self.body = model
            self.fpn = torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool(),
            )
            self.out_channels = out_channels

        def forward(self, x):
            fpn_input = {str(i): f for i, f in enumerate(self.body(x))}
            return self.fpn(fpn_input)
        
def get_model_convnext_v2(num_classes: int = 1, method: str = "lora", r: int = 16):

    backbone_model = timm.create_model(
        'convnextv2_base.fcmae_ft_in22k_in1k_384', 
        pretrained=True, 
        features_only=True
    )
    # backbone_model = timm.create_model(
    #     'convnextv2_huge.fcmae_ft_in22k_in1k_512', 
    #     pretrained=True, 
    #     features_only=True
    # )

    in_channels_list = backbone_model.feature_info.channels() 
    out_channels = 256

    backbone = ConvNeXtBackbone(backbone_model, in_channels_list, out_channels)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"],
            output_size=7, sampling_ratio=2
        )
    )

    if method == "lora":
        config = LoraConfig(
            r=r, lora_alpha=r*2,
            target_modules=["mlp.fc1", "mlp.fc2"], 
            modules_to_save=["roi_heads", "rpn", "fpn"]
        )
        model = get_peft_model(model, config)
        
    return model

class OrganoidDetectionModule(L.LightningModule):

    def __init__(self, backbone = "rcnn", method="lora", num_classes=1, r=16, lr=1e-4):

        super().__init__()
        self.save_hyperparameters()
        if backbone == "rcnn":
            self.model = get_model(method, num_classes, r)
        elif backbone == "dinov2":
            self.model = get_model_dinov2(method, num_classes, r)
        elif backbone == "swin":
            self.model = get_model_swin(method=method, num_classes=num_classes, r=r)
        elif backbone == "convnextv2":
            self.model = get_model_convnext_v2(method=method, num_classes=num_classes,r=r )
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

        if "map_per_class" in mAP and mAP["map_per_class"].ndim > 0:
            for i, val in enumerate(mAP["map_per_class"]):
                self.log(f"val/class_{i+1}_mAP", val)

        self.val_metric.reset()
    def test_step(self, batch, batch_idx):
        # Same logic as validation
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        # Same logic as validation
        return self.on_validation_epoch_end()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)