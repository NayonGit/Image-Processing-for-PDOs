import glob
import os
import torch
import torch.nn as nn
import torchvision.models as models
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional

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

REPO_DIR = '../models/dinov3'
WEIGHTS_DIR = '../models/weights'

""" 
This file contains the LoRA method for the backbone's creation.
It supports, for now, three models: DINOv2, DINOv3 and ResNet50.
The main function is `create_backbone_with_lora` which takes the backbone name, model size, PEFT method (lora, dora, none), LoRA hyperparameters,
and returns a backbone model with LoRA adapters applied. The target modules for LoRA are automatically detected based on the backbone architecture, but can also be customized.
"""

# =============================================================================
# ResNet Backbone for DETR
# =============================================================================

class ResNetBackbone(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        # We load the model with pretrained weights, but without the classification head
        full_resnet = getattr(models, model_name)(weights="DEFAULT")
        # We remove the classification head and the final global pooling
        self.body = nn.Sequential(*list(full_resnet.children())[:-2])
        self.num_features = full_resnet.fc.in_features  # 2048 for resnet50

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # body output (B, C, H, W)
        x = self.body(x)
        # We flatten the spatial dimensions: (B, C, H*W) then (B, H*W, C)
        # To simulate the token structure expected by DETR
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
    
# =============================================================================
# Backbone Factory with PEFT LoRA
# =============================================================================

def get_lora_target_modules(backbone_name: str, target_all: bool = False) -> List[str]:
    """
    Get the appropriate target modules for LoRA based on backbone architecture.
    
    Args:
        backbone_name: Name of the backbone ('dinov2', 'vit', etc.)
        target_all: If True, target all linear layers
    
    Returns:
        List of module name patterns to target with LoRA
    """
    # DINO uses fused QKV projections in attention blocks
    if "dino" in backbone_name:
        if target_all:
            return ["qkv", "proj", "fc1", "fc2"]  # Target all linear layers
        else:
            return ["qkv"]  # Target the fused QKV projection
    elif "rcnn" in backbone_name:
        if target_all:
            return ["conv1", "conv2", "conv3", "downsample.0"] # Target all conv layers in ResNet backbone
        else:
            return ["conv2"]  # Target attention-related layers (if applicable)
    else:
        raise ValueError(f"Unknown backbone for detecting LoRA target modules: {backbone_name}.")


def create_backbone_with_lora(
    name: str = "dinov2",
    model_size: str = "base",
    method: str = "lora",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    target_all: bool = False
) -> nn.Module:
    """
    Create a foundation model backbone with LoRA using PEFT library.
    
    Args:
        name: Backbone name ('dinov2', 'dinov3')
        model_size: Model size ('small', 'base', 'large', 'giant')
        method: PEFT method to use (lora, dora, none)
        lora_rank: LoRA rank (r parameter)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout for LoRA layers
        target_modules: Custom target modules (auto-detected if None)
        target_all: If True, target all linear layers instead of just attention projections
    
    Returns:
        PEFT model with LoRA adapters applied
    """
    if name == "dinov2":
        size_map = DINOV2_SIZE_MAP
        if model_size not in size_map:
            raise ValueError(f"Invalid model size for DINOv2: {model_size}.")
        
        hub_name, num_features = size_map[model_size]
        model = torch.hub.load("facebookresearch/dinov2", hub_name)
        model.num_features = num_features

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            out = self.forward(x, is_training=True)
            return out["x_norm_patchtokens"]

        model.extract_features = extract_features.__get__(model, type(model))
    
    elif name == "dinov3":
        size_map = DINOV3_SIZE_MAP
        if model_size not in size_map:
            raise ValueError(f"Invalid model size for DINOv3: {model_size}.")
        
        hub_name, num_features = size_map[model_size]
        search_pattern = os.path.join(WEIGHTS_DIR, f"{hub_name}.pth")
        files = glob.glob(search_pattern)
        
        if not files:
            raise ValueError(f"No weights file found for pattern '{hub_name}' in {WEIGHTS_DIR}.")
            
        checkpoint_path = files[0] # First file found
        model = torch.hub.load(
            REPO_DIR, 
            hub_name, 
            source='local', 
            weights=os.path.abspath(checkpoint_path)
        )
        model.num_features = num_features

        def extract_features(self, x: torch.Tensor) -> torch.Tensor:
            out = self.forward(x, is_training=True)
            return out["x_norm_patchtokens"]

        model.extract_features = extract_features.__get__(model, type(model))
    elif name =="rcnn":
        resnet_type = "resnet50"
        model = ResNetBackbone(model_name=resnet_type)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if method == "none":
        print(f"[Info] No PEFT adaptation applied to {name}. Freezing backbone.")
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    # Auto-detect target modules if not provided
    if target_modules is None:
        target_modules = get_lora_target_modules(name, target_all=target_all)
    
    # Configure LoRA using PEFT
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        use_dora = (method == "dora"),
        bias="none",  # Don't train biases otherwise 'lora_only' or 'all'
        task_type=TaskType.FEATURE_EXTRACTION  # We're using the backbone for feature extraction
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    print(f"[PEFT] Applied LoRA with config:")
    print(f"  - rank: {lora_rank}")
    print(f"  - alpha: {lora_alpha}")
    print(f"  - dropout: {lora_dropout}")
    print(f"  - target_modules: {target_modules}")
    model.print_trainable_parameters()
    
    return model