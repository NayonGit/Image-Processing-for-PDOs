import glob
import os
import torch
import torch.nn as nn
from typing import Optional

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


# =============================================================================
# 7. Backbone Factory with Frozen Backbone
# =============================================================================

def create_frozen_backbone(
    name: str = "dinov2",
    model_size: str = "base",
) -> nn.Module:
    """
    Create a foundation model backbone with all parameters frozen.
    Only the detection head will be trainable.
    
    Args:
        name: Backbone name ('dinov2', 'dinov3')
        model_size: Model size ('small', 'base', 'large', 'giant')
    
    Returns:
        Backbone model with all parameters frozen (requires_grad=False)
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

    else:
        raise ValueError(f"Unknown backbone: {name}")

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"[Backbone] Created frozen {name} ({model_size}) with {num_features} features")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Backbone] Total parameters: {total_params:,}")
    
    return model