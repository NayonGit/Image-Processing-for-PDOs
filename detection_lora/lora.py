import glob
import os
import torch
import torch.nn as nn
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


# =============================================================================
# 7. Backbone Factory with PEFT LoRA
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
    else:
        raise ValueError(f"Unknown backbone for detecting LoRA target modules: {backbone_name}.")


def create_backbone_with_lora(
    name: str = "dinov2",
    model_size: str = "base",
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

    else:
        raise ValueError(f"Unknown backbone: {name}")

    # Auto-detect target modules if not provided
    if target_modules is None:
        target_modules = get_lora_target_modules(name, target_all=target_all)
    
    # Configure LoRA using PEFT
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
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