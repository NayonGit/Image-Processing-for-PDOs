import h5py
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import random
import torch
from torch.utils.data import DataLoader
import os
from typing import Optional, List

from CenterNet.data_centernet import OrganoidDetectionDataset, DATASET_INFO
from CenterNet.models_centernet import OrganoidDetectionModule
from CenterNet.utils_centernet import centernet_collate_fn

"""
This file contains the main functions to run training and testing for the CenterNet + DINOv2/DINOv3 project. 
It defines two main functions:
- run_experiment: for training the model with different PEFT methods (frozen, LoRA, DoRA) and various backbones (DINOv2, DINOv3).
- run_test: for evaluating a trained model on the test set.
patch_size should equal 224 for DINOv2 and 256 for DINOv3
downsample should equal 3.5 for DINOv2 and 4 for DINOv3
The reason is entirely geometric, as Dinov3 uses a patch size of 16x16, while Dinov2 uses 14x14. 
With a downsample of 4, Dinov3's feature maps will be perfectly aligned with the patch grid, while Dinov2's will be slightly off, hence the 3.5 to better align with the 14x14 patches.
"""
# =============================================================================
# Training
# =============================================================================

def run_experiment(name = "dinov2", model_size = "base",method="lora", dataset_name="tellu", r=8, lr=1e-4, lr_backbone=1e-5, 
                   batch_size=8, max_epochs=100, patch_size = 224, downsample =3.5,
                   resume_from_checkpoint=None):
    
    L.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    info = DATASET_INFO[dataset_name]
    with h5py.File(info["path"], "r") as hdf:
        all_img_names = list(hdf["train"]["images"].keys())

    # Split Train/Val
    random.shuffle(all_img_names)
    split_idx = int(0.85 * len(all_img_names))
    train_names = all_img_names[:split_idx]
    val_names = all_img_names[split_idx:]
    
    train_set = OrganoidDetectionDataset(dataset_name, split="train", img_names_filter=train_names, patch_size =patch_size, downsample = downsample)
    val_set = OrganoidDetectionDataset(dataset_name, split="train", img_names_filter=val_names, patch_size = patch_size, downsample = downsample)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=4, collate_fn=centernet_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                            num_workers=4, collate_fn=centernet_collate_fn)

    model = OrganoidDetectionModule(method=method, num_classes=train_set.num_classes, 
                                    r=r, lr=lr, lr_backbone=lr_backbone, name=name, model_size =model_size, patch_size = patch_size, stride = downsample)
    
    output_dir = f"{name}_{model_size}_centernet_models"
    checkpoint = ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"), 
                                 monitor="val/mAP_50", mode="max", 
                                 save_last=True, filename=f"best-{method}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=output_dir,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint, EarlyStopping(monitor="val/mAP_50", patience=15, mode="max")]
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    return trainer

# =============================================================================
# Testing
# =============================================================================

def run_test(ckpt_path: str, dataset_name: str = "tellu", batch_size: int = 4, 
             patch_size: int = 224, downsample: float = 3.5, num_workers: int = 4, seed: int = 42, **kwargs):
    L.seed_everything(seed)

    test_dataset = OrganoidDetectionDataset(
        dataset_name=dataset_name,
        split="test",
        patch_size=patch_size,
        downsample=downsample
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=centernet_collate_fn,
        pin_memory=True
    )

    print(f"[Info] Loading model from {ckpt_path}")
    model = OrganoidDetectionModule.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true"
    )

    print(f"[Data] Test set size: {len(test_dataset)}")
    test_metrics = trainer.test(model, test_loader)
    
    return test_metrics[0]