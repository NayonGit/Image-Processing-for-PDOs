import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from detection_lora.data import OrganoidDetectionDataset
from detection_lora.dino_detection import LoRADINOModel


def train(
    dataset_name: str = "tellu",
    h5_path: str | None = None,
    train_val_split: float = 0.85,
    backbone_name: str = "dinov2",
    backbone_size: str = "small",
    num_classes: int | None = None,
    # LoRA
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    # train
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    batch_size: int = 2,
    max_epochs: int = 100,
    patience: int = 15,
    num_workers: int = 4,
    seed: int = 42,
    output_dir: str = "./checkpoints_dino_lora",
    resume_from_checkpoint: Optional[str] = None,
):
    L.seed_everything(seed)

    full_dataset = OrganoidDetectionDataset(dataset_name=dataset_name, split="train", h5_path=h5_path)
    if num_classes is None:
        num_classes = full_dataset.num_classes
        print(f"[Info] Inferred num_classes={num_classes}")

    train_size = int(len(full_dataset) * train_val_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    model = LoRADINOModel(
        backbone_name=backbone_name,
        backbone_size=backbone_size,
        num_classes=num_classes,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lr=lr,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=LoRADINOModel.collate_fn, pin_memory=True, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=LoRADINOModel.collate_fn, pin_memory=True, persistent_workers=num_workers > 0
    )

    callbacks = [
        EarlyStopping(monitor="val/mAP_50", patience=patience, mode="max"),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch}-{val/mAP_50:.4f}",
            monitor="val/mAP_50",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        default_root_dir=output_dir,
        callbacks=callbacks,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=1,
        gradient_clip_val=0.1,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    trainer.test(model, val_loader, ckpt_path=callbacks[1].best_model_path)
    return trainer.callback_metrics