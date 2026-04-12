import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os

from FasterRCNN.data_rcnn import OrganoidDetectionDataset
from FasterRCNN.models_rcnn import OrganoidDetectionModule        

"""
This file contains the main functions to run training and testing for the Faster R-CNN + various backbones project.
It defines two main functions:
- run_train: for training the model with different PEFT methods (frozen, LoRA, DoRA) and various backbones (ConvNeXt-V2, SwinV2, DINOv2, ResNet50).
- run_test: for evaluating a trained model on the test set.
"""
# =============================================================================
# Training
# =============================================================================

def run_train(backbone = "rcnn",
              method="lora", 
              dataset_name="tellu", 
              r=16, 
              batch_size=4, 
              max_epochs=50, 
              output_dir="rcnn_models", 
              resume_from = None):
    
    L.seed_everything(42)

    # Dataset setup
    dataset = OrganoidDetectionDataset(dataset_name=dataset_name, split="train")
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # DataLoaders (collate_fn for Faster R-CNN lists)
    def collate_fn(batch): return tuple(zip(*batch))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                            num_workers=4, collate_fn=collate_fn)
    
    model = OrganoidDetectionModule(backbone = backbone, method=method, num_classes=dataset.num_classes, r=r)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"), monitor="val/mAP_50", mode="max", filename=f"best-{method}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint, EarlyStopping(monitor="val/mAP_50", patience=10, mode="max")]
    )

    if resume_from:
        print(f"[Info] Resuming training from: {resume_from}")
    else:
        print(f"[Info] Starting fresh training run")

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)
    return trainer

# =============================================================================
# Testing
# =============================================================================

def run_test(
    ckpt_path: str,
    backbone = "rcnn",
    method = "lora",
    dataset_name = "tellu",
    r = 8,
    num_classes = 2,
    batch_size = 4,
    max_epochs = 50,
    output_dir = "rcnn_models/test",
    seed = 42
):
    L.seed_everything(seed)

    test_dataset = OrganoidDetectionDataset(
        dataset_name=dataset_name,
        split="test",
    )
    
    if num_classes is None:
        num_classes = test_dataset.num_classes
    
    print(f"[Data] Test: {len(test_dataset)}")
    print(f"[Info] Inferred num_classes={num_classes} from dataset.")

    model = OrganoidDetectionModule.load_from_checkpoint(ckpt_path)
    print(f"[Info] Loaded model with backbone: {backbone} and method: {method}")

    def collate_fn(batch): return tuple(zip(*batch))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
    )

    test_metrics = trainer.test(model, test_loader, ckpt_path=ckpt_path)
    
    return test_metrics