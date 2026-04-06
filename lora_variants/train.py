import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os

from lora_variants.data_rcnn import OrganoidDetectionDataset
from lora_variants.models import OrganoidDetectionModule

def run_experiment(method="lora", dataset_name="tellu", r=16, batch_size=4, max_epochs=50, output_dir="rcnn_models"):
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

    model = OrganoidDetectionModule(method=method, num_classes=dataset.num_classes, r=r)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"), monitor="val/mAP_50", mode="max", filename=f"best-{method}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint, EarlyStopping(monitor="val/mAP_50", patience=10, mode="max")]
    )

    trainer.fit(model, train_loader, val_loader)
    return trainer