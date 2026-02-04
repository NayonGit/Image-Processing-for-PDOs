from torch.nn import Module
from configs import default_data_dir

def box_cxcywh_to_xyxy(boxes):
    import torch
    cx, cy, w, h = boxes.unbind(1)
    return torch.stack([
        cx - 0.2 * w,
        cy - 0.2 * h,
        cx + 0.2 * w,
        cy + 0.2 * h,
    ], dim=1)

def train(foundation: Module,
          dataset_name: str,
          split: int,
          download: bool,
          download_dir: str = '/tmp',
          data_dir: str = default_data_dir,
          num_splits: int = 4,
          task_specific: dict = None,
          **kwargs) -> None:
    """
        Train a model on a specified dataset.
        Args:
            foundation [torch.nn.Module | L.LightningModule]: model to use for the feature extraction
            dataset_name [str]: name of the dataset to train on
            split [int]: index of the split to train on
            download_dir [str]: directory to download the dataset to
            download [bool]: whether to download the dataset
            data_dir [str]: directory to save the prepared dataset to
            num_splits [int]: number of splits to create
            task_specific [dict]: task specific_parameters
            kwargs: additional arguments to update the training configuration
        
        Returns:
            None
    """
    from datasets import get_dataset_split
    from configs import train_config, dataset_config
    import torch
    import lightning as L
    from .models import get_model
    from torch.utils.data import DataLoader
    from datasets import get_dataset
    from datasets import prepare_dataset
    from datasets import PrecomputedDataset
    from .train_utils import update_easydict, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torchvision.utils import save_image, draw_bounding_boxes
    import os

    local_root = os.path.expanduser('~/dino_outputs')  # crée ~/dino_outputs
    os.makedirs(local_root, exist_ok=True)

    updated_train_config = update_easydict(train_config, kwargs)
    seed_everything(updated_train_config.seed)

    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    model = get_model(dataset_config[dataset_name].task)(foundation,
                                                         updated_train_config.optimizer,
                                                         updated_train_config.optimizer_params,
                                                         dataset_config[dataset_name].num_classes,
                                                         **task_specific)

    img_size = getattr(model, 'img_size', dataset_config[dataset_name].size)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    if download:
        prepare_dataset(dataset_name=dataset_name, download_dir=download_dir, data_dir=data_dir)
    train_names, val_names, test_names = get_dataset_split(dataset_name=dataset_name,
                                                           split=split,
                                                           num_splits=num_splits,
                                                           data_dir=data_dir,
                                                           seed=updated_train_config.seed)

    train_dataset = get_dataset(name=dataset_name,
                                data_dir=data_dir,
                                names=train_names,
                                download=False,
                                mode='train',
                                img_size=img_size)
    val_dataset = get_dataset(name=dataset_name,
                              data_dir=data_dir,
                              names=val_names,
                              download=False,
                              mode='train',
                              img_size=img_size)
    test_dataset = get_dataset(name=dataset_name,
                               data_dir=data_dir,
                               names=test_names,
                               download=False,
                               mode='test',
                               img_size=img_size) if test_names is not None else None

    if ('precomputed' in task_specific) and (task_specific['precomputed']):
       train_dataset = PrecomputedDataset(train_dataset, foundation)
       val_dataset = PrecomputedDataset(val_dataset, foundation)
       test_dataset = PrecomputedDataset(test_dataset, foundation)

    collate_fn = getattr(model, 'collate_fn', None)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=updated_train_config.batch_size,
                                shuffle=True,
                                num_workers=updated_train_config.num_workers,
                                collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=updated_train_config.batch_size,
                                shuffle=False,
                                num_workers=updated_train_config.num_workers,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=updated_train_config.batch_size,
                                shuffle=False,
                                num_workers=updated_train_config.num_workers,
                                collate_fn=collate_fn) if test_names is not None else None

    early_stopping = EarlyStopping(monitor='val/loss',
                                patience=updated_train_config.patience)
    model_checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_last=False, save_top_k=1)
    callbacks = [early_stopping, model_checkpoint]
    trainer = L.Trainer(max_epochs=updated_train_config.max_epochs,
                        default_root_dir='local_root', 
                        callbacks=callbacks,
                        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                        devices=1,
                        log_every_n_steps=1)
    trainer.fit(model,
                train_dataloader,
                val_dataloader)

    trainer.test(model,
                    test_dataloader if test_names is not None else val_dataloader,
                    ckpt_path=model_checkpoint.best_model_path, weights_only=False)

    img, label = next(iter(test_dataloader if test_names is not None else val_dataloader))
    img, label = next(iter(test_dataloader if test_names is not None else val_dataloader))
    device = next(model.parameters()).device
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
    pred_logits = output["pred_logits"][0]
    pred_boxes = output["pred_boxes"][0]
    probs = pred_logits.softmax(dim=-1)
    scores, labels = probs.max(dim=-1)

# seuil de confiance
    labels &= scores > 0.7
    print(labels)

    scores = scores[labels]
    labels = labels[labels]
    boxes = pred_boxes[labels]
    boxes = box_cxcywh_to_xyxy(boxes)
    img_vis = (img[0].cpu() * 255).to(torch.uint8)
    _, H, W = img_vis.shape

    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H
    boxes = boxes.clamp(min=0)

    text_labels = [
        f"cls {lbl.item()} | {score:.2f}"
        for lbl, score in zip(labels, scores)
    ]
    img_annotated = draw_bounding_boxes(
        img_vis,
        boxes,
        labels=text_labels,
        colors="red",
        width=2,
        font_size=16)
    save_path = f"{dataset_name}_detr_prediction.png"
    save_image(img_annotated.float() / 255.0, save_path)
    return {
        "trainer": trainer,
        "model": model,
        "metrics": trainer.callback_metrics,
        "checkpoint": model_checkpoint.best_model_path,
    }