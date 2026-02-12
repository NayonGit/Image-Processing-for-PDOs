import os
import torch
import numpy as np
import torchmetrics
import lightning as L
import torch.nn as nn
from torch import optim
from . import train_utils
import configs
from .segmenter import MaskTransformer
from torchvision.transforms import Resize
from .detection import TransformerDetection, SetCriterion, HungarianMatcher

def get_model(task: str) -> L.LightningModule:
    """
        Get the model based on the task.
        arguments:
            task [str]: the task to perform
        returns:
            model [L.LightningModule]: the model to use
    """
    if task == 'classification':
        model = ClassificationModel
    elif task == 'segmentation':
        model = SegmentationModel
    elif task in ['detection', 'localization']:
        model = DetectionModel
    else:
        raise ValueError(f'Task {task} not supported.')
    return model

def get_boundaries(num_patches, overlap_size, patch_size=224):
    img_size = patch_size*num_patches - overlap_size*(num_patches-1)
    start = 0
    boundaries = []
    for _ in range(num_patches):
        end = start+patch_size
        boundaries.append([start, end])
        start = end - overlap_size
    return boundaries, img_size

class ClassificationModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = 2,
                 precomputed: bool = False):
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params

        self.loss = torch.nn.BCELoss()
        if num_classes == 2:
            self.metric_f1 = torchmetrics.F1Score(task='binary')
            self.metric_auc = torchmetrics.AUROC(task='binary')
            self.foundation = None if precomputed else model
            self.model = nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())
        else:
            self.metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
            self.metric_auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)
            self.foundation = None if precomputed else model
            self.model = nn.Sequential(nn.LazyLinear(num_classes), nn.Softmax(dim=1))

    def on_train_epoch_start(self):
        self.train_labels = []
        self.train_preds = []
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_labels.append(y)
        self.train_preds.append(y_hat.detach())

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.train_labels = torch.cat(self.train_labels)
        self.train_preds = torch.cat(self.train_preds)
        metric_f1 = self.metric_f1(self.train_preds, self.train_labels)
        metric_auc = self.metric_auc(self.train_preds, self.train_labels)
        self.log('train/metric_f1', metric_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/metric_auc', metric_auc, on_epoch=True, on_step=False, prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_preds = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        self.val_labels = torch.cat(self.val_labels)
        self.val_preds = torch.cat(self.val_preds)
        metric_f1 = self.metric_f1(self.val_preds, self.val_labels)
        metric_auc = self.metric_auc(self.val_preds, self.val_labels)
        self.log('val/metric_f1', metric_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/metric_auc', metric_auc, on_epoch=True, on_step=False, prog_bar=True)
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_labels.append(y)
        self.val_preds.append(y_hat.detach())

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def forward(self, x):
        if self.foundation is not None:
            x = self.foundation.forward_pass(self.foundation, x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)

class SegmentationModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = None,
                 segmenter_config: dict = None,
                 num_patches: int = None,
                 img_size: tuple = None):
        super(SegmentationModel, self).__init__()
        self.automatic_optimization=False
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.prediction_head = MaskTransformer(**segmenter_config)
        self.prediction_head.mask_norm = nn.Identity()
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params
        self.boundaries, self.img_size = get_boundaries(num_patches, 30)
        self.loss = train_utils.dice_loss
        self.metric = train_utils.dice_metric
        self.resize = Resize((224, 224), antialias=False)

    def forward(self, x):
        device = x.device
        shape = x.shape[-2], x.shape[-1]
        with torch.no_grad():
            x = self.model.forward_pass(self.model, x, device)
        x = torch.nn.functional.sigmoid(self.prediction_head(x.to(device), shape))
        x = torch.nn.functional.interpolate(x, shape)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        losses = []
        for x_start, x_end in self.boundaries:
            for y_start, y_end in self.boundaries:
                opt.zero_grad()
                x_patch = x[..., x_start:x_end, y_start:y_end]
                y_patch = y[..., x_start:x_end, y_start:y_end]
                x_patch = self.resize(x_patch)
                y_patch = self.resize(y_patch)
                y_hat_patch = self(x_patch)

                loss = self.loss(y_hat_patch, y_patch)
                self.manual_backward(loss)
                opt.step()
                losses.extend([loss] * len(x))
        loss = sum(losses)/len(losses)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        if self.trainer.testing:
            pred_full = torch.zeros_like(y, dtype=torch.float, device=torch.device('cpu'))
            pred_norm = torch.zeros_like(y, dtype=torch.float, device=torch.device('cpu'))
        for x_start, x_end in self.boundaries:
            for y_start, y_end in self.boundaries:
                x_patch = x[..., x_start:x_end, y_start:y_end]
                y_patch = y[..., x_start:x_end, y_start:y_end]
                orig_shape = x_patch.shape[-2:]
                x_patch = self.resize(x_patch)
                y_patch = self.resize(y_patch)
                y_hat_patch = self(x_patch)
                loss = self.loss(y_hat_patch, y_patch)
                losses.extend([loss] * len(x))
                if self.trainer.testing:
                    y_hat_patch = Resize(orig_shape, antialias=False)(y_hat_patch)
                    pred_full[..., x_start:x_end, y_start:y_end] += y_hat_patch.detach().cpu()
                    pred_norm[..., x_start:x_end, y_start:y_end] += 1

        loss = sum(losses)/len(losses)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.trainer.testing:
            pred_full /= pred_norm
            metric = self.metric(pred_full.cpu(), y.int().cpu())
            self.log('val/metric_dice', metric, on_epoch=True, on_step=False, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)

class DetectionModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = None,
                 num_patches: int = None,
                 num_queries: int = None,
                 img_size: tuple = None,
                 num_decoder_heads: int = None,
                 num_decoder_layers: int = None,
                 dataset: str = None,
                 cost_giou: int = None,
                 eos_coef: float = None,
                 cost_class: int = None,
                 cost_bbox: int = None):
        super(DetectionModel, self).__init__()
        self.automatic_optimization=False
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        
        self.img_size = img_size
        self.dataset = dataset
        self.prediction_head = TransformerDetection(self.model.num_features,
                                                    num_queries,
                                                    num_decoder_heads,
                                                    num_decoder_layers,
                                                    num_classes)
        self.boundaries, self.img_size = get_boundaries(num_patches, 30)
        self.metric = torchmetrics.detection.MeanAveragePrecision(iou_type='bbox')
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params
        self.resize = Resize((224, 224), antialias=False)
        self.matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.loss = SetCriterion(num_classes, matcher=self.matcher, eos_coef=eos_coef, losses=['labels', 'boxes', 'cardinality'],
                         weight_dict={'loss_ce': 1, 'loss_bbox': cost_bbox, 'loss_giou': cost_giou})

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        losses = []
        for start_y, end_y in self.boundaries:
            for start_x, end_x in self.boundaries:
                opt.zero_grad()
                x_patch = x[..., start_x:end_x, start_y:end_y]
                patch_lower = torch.tensor([start_y, start_x, start_y, start_x]) / self.img_size
                patch_upper = torch.tensor([end_y, end_x, end_y, end_x]) / self.img_size
                lab_patch = []
                for lab in y:
                    valid_indices = torch.where(((lab['boxes'].cpu() > patch_lower) & (lab['boxes'].cpu() < patch_upper)).all(1))
                    if self.dataset == 'tellu':
                        lab_patch.append({'boxes': (lab['boxes'][valid_indices].cpu().float()*self.img_size-torch.tensor([start_y, start_x, start_y, start_x]))/224, 'labels': torch.ones_like(lab['labels'][valid_indices].cpu())})
                    else:
                        lab_patch.append({'boxes': (lab['boxes'][valid_indices].cpu().float()*self.img_size-torch.tensor([start_y, start_x, start_y, start_x]))/224, 'labels': lab['labels'][valid_indices].cpu()})
                y_hat_patch = self(x_patch)
                y_hat_patch = {key: value.cpu() for key, value in y_hat_patch.items()}
                patch_loss = self.loss(y_hat_patch, lab_patch)
                total_loss = 0
                for _, val in patch_loss.items():
                    total_loss += val
                self.manual_backward(total_loss)
                opt.step()
                losses.extend([total_loss] * len(x))
        
        loss = sum(losses)/len(losses)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = []
        map_label, map_pred = [], []
        for start_y, end_y in self.boundaries:
            for start_x, end_x in self.boundaries:
                x_patch = x[..., start_x:end_x, start_y:end_y]
                patch_lower = torch.tensor([start_y, start_x, start_y, start_x]) / self.img_size
                patch_upper = torch.tensor([end_y, end_x, end_y, end_x]) / self.img_size
                lab_patch = []
                for lab in y:
                    valid_indices = torch.where(((lab['boxes'].cpu() > patch_lower) & (lab['boxes'].cpu() < patch_upper)).all(1))
                    if self.dataset == 'tellu':
                        lab_patch.append({'boxes': (lab['boxes'][valid_indices].cpu().float()*self.img_size-torch.tensor([start_y, start_x, start_y, start_x]))/224, 'labels': torch.ones_like(lab['labels'][valid_indices].cpu())})
                    else:
                        lab_patch.append({'boxes': (lab['boxes'][valid_indices].cpu().float()*self.img_size-torch.tensor([start_y, start_x, start_y, start_x]))/224, 'labels': lab['labels'][valid_indices].cpu()})
                
                y_hat_patch = self(x_patch)
                y_hat_patch = {key: value.cpu() for key, value in y_hat_patch.items()}
                patch_loss = self.loss(y_hat_patch, lab_patch)
                total_loss = 0
                for _, val in patch_loss.items():
                    total_loss += val
                losses.extend([total_loss] * len(x))

                if self.trainer.testing:
                    match = self.matcher(y_hat_patch, lab_patch)
                    for idx, m in enumerate(match):
                        if len(lab_patch[idx]['labels']) > 0:
                            map_pred.append({'boxes': torch.stack([y_hat_patch['pred_boxes'][idx][i.item()] for i in m[0]]),
                                            'scores': torch.stack([nn.functional.softmax(y_hat_patch['pred_logits'][idx][i.item()], 0).max() for i in m[0]]),
                                            'labels': torch.stack([y_hat_patch['pred_logits'][idx][i.item()].argmax() for i in m[0]])})
                            map_label.append(lab_patch[idx])
        
        if self.trainer.testing:
            met = self.metric(map_pred, map_label)['map_50']
            self.log('val/metric_map50', met, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
        
        loss = sum(losses)/len(losses)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))

        return loss
    
    def forward(self, x):
        x = self.resize(x)
        device = x.device
        with torch.no_grad():
            x = self.model.forward_pass(self.model, x, device).permute(1, 0, 2)
        pred = self.prediction_head(x.to(device))
        return pred
    
    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch]).float()
        lbls = [b[1] for b in batch]
        return imgs, lbls

    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)