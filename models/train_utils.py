import os
import torch
import lightning as L
from easydict import EasyDict as edict

def seed_everything(seed: int):
    """
        Seed everything for reproducibility.
        arguments:
            seed [int]: random seed
        returns:
            None
    """
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def dice_loss(y_hat: torch.tensor,
              y: torch.tensor) -> torch.tensor:
    """
        Compute the Dice loss for segmentation.
        arguments:
            y_hat [torch.tensor]: predicted mask
            y [torch.tensor]: ground truth mask
        returns:
            dice_loss [torch.tensor]: computed Dice loss
    """
    y_hat = y_hat.flatten()
    y = y.flatten()
    intersection = (y_hat * y).sum()
    dice_loss = 1 - ((2. * intersection) / (y_hat.sum() + y.sum()))
    return dice_loss

def dice_metric(y_hat: torch.tensor,
                y: torch.tensor) -> torch.tensor:
    """
        Compute the Dice metric for segmentation.
        arguments:
            y_hat [torch.tensor]: predicted mask
            y [torch.tensor]: ground truth mask
        returns:
            dice [torch.tensor]: computed Dice metric
    """
    y_hat = y_hat.flatten() > 0.5
    y = y.flatten()
    intersection = (y_hat * y).sum()
    dice = (2. * intersection) / (y_hat.sum() + y.sum())
    return dice

def update_easydict(easydict_obj, kwargs):
    """
    Updates values in an EasyDict object with values from kwargs.
        arguments:
            easydict_obj [edict]: EasyDict object to update
            kwargs [dict]: dictionary of key-value pairs to update
        reuturns:
            updated_easydict [edict]: updated EasyDict object
    """
    for key, value in kwargs.items():
        keys = key.split('.')
        d = easydict_obj
        for subkey in keys[:-1]:
            if subkey not in d:
                d[subkey] = edict()
            d = d[subkey]
        d[keys[-1]] = value
    return easydict_obj