import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

def decode_centernet(output, K=100, threshold=0.3, stride=3.5, patch_size=224):
    """
    Transforms the CenterNet's outputs in boxes [x1, y1, x2, y2].
    
    Args:
        output (dict): Dictionary containing 'hm', 'wh', 'reg' (Tensors [B, C, H, W])
        K (int): Maximum number of objects to extract per image.
        threshold (float): Minimum confidence score.
        stride (int): Downsampling Factor of the model.
    """
    hm = output["hm"]
    wh = output["wh"]
    reg = output["reg"]
    
    batch, cat, height, width = hm.size()

    # Local NMS : Keep the local peaks in the heatmap 
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep

    # Extract Top-K scores and their indices
    scores, inds = torch.topk(hm.view(batch, -1), K)
    
    # Calculate the coordinates of the Top-K indices 
    clses = (inds // (height * width)).int()
    inds = inds % (height * width)
    ys = (inds // width).int()
    xs = (inds % width).int()

    # Retrieve the offsets (reg) and sizes (wh) corresponding to the Top-K indices
    # We permute to have [B, H*W, C] and can then index with the Top-K
    reg = reg.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
    wh = wh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
    
    batch_inds = torch.arange(batch, device=inds.device).unsqueeze(1).repeat(1, K)
    reg_vals = reg[batch_inds, inds] # [B, K, 2]
    wh_vals = wh[batch_inds, inds]   # [B, K, 2]

    # We adjust the centers (xs, ys) with the predicted offsets (reg_vals) and re-scale to the original image space using the stride
    final_xs = (xs.view(batch, K, 1) + reg_vals[:, :, 0:1]) * stride
    final_ys = (ys.view(batch, K, 1) + reg_vals[:, :, 1:2]) * stride

    # Bounding Boxes [x1, y1, x2, y2]
    w, h = wh_vals[:, :, 0:1] * patch_size, wh_vals[:, :, 1:2] * patch_size
    bboxes = torch.cat([
        final_xs - w/2, # x1
        final_ys - h/2, # y1
        final_xs + w/2, # x2
        final_ys + h/2  # y2
    ], dim=2)

    results = []
    for b in range(batch):
        mask = scores[b] > threshold
        results.append({
            "boxes": bboxes[b][mask],
            "scores": scores[b][mask],
            "labels": clses[b][mask]
        })
        
    return results

def centernet_collate_fn(batch):
    """
    Gather the images and heatmaps into tensors, 
    while keeping the GT boxes and labels as lists.
    """
    images = torch.stack([item[0] for item in batch])
    
    # we stack CenterNet's targets (hm, wh, reg, reg_mask) 
    targets = {
        k: torch.stack([item[1]["targets"][k] for item in batch])
        for k in batch[0][1]["targets"].keys()
    }
    
    gt_boxes = [item[1]["gt_boxes"] for item in batch]
    gt_labels = [item[1]["gt_labels"] for item in batch]
    
    return images, {
        "targets": targets,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels
    }

