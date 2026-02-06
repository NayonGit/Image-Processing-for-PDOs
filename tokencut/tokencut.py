"""
Main functions for applying Normalized Cut.
From LOST: https://github.com/valeoai/LOST and TokenCut: https://github.com/YangtaoWANG95/TokenCut
"""
import cv2
import math
import numpy as np
import os
import scipy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from scipy.linalg import eigh
from scipy import ndimage


def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    feats = feats[0,1:,:]
    # added
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) 
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox(es)
    # use first two dimensions (H,W) of img_raw.shape (works with (H,W) or (H,W,C))
    hw = tuple(init_image_size[0:2]) if hasattr(init_image_size, "__len__") and len(init_image_size) >= 2 else None
    preds, pred_feats, objects, cc_masks = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=hw, principle_object=False)
    # build a combined mask of all components
    mask = np.zeros(dims)
    for m in cc_masks:
        mask[m[0], m[1]] = 1

    return np.asarray(preds), objects, mask, seed, None, eigenvec.reshape(dims)

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract box(es) corresponding to connected components in the bipartition.
    If principle_object==True returns single box corresponding to seed (backward compatible).
    If principle_object==False returns list of boxes for each connected component.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition) 

    if principle_object:
        cc = objects[np.unravel_index(seed, dims)]
        mask = np.where(objects == cc)
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        pred_feats = [ymin, xmin, ymax, xmax]
        return pred, pred_feats, objects, mask
    else:
        preds = []
        pred_feats_list = []
        masks = []
        # iterate labels 1..num_objects
        for lbl in range(1, num_objects + 1):
            mask_idx = np.where(objects == lbl)
            if mask_idx[0].size == 0:
                continue
            ymin, ymax = int(mask_idx[0].min()), int(mask_idx[0].max()) + 1
            xmin, xmax = int(mask_idx[1].min()), int(mask_idx[1].max()) + 1
            r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
            r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
            pred = [r_xmin, r_ymin, r_xmax, r_ymax]
            if initial_im_size:
                pred[2] = min(pred[2], initial_im_size[1])
                pred[3] = min(pred[3], initial_im_size[0])
            preds.append(pred)
            pred_feats_list.append([ymin, xmin, ymax, xmax])
            masks.append((mask_idx[0], mask_idx[1]))
        return preds, pred_feats_list, objects, masks

def visualize_predictions(img, pred, vis_folder, im_name, save=False):
    """
    Visualization of predicted box(es).
    Accepts single box [x1,y1,x2,y2] or list/array of boxes.
    """
    image = np.copy(img)
    # multiple boxes
    if hasattr(pred, "__len__") and len(pred) > 0 and isinstance(pred[0], (list, tuple, np.ndarray)):
        for p in pred:
            cv2.rectangle(image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 0), 3)
    else:
        # single box
        p = pred
        cv2.rectangle(image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 0), 3)

    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_pred.jpg"
        Image.fromarray(image).save(pltname)

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title("Predicted box(es)")
    plt.axis('off')
    plt.show()

    return image
  
def visualize_predictions_gt(img, pred, gt, vis_folder, im_name, dim, scales, save=False):
    """
    Visualization of predicted box(es) and ground-truth boxes.
    pred can be a single box or a list/array of boxes.
    """
    image = np.copy(img)
    # draw predicted box(es)
    if hasattr(pred, "__len__") and len(pred) > 0 and isinstance(pred[0], (list, tuple, np.ndarray)):
        for p in pred:
            cv2.rectangle(image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 0), 3)
    else:
        p = pred
        cv2.rectangle(image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 0), 3)

    # Plot ground truth box(es)
    if len(gt) > 0:
        for i in range(len(gt)):
            cv2.rectangle(
                image,
                (int(gt[i][0]), int(gt[i][1])),
                (int(gt[i][2]), int(gt[i][3])),
                (0, 0, 255), 3,
            )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_BBOX.jpg"
        Image.fromarray(image).save(pltname)

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title("Predicted box and GT box")
    plt.axis('off')
    plt.show()

    return image

def visualize_eigvec(eigvec, vis_folder, im_name, dim, scales, save=False):
    """
    Visualization of the second smallest eigvector
    """
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_attn.jpg"
        plt.imsave(fname=pltname, arr=eigvec, cmap='viridis')
        print(f"Eigen attention saved at {pltname}.")
    
    plt.figure(figsize=(6,6))
    plt.imshow(eigvec, cmap='cividis')
    plt.axis('off')
    plt.title("Second smallest eigenvector")
    plt.show()

def extract_patch_feats(model, transform, img_raw, block_index=11, device='cpu'):
    img_pil = Image.fromarray(img_raw).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.patch_embed(img_tensor)
        if x.ndim == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1).to(x.device)
        x = torch.cat((cls_token, x), dim=1)
        for i in range(block_index + 1):
            x = model.blocks[i](x)
    return x  # tensor shape [1, num_tokens, dim]

def _trim_reg_tokens_for_grid(x):
    # keep cls token then drop the first n_reg regularization tokens if present
    num_tokens = x.shape[1]
    total_patches = num_tokens - 1
    grid_size = int(math.sqrt(total_patches))
    n_reg = total_patches - (grid_size * grid_size)
    if n_reg > 0:
        x_trim = torch.cat([x[:, 0:1, :], x[:, 1 + n_reg :, :]], dim=1)
    else:
        x_trim = x
    return x_trim, grid_size, n_reg

def tokencut_on_image(model, transform, img_raw, img_size, block_index=11, tau=0.2, eps=1e-5, no_binary_graph=False, vis_folder="tokencut_vis", im_name="img", plot_vis=True):
    feats = extract_patch_feats(model, transform, img_raw, block_index)
    feats_trim, grid_size, n_reg = _trim_reg_tokens_for_grid(feats)

    # compute scale factors relative to the original image (pixels per grid cell)
    orig_h, orig_w = img_raw.shape[:2]
    scale_h = float(orig_h) / float(grid_size)
    scale_w = float(orig_w) / float(grid_size)
    scales = [scale_h, scale_w]

    preds, objects, mask, seed, bins, eigenvec = ncut(feats_trim, [grid_size, grid_size], scales, img_raw.shape, tau=tau, eps=eps, no_binary_graph=no_binary_graph)

    # Visualizations (saved to vis_folder). Now preds are in original-image pixel coordinates.
    if plot_vis:
        #visualize_predictions(img_raw, preds, vis_folder, im_name)
        visualize_eigvec(eigenvec, vis_folder, im_name, [grid_size, grid_size], scales)

    # maintain backward compatibility: single first box
    first_pred = preds[0] if (hasattr(preds, "__len__") and len(preds) > 0) else None
    return {
        "pred_boxes": np.asarray(preds),
        "pred_box": np.asarray(first_pred) if first_pred is not None else None,
        "objects": objects,
        "mask": mask,
        "seed": seed,
        "eigenvec": eigenvec,
        "grid_size": grid_size,
        "scales": scales,
    }

# Simple IoU helper for xyxy boxes
def iou_xyxy(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0