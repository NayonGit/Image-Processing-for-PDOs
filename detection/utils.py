import math
import torch
from torchvision.ops.boxes import box_area
from typing import List, Dict, Tuple

# =============================================================================
# 1. Box Utilities
# =============================================================================

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height]"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Compute IoU between two sets of boxes in [x_min, y_min, x_max, y_max] format."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU between two sets of boxes in [x_min, y_min, x_max, y_max] format."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


# =============================================================================
# 2. Patch Utilities
# =============================================================================

"""
def get_boundaries(num_patches: int, overlap_size: int, patch_size: int = 224) -> Tuple[List[List[int]], int]:
    img_size = patch_size * num_patches - overlap_size * (num_patches - 1)
    start = 0
    boundaries = []
    for _ in range(num_patches):
        end = start + patch_size
        boundaries.append([start, end])
        start = end - overlap_size
    return boundaries, img_size
"""

def get_boundaries(num_patches: int = None, overlap_size: int = 30, patch_size: int = 224, img_size: int = None):
    """
    Calculate patch boundaries with overlap.

    If num_patches is None, img_size must be provided and num_patches is computed as:
      N = ceil((img_size - overlap_size) / (patch_size - overlap_size))
    """
    if num_patches is None and img_size is None:
        raise ValueError("Either num_patches or img_size must be provided")

    stride = patch_size - overlap_size

    if img_size is None:
        img_size = patch_size + (num_patches - 1) * stride
    elif num_patches is None:
        num_patches = math.ceil((img_size - overlap_size) / stride)

    boundaries = []
    start = 0
    for i in range(num_patches):
        if i == num_patches - 1:
            end = img_size
            start = max(0, img_size - patch_size)
        else:
            end = start + patch_size
        boundaries.append([start, end])
        if i != num_patches - 1:
            start = end - overlap_size
    return boundaries, img_size


"""
def extract_patch_targets(
    targets: List[Dict],
    patch_bounds: Tuple[int, int, int, int],
    img_size: int,
    patch_size: int = 224,
) -> List[Dict]:
    start_y, end_y, start_x, end_x = patch_bounds
    device = targets[0]["boxes"].device if targets and "boxes" in targets[0] else torch.device("cpu")
    patch_lower = torch.tensor([start_x, start_y, start_x, start_y], dtype=torch.float32, device=device) / img_size
    patch_upper = torch.tensor([end_x, end_y, end_x, end_y], dtype=torch.float32, device=device) / img_size
    
    patch_targets = []
    for tgt in targets:
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        valid_mask = ((boxes_xyxy > patch_lower) & (boxes_xyxy < patch_upper)).all(dim=1)
        
        if valid_mask.sum() > 0:
            valid_boxes = boxes_xyxy[valid_mask]
            valid_labels = labels[valid_mask]
            
            valid_boxes_pixel = valid_boxes * img_size
            valid_boxes_pixel -= torch.tensor([start_x, start_y, start_x, start_y], dtype=torch.float32, device=device)
            valid_boxes_norm = valid_boxes_pixel / patch_size
            valid_boxes_cxcywh = box_xyxy_to_cxcywh(valid_boxes_norm)
            valid_boxes_cxcywh = valid_boxes_cxcywh.clamp(0, 1)
            
            patch_targets.append({
                "boxes": valid_boxes_cxcywh,
                "labels": valid_labels,
            })
        else:
            patch_targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            })
    
    return patch_targets
"""

def extract_patch_targets(
    targets: List[Dict],
    patch_bounds: Tuple[int, int, int, int],
    img_size: int,
    dataset_name: str = "",
    patch_size: int = 224,
) -> List[Dict]:
    """
    Extract targets (boxes and labels) that fall within a patch.
    """
    start_y, end_y, start_x, end_x = patch_bounds
    device = targets[0]["boxes"].device if targets and "boxes" in targets[0] else torch.device("cpu")
    patch_lower = torch.tensor([start_x, start_y, start_x, start_y], dtype=torch.float32, device=device) / img_size
    patch_upper = torch.tensor([end_x, end_y, end_x, end_y], dtype=torch.float32, device=device) / img_size
    
    patch_targets = []
    for tgt in targets:
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        
        if len(boxes) == 0:
            patch_targets.append({"boxes": boxes, "labels": labels})
            continue

        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        
        # 1. Vérifier si le centre de la boîte est dans le patch
        centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2.0
        valid_mask = ((centers >= patch_lower[:2]) & (centers <= patch_upper[:2])).all(dim=1)
        
        if valid_mask.sum() > 0:
            valid_boxes = boxes_xyxy[valid_mask]
            valid_labels = labels[valid_mask]
            
            # 2. Clamper les boîtes pour qu'elles restent dans les limites du patch
            valid_boxes = torch.max(valid_boxes, patch_lower)
            valid_boxes = torch.min(valid_boxes, patch_upper)
            
            # 3. Conversion en coordonnées locales du patch
            valid_boxes_pixel = valid_boxes * img_size
            valid_boxes_pixel -= torch.tensor([start_x, start_y, start_x, start_y], dtype=torch.float32, device=device)
            valid_boxes_norm = valid_boxes_pixel / patch_size
            valid_boxes_cxcywh = box_xyxy_to_cxcywh(valid_boxes_norm).clamp(0, 1)
            
            patch_targets.append({
                "boxes": valid_boxes_cxcywh,
                "labels": valid_labels,
            })
        else:
            patch_targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            })
    return patch_targets