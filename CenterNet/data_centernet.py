import h5py
import math
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

""" This file defines the OrganoidDetectionDataset class, 
which reads images and bounding-box annotations from an H5 file.
The dataset supports three datasets: tellu, orgaquant and multiorg, with different annotation formats (detection vs localization).
The dataset returns patches of the images along with CenterNet-style targets (heatmap, wh, reg, reg_mask) for training the detection head.
"""

DATASET_INFO = {
    "tellu": {
        "path": "data/tellu_processed.h5",
        "task": "detection", # labels = [cls, x1, y1, x2, y2]
        "original_size": (960, 1280), #(1024, 1024),
        "img_size": 1024,
        "num_classes": 1,
    },

    "orgaquant": {
        "path": "data/orgaquant_processed.h5",
        "task": "localization", # labels = [x1, y1, x2, y2]
        "original_size": None,  # taille variable, lue dynamiquement
        "img_size": 512,
        "num_classes": 1,
    },

    "multiorg": {
        "path": "data/multiorg_processed.h5",
        "task": "detection",  # labels = [cls, x1, y1, x2, y2]
        "original_size": (512, 512), #(2048, 2048),
        "img_size": 512,
        "num_classes": 1,
    },
}


class OrganoidDetectionDataset(Dataset):
    """
    Dataset that reads images and bounding-box annotations from an H5
    file produced by the download pipeline.

    Supported datasets:
        - tellu      (detection,    labels = [cls, x1, y1, x2, y2])
        - orgaquant  (localization, labels = [x1, y1, x2, y2])
        - multiorg   (localization, labels = [cls, x1, y1, x2, y2] but cls=0)

    """

    def __init__(self, dataset_name: str, split: str = "train", h5_path: str | None = None, img_names_filter: list = None, patch_size=224, overlap=30, downsample = 3.5):
        super().__init__()
        info = DATASET_INFO[dataset_name]
        self.dataset_name = dataset_name
        self.task = info["task"]
        self.img_size = info["img_size"]
        self.patch_size = patch_size
        self.overlap = overlap
        self.num_classes = info["num_classes"]
        self.h5_path = h5_path or info["path"]
        self.split = split
        self.dynamic_size = info["original_size"] is None  # size to read dynamically
        self.downsample = downsample
        self.output_size = self.patch_size // self.downsample 

        if not self.dynamic_size:
            self.orig_h, self.orig_w = info["original_size"]
        else:
            self.orig_h, self.orig_w = None, None

        with h5py.File(self.h5_path, "r") as hdf:
            group_img = hdf[split]["images"]
            group_lbl = hdf[split]["labels"]
            all_names = list(group_img.keys())
            self.img_names = img_names_filter if img_names_filter is not None else all_names

            self.patch_indices = [] # List of (img_name, y_start, x_start, y_end, x_end)
            for name in self.img_names:
                # we get the real size of THIS image
                img_ds = group_img[name]
                h, w = group_img[name].shape[:2]
                lbl_np = np.array(group_lbl[name])
                # boundaries computation for W and H
                y_bounds = self._get_boundaries(h, patch_size, overlap)
                x_bounds = self._get_boundaries(w, patch_size, overlap)
                
                for y_start, y_end in y_bounds:
                    for x_start, x_end in x_bounds:
                        boxes, _ = self._parse_labels_patch(lbl_np, y_start, x_start, y_end, x_end)
                        if len(boxes) > 0:
                            self.patch_indices.append((name, y_start, x_start, y_end, x_end))
                        else:
                            # we only read the small patch for this test
                            patch_pixels = img_ds[y_start:y_end, x_start:x_end]
                            
                            # we check if the patch is useful (contains objects or has enough intensity)
                            if self._is_patch_useful(patch_pixels, boxes):
                                self.patch_indices.append((name, y_start, x_start, y_end, x_end))
        # ImageNet-normalized transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"[Dataset] {dataset_name}/{split}:"
            f"{len(self.img_names)} images to {len(self.patch_indices)} patchs of size {self.patch_size}, "
            f"task={self.task}, "
            f"num_classes={self.num_classes}, "
            f"original_size={info['original_size']}, "
        )

    def _get_boundaries(self, length, patch_size, overlap):
        """Compute [start, end] boundaries to cover 'length' with overlap."""
        stride = patch_size - overlap
        if length <= patch_size:
            return [[0, length]]
        
        boundaries = []
        for i in range(math.ceil((length - overlap) / stride)):
            start = i * stride
            end = start + patch_size
            if end > length: # last patch 
                end = length
                start = max(0, length - patch_size)
            boundaries.append([start, end])
        return list(dict.fromkeys(map(tuple, boundaries))) # Remove duplicates if image is small


    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, idx: int):
        name, y1, x1, y2, x2 = self.patch_indices[idx]
        with h5py.File(self.h5_path, "r") as hdf:
            group = hdf[self.split]
            img_np = np.array(group["images"][name])
            lbl_np = np.array(group["labels"][name])

        # Percentile-based Contrast Stretching (1% - 99% percentiles)
        p1, p99 = np.percentile(img_np, (1, 99))
        img_np = np.clip(img_np, p1, p99)
        img_np = (img_np - p1) / (p99 - p1 + 1e-6) * 255
        img_np = img_np.astype(np.uint8)

        # Patch
        patch_np = img_np[y1:y2, x1:x2]
        img_pil = Image.fromarray(patch_np).convert("RGB")

        image = self.transform(img_pil)

        boxes_px, labels = self._parse_labels_patch(lbl_np, y1, x1, y2, x2)

        targets = self._make_centernet_target(boxes_px, labels)

        return image, {
                    "targets": targets,         # Used for Training Loss
                    "gt_boxes": boxes_px,       # Used for val/mAP_50 (in patch pixels)
                    "gt_labels": labels         # Used for val/mAP_50 (starts at 0)
                }    
    
    def _make_centernet_target(self, boxes, labels):
        """
        Transform the boxes [N, 4] (x1, y1, x2, y2) into CenterNet targets.
        """
        out_s = int(self.output_size)

        hm = torch.zeros((self.num_classes, out_s, out_s), dtype=torch.float32)
        wh = torch.zeros((2, out_s, out_s), dtype=torch.float32)
        reg = torch.zeros((2, out_s, out_s), dtype=torch.float32)
        reg_mask = torch.zeros((1, out_s, out_s), dtype=torch.float32)

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(labels[i])  
            
            # Computes the center of the box in pixel coordinates (relative to the patch)
            ctx = (box[0] + box[2]) / 2.0
            cty = (box[1] + box[3]) / 2.0
            
            # Coordinates on the output grid (downsampled)
            ctx_grid = ctx / self.downsample
            cty_grid = cty / self.downsample
            
            ix, iy = int(ctx_grid), int(cty_grid)

            if 0 <= ix < self.output_size and 0 <= iy < self.output_size:
                # Heatmap : we use a gaussian blob instead of a single pixel to stabilize training (especially for small objects)
                self._draw_gaussian(hm[cls_id], (ix, iy), radius=2)
                
                # Object Size : width and height normalized by the patch size
                wh[0, iy, ix] = (box[2] - box[0]) / self.patch_size
                wh[1, iy, ix] = (box[3] - box[1]) / self.patch_size
                
                # Offset 
                reg[0, iy, ix] = ctx_grid - ix
                reg[1, iy, ix] = cty_grid - iy
                
                # Mask to indicate where the loss should be computed
                reg_mask[0, iy, ix] = 1

        return {"hm": hm, "wh": wh, "reg": reg, "reg_mask": reg_mask}
    
    def _draw_gaussian(self, heatmap, center, radius):
        """Adds a gaussian blob to the heatmap at the specified center with the given radius."""
        x, y = center
        height, width = heatmap.shape
        
        y_range = torch.arange(max(0, y - radius), min(height, y + radius + 1), device=heatmap.device)
        x_range = torch.arange(max(0, x - radius), min(width, x + radius + 1), device=heatmap.device)
        
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        dist = ((xx - x)**2 + (yy - y)**2) / (2 * (radius / 3)**2)
        gaussian = torch.exp(-dist)
        
        current_region = heatmap[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1]
        heatmap[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] = torch.max(current_region, gaussian)

    def _is_patch_useful(self, img_patch, labels_in_patch):
        """
        DDetermines if a patch is worth using.
        """
        if len(labels_in_patch) > 0:
            return True
            
        mean_intensity = img_patch.mean()
        
        if mean_intensity < 0.02: 
            return False
            
        return random.random() < 0.1  # On ne garde que 10% du bruit pur
    
    def _parse_labels_patch(self, lbl_np, y1, x1, y2, x2):
        """
        Parse raw label array from H5 file into normalised bounding boxes and class labels for a given patch.

        Detection  (tellu):       lbl = [cls, x1, y1, x2, y2]
        Localization (orgaquant, multiorg): lbl = [x1, y1, x2, y2]  → class 0
        """
    
        if lbl_np.ndim == 1:
            # Single annotation or empty
            lbl_np = lbl_np.reshape(-1, lbl_np.shape[0]) if lbl_np.size > 0 else lbl_np.reshape(0, 0)

        if lbl_np.size == 0 or len(lbl_np) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )

        if self.task == "detection":
            if self.dataset_name == "tellu":
                coords = lbl_np[:, 1:5].astype(np.float32)
                cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
            else:
                coords = lbl_np[:, 1:5].astype(np.float32)
                cls_ids = lbl_np[:, 0].astype(np.int64)
        else:
            # Localization : class unique 0
            coords = lbl_np[:, 0:4].astype(np.float32)
            cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
        
        # coords_patch = coords_globales - offset_patch
        lx1, ly1 = coords[:, 0] - x1, coords[:, 1] - y1
        lx2, ly2 = coords[:, 2] - x1, coords[:, 3] - y1

        lx1 = np.clip(lx1, 0, self.patch_size)
        ly1 = np.clip(ly1, 0, self.patch_size)
        lx2 = np.clip(lx2, 0, self.patch_size)
        ly2 = np.clip(ly2, 0, self.patch_size)

        w_local = lx2 - lx1
        h_local = ly2 - ly1

        mask = (w_local > 5) & (h_local > 5)

        boxes = np.stack([lx1[mask], ly1[mask], lx2[mask], ly2[mask]], axis=1)
        boxes = np.clip(boxes, 0, self.patch_size)

        final_boxes = np.stack([lx1[mask], ly1[mask], lx2[mask], ly2[mask]], axis=1)
        final_labels = cls_ids[mask]  # no addition for background

        boxes_t = torch.as_tensor(final_boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(final_labels, dtype=torch.int64).reshape(-1) # Dimension [N]

        return boxes_t, labels_t