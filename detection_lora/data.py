import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from detection_lora.utils import box_xyxy_to_cxcywh


DATASET_INFO = {
    "tellu": {
        "path": "data/tellu.h5",
        "task": "detection", # labels = [cls, x1, y1, x2, y2]
        "original_size": (960, 1280), #(1024, 1024),
        "img_size": 1024,
        "num_classes": 1,
    },
    "orgaquant": {
        "path": "data/orgaquant.h5",
        "task": "localization", # labels = [x1, y1, x2, y2]
        "original_size": None,  # taille variable, lue dynamiquement
        "img_size": 512,
        "num_classes": 1,
    },
    "multiorg": {
        "path": "data/multiorg.h5",
        "task": "detection",  # labels = [cls, x1, y1, x2, y2]
        "original_size": (512, 512), #(2048, 2048),
        "img_size": 2048,
        "num_classes": 1,
    },
}

# =============================================================================
# 9. Dataset (tellu / orgaquant / multiorg)
# =============================================================================

class OrganoidDetectionDataset(Dataset):
    """
    Dataset that reads images and bounding-box annotations from an H5
    file produced by the download pipeline.

    Supported datasets:
        - tellu      (detection,    labels = [cls, x1, y1, x2, y2])
        - orgaquant  (localization, labels = [x1, y1, x2, y2])
        - multiorg   (localization, labels = [cls, x1, y1, x2, y2] but cls=0)

    All boxes are returned as normalized cxcywh ∈ [0, 1].
    """

    def __init__(self, dataset_name: str, split: str = "train", h5_path: str | None = None):
        super().__init__()
        info = DATASET_INFO[dataset_name]
        self.dataset_name = dataset_name
        self.task = info["task"]
        self.num_classes = info["num_classes"]
        self.dynamic_size = info["original_size"] is None  # taille à lire dynamiquement
        if not self.dynamic_size:
            self.orig_h, self.orig_w = info["original_size"]
        else:
            self.orig_h, self.orig_w = None, None
        self.img_size = info["img_size"]
        self.h5_path = h5_path or info["path"]
        self.split = split

        with h5py.File(self.h5_path, "r") as hdf:
            group = hdf[split]
            self.img_names = list(group["images"].keys())

        # ImageNet-normalised transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"[Dataset] {dataset_name}/{split}:"
            f"{len(self.img_names)} images, "
            f"task={self.task}, "
            f"num_classes={self.num_classes}, "
            f"original_size={info['original_size']}, "
            f"resized_to={self.img_size}x{self.img_size}"
        )

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        name = self.img_names[idx]

        with h5py.File(self.h5_path, "r") as hdf:
            group = hdf[self.split]
            img_np = np.array(group["images"][name])
            lbl_np = np.array(group["labels"][name])
        
        orig_h, orig_w = img_np.shape[:2]

        # Etirement de contraste robuste (1% - 99% percentiles)
        p1, p99 = np.percentile(img_np, (1, 99))
        img_np = np.clip(img_np, p1, p99)
        img_np = (img_np - p1) / (p99 - p1 + 1e-6) * 255
        img_np = img_np.astype(np.uint8)

        img_pil = Image.fromarray(img_np).convert("RGB")
        image = self.transform(img_pil)  # (3, img_size, img_size)
        boxes, labels = self._parse_labels(lbl_np, orig_h, orig_w)
        return image, {"boxes": boxes, "labels": labels}

    def _parse_labels(self, lbl_np: np.ndarray, orig_h: int = None, orig_w: int = None):
        """
        Parse raw label array from H5 file into normalised cxcywh boxes + class labels.

        Detection  (tellu):       lbl = [cls, x1, y1, x2, y2]
        Localization (orgaquant, multiorg): lbl = [x1, y1, x2, y2]  → class 0
        """
        orig_h = orig_h if self.dynamic_size else self.orig_h
        orig_w = orig_w if self.dynamic_size else self.orig_w

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
                # [x1, y1, x2, y2] with single class
                cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
                coords = lbl_np[:, 1:5].astype(np.float32)
            else:
                # [cls, x1, y1, x2, y2]
                cls_ids = lbl_np[:, 0].astype(np.int64)
                coords = lbl_np[:, 1:5].astype(np.float32)
        else:
            # [x1, y1, x2, y2] with single class 0
            cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
            coords = lbl_np[:, 0:4].astype(np.float32)

        x1 = coords[:, 0] / orig_w
        y1 = coords[:, 1] / orig_h
        x2 = coords[:, 2] / orig_w
        y2 = coords[:, 3] / orig_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy).clamp(0, 1)

        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]
        valid = (w > 0) & (h > 0)
        boxes_cxcywh = boxes_cxcywh[valid]
        cls_ids = cls_ids[valid.numpy()]

        labels = torch.tensor(cls_ids, dtype=torch.int64)
        return boxes_cxcywh, labels
    


