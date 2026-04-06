import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_INFO = {
    "tellu": {
        "path": "data/tellu_processed.h5",
        "task": "detection",  # labels = [cls, x1, y1, x2, y2]
        "original_size": (960, 1280),
        "img_size": 1024,
        "num_classes": 1,  # 1,
    },
    "orgaquant": {
        "path": "data/orgaquant_processed.h5",
        "task": "localization",  # labels = [x1, y1, x2, y2]
        "original_size": None,
        "img_size": 512,
        "num_classes": 1,
    },
    "multiorg": {
        "path": "data/multiorg_processed.h5",
        "task": "detection",  # labels = [cls, x1, y1, x2, y2]
        "original_size": (512, 512),
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
        - multiorg   (detection,    labels = [cls, x1, y1, x2, y2] but cls=0)

    Returns DINO-formatted samples with absolute XYXY boxes in resized image space:
      - image: normalized tensor [3, img_size, img_size]
      - target:
          boxes: XYXY absolute (pixel) in resized image space
          labels: int64
          orig_size: [H_orig, W_orig]
          size: [H_resized, W_resized]
          image_id: [idx]
    """

    def __init__(self, dataset_name: str, split: str = "train", h5_path: str | None = None):
        super().__init__()
        info = DATASET_INFO[dataset_name]
        self.dataset_name = dataset_name
        self.task = info["task"]
        self.num_classes = info["num_classes"]
        self.img_size = info["img_size"]
        self.h5_path = h5_path or info["path"]
        self.split = split

        with h5py.File(self.h5_path, "r") as hdf:
            group = hdf[split]
            self.img_names = list(group["images"].keys())

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"[Dataset] {dataset_name}/{split}: {len(self.img_names)} images, img_size={self.img_size}")

    def __len__(self) -> int:
        return len(self.img_names)

    def _extract_cls_and_coords(self, lbl_np: np.ndarray):
        if lbl_np.ndim == 1:
            lbl_np = lbl_np.reshape(-1, lbl_np.shape[0]) if lbl_np.size > 0 else lbl_np.reshape(0, 0)

        if lbl_np.size == 0 or len(lbl_np) == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        if self.task == "detection":
            if self.dataset_name == "tellu":
                #cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
                cls_ids = lbl_np[:, 0].astype(np.int64) - 1  # tellu has cls=1 for organoid, we want cls=0
                coords = lbl_np[:, 1:5].astype(np.float32)
            else:
                cls_ids = lbl_np[:, 0].astype(np.int64)
                coords = lbl_np[:, 1:5].astype(np.float32)
        else:
            cls_ids = np.zeros(len(lbl_np), dtype=np.int64)
            coords = lbl_np[:, 0:4].astype(np.float32)

        return cls_ids, coords

    def __getitem__(self, idx: int):
        name = self.img_names[idx]

        with h5py.File(self.h5_path, "r") as hdf:
            group = hdf[self.split]
            img_np = np.array(group["images"][name])
            lbl_np = np.array(group["labels"][name])

        orig_h, orig_w = img_np.shape[:2]

        p1, p99 = np.percentile(img_np, (1, 99))
        img_np = np.clip(img_np, p1, p99)
        img_np = (img_np - p1) / (p99 - p1 + 1e-6) * 255
        img_np = img_np.astype(np.uint8)

        img_pil = Image.fromarray(img_np).convert("RGB")
        image = self.transform(img_pil)

        cls_ids, coords = self._extract_cls_and_coords(lbl_np)
        if len(coords) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            sx = self.img_size / float(orig_w)
            sy = self.img_size / float(orig_h)
            x1 = coords[:, 0] * sx
            y1 = coords[:, 1] * sy
            x2 = coords[:, 2] * sx
            y2 = coords[:, 3] * sy
            boxes = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)
            valid = ((boxes[:, 2] - boxes[:, 0]) > 1e-6) & ((boxes[:, 3] - boxes[:, 1]) > 1e-6)
            boxes = boxes[valid]
            labels = torch.tensor(cls_ids[valid.numpy()], dtype=torch.int64)

        target = {
            "boxes": boxes,  # xyxy abs
            "labels": labels,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([self.img_size, self.img_size], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target