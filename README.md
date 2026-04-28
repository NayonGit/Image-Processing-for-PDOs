# Exploring Foundation Models for Patient-Derived Organoids Detection

This repository contains the official implementation for the comparative study of Foundation Models (FMs) and CNNs in the context of Patient-Derived Organoids (PDOs) detection. The project is organized into three main architectural frameworks, each supported by a dedicated module and a high-performance pipeline.

## Project Structure

The codebase is organized into three specialized directories, each corresponding to a distinct detection paradigm:

* **/FasterRCNN**: Standard region-based detection. Best for high-accuracy and fast convergence with CNN backbones.
* **/DETR**: Transformer-based end-to-end detection. Optimized for global context via DINOv2/v3 backbones.
* **/CenterNet**: Anchor-free heatmap-based detection. Focuses on point-based localization from dense feature maps.

---

## Data Management & Setup

All data utilities are located in the `datasets/` directory. We support **OrgaQuant**, **Tellu**, and **MultiOrg** datasets.
As the MultiOrg dataset is quite large, these operations might take some hours to run.

### 1. Download Raw Data
To retrieve the raw files (HDF5 and images), run the download script. It handles both direct URLs and Google Drive links.

```bash
python datasets/download.py
```

### 2. Standardization & Preprocessing
To ensure rigorous evaluation and prevent data leakage, you must run the standardization pipeline.

```bash
python datasets/standardize_datasets.py
```

---

## Running Experiments

Each architectural framework is launched using its specific entry-point script located at the root of the repository:

### 🔹 Faster R-CNN
Used for benchmarking CNNs (ResNet, ConvNeXt-V2) and ViTs (Swinv2, Dinov2).

```bash
python run_faster_rcnn.py --backbone convnextv2 --method lora --dataset orgaquant
```

### 🔹 DETR (DEtection TRansformer)

Leverages DINOv2/v3 backbones with DETR head. This process takes up to 150 epochs to converge with standard DETR heads, even with CNN backbones.
Deformable DETR is a great improvement, and should be further explored.

```bash
python run_detr.py --backbone dinov2 --method lora --dataset orgaquant
```

### 🔹 CenterNet 

This approach uses directly Dinov2/v3's features to generate efficient heatmaps for Detection. It significantly improves with the number of parameters of the backbone.

```bash
python run_centernet.py --backbone dinov2 --method lora --dataset orgaquant
```



