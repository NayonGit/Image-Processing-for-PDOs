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
We report here the state-of-the-art results, presented in each original paper, as some indicators of performance.

| Dataset | Metric | SOTA Value |
| :--- | :---: | :---: |
| OrgaQuant | mAP70 | 80% |
| Tellu | mAP50 | 79% |
| MultiOrg | mAP50 / mAP75 | 65.79% / 23.42% |



### 1. Download Raw Data
To retrieve the raw files (HDF5 and images), run the download script. It handles both direct URLs and Google Drive links.

```bash
python datasets/download.py
```

| Dataset | Year | Classes | Nb of Annotations |
| :--- | :---: | :---: | :---: |
| **OrgaQuant** | 2019 | 1 | 14,240 |
| **Tellu** | 2023 | 4 | 20,000+ |
| **MultiOrg** | 2024 | 1 | 60,000+ |

### 2. Standardization & Preprocessing
To ensure rigorous evaluation and prevent data leakage, you must run the standardization pipeline.

```bash
python datasets/standardize_datasets.py
```

After preprocessing and plate-based stratification (for MultiOrg), the data distribution is as follows:

| Dataset | Split | Images | Total Objects |
| :--- | :--- | :---: | :---: |
| **OrgaQuant** | Train / Test | 1642 / 112 | 13,004 / 1,135 |
| **Tellu** | Train / Test | 754 / 84 | 20,721 / 2,242 |
| **MultiOrg** | Train / Test | 5851 / 793 | 13,507 / 1,319 |

---

## Running Experiments

Each architectural framework is launched using its specific entry-point script located at the root of the repository:

### 🔹 Faster R-CNN
Used for benchmarking CNNs (ResNet, ConvNeXt-V2) and ViTs (Swinv2, Dinov2).

```bash
python run_faster_rcnn.py train --backbone convnextv2 --method lora --dataset orgaquant
```

Here are the results using a standard ResNet50. 

| Method | OrgaQuant (mAP/50/75) | Tellu (mAP/50/75) | MultiOrg (mAP/50/75) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 72.36 / 91.41 / **87.20** | 45.78 / **74.13** / **53.43** | 36.03 / 76.86 / 24.25 |
| **LoRA (r=8)** | **73.44** / **91.44** / 86.08 | **45.96** / 74.09 / 52.73 | **39.63** / **77.01** / 33.27 |
| **DoRA (r=8)** | 73.25 / 91.24 / 87.19 | 45.90 / 73.84 / 52.75 | 38.55 / 76.08 / **38.55** |

Comparison of different backbones using the Faster R-CNN architecture on OrgaQuant.

| Backbone Model | Performance (mAP50) |
| :--- | :--- |
| ResNet50 | 91.44% |
| Swinv2 (Base) | 90.65% |
| **ConvNeXtv2 (Base)** | **91.83%** |
| ConvNeXtv2 (Huge) | 91.44% |
| DINOv2-L (Large) | 73.57% |

### 🔹 DETR (DEtection TRansformer)

Leverages DINOv2/v3 backbones with DETR head. This process takes up to 150 epochs to converge with standard DETR heads, even with CNN backbones.
Deformable DETR is a great improvement, and should be further explored: a simple implementation is provided in this work, and usable easily. This motivates use to further the work by implementing the DINO module, which is still something in development.

```bash
## Standard Use
python run_detr.py train --backbone dinov2 --method lora --dataset orgaquant
## With Deformable DETR
python run_detr.py train --backbone dinov2 --method lora --dataset orgaquant --deformable-attention
```
Performances obtained across all datasets with different configurations. The computational time required for a single training is tremendously high, thus the lack of some data. 

| Backbone | Architecture | Fine-Tuning | OrgaQuant (mAP50 / mAP) | Tellu (mAP50 / mAP) | MultiOrg (mAP50 / mAP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DINOv2 | DETR | None | 72.5% / 43.3% | 64.9% / 30.0% | 49.6% / 17.8% |
| DINOv3 | DETR | None | 55.0% / 29.7% | 53.4% / 23.0% | 54.9% / 20.3% |
| **DINOv2** | **DETR** | **LoRA** | **82.6% / 62.6%** | **82.6% / 47.0%** | **67.8% / 30.0%** |
| DINOv3 | DETR | LoRA | 81.4% / 60.3% | 80.7% / 46.7% | 63.3% / 29.1% |
| DINOv2 | DETR | DoRA | 81.1% / 57.5% | 81.0% / 46.3% | -- / -- |
| DINOv2 | Def. DETR | LoRA | 82.4% / 60.8% | 80.6% / 47.4% | -- / -- |

*Note: Dashes (--) indicate unassessed configurations due to computational constraints.*

Deformable DETR outspeeds by a great margin DETR, allowing faster training.

<img width="505" height="333" alt="image" src="https://github.com/user-attachments/assets/506c873a-ee4b-4426-bc4f-650e63c827b1" />

### 🔹 CenterNet 

This approach uses directly Dinov2/v3's features to generate efficient heatmaps for Detection. It significantly improves with the number of parameters of the backbone.

```bash
python run_centernet.py train --backbone dinov2 --method lora --dataset orgaquant
```

Performances obtained on OrgaQuant.

| Backbone Model | Performance (mAP50) |
| :--- | :--- |
| DINOv2-B (Base) | 73.34% |
| DINOv2-L (Large) | 78.29% |
| **DINOv2-G (Giant)** | **79.46%** |
| DINOv3-B (Base) | 70.31% |
| DINOv3-L (Large) | 74.67% |



