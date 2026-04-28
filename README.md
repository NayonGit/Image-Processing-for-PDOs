# Exploring Foundation Models for Patient-Derived Organoids Detection

This repository contains the official implementation for the comparative study of Foundation Models (FMs) and CNNs in the context of Patient-Derived Organoids (PDOs) detection. The project is organized into three main architectural frameworks, each supported by a dedicated module and a high-performance pipeline.

## 📁 Project Structure

The codebase is organized into three specialized directories, each corresponding to a distinct detection paradigm:

* **/FasterRCNN**: Standard region-based detection. Best for high-accuracy and fast convergence with CNN backbones.
* **/DETR**: Transformer-based end-to-end detection. Optimized for global context via DINOv2/v3 backbones.
* **/CenterNet**: Anchor-free heatmap-based detection. Focuses on point-based localization from dense feature maps.

---

## 📂 Data Management & Setup

All data utilities are located in the `datasets/` directory. We support **OrgaQuant**, **Tellu**, and **MultiOrg** datasets.

### 1. Download Raw Data
To retrieve the raw files (HDF5 and images), run the download script. It handles both direct URLs and Google Drive links.

```bash
python datasets/download.py
