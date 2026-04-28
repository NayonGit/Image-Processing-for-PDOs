Welcome to the Exploring Foundation Models for PDOs GitHub Repository !

📂 Data Management & Setup
This project utilizes three major Patient-Derived Organoids (PDOs) datasets: OrgaQuant, Tellu, and MultiOrg. All data management tools are located in the datasets/ directory.

1. Download Raw Data
To retrieve the raw .h5 and image files, use the provided download.py script. It handles both direct URLs and Google Drive links (via gdown).

Bash
python datasets/download.py
2. Standardization & Preprocessing
Because each dataset comes with different partitioning and internal structures, we provide a standardization pipeline. Running standardize_datasets.py performs the following critical operations:

OrgaQuant: Validates and copies the original 2019 benchmark structure.

Tellu: Generates a deterministic 90/10 Train/Test split from the raw pool.

MultiOrg: Implements stratified sampling by imaging plate. This is crucial to prevent spatial data leakage, ensuring that patches from the same biological plate are not shared between training and testing sets. It also subsamples the dataset to a manageable scale (default: 5000 train, 600 test images).

To process all datasets, run:

Bash
python datasets/standardize_datasets.py
Note: This script will output detailed statistics for each dataset, including image counts, total annotated objects, and average organoids per image, ensuring your local setup matches our experimental environment.
