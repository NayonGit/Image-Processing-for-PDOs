#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datasets.prepare import prepare_dataset

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))


def main():
    # List of datasets you want to install
    datasets_to_install = ['tellu','orgaquant', 'multiorg']

    # Define paths
    DOWNLOAD_TEMP_DIR = os.path.join('.', 'tmp')
    FINAL_DATA_DIR = os.path.join('.', 'data')

    # Create directories if they don't exist
    os.makedirs(DOWNLOAD_TEMP_DIR, exist_ok=True)
    os.makedirs(FINAL_DATA_DIR, exist_ok=True)

    print(f" Dataset Preparation Script ")
    print(f"Root Directory: {root_dir}")
    print(f"Storage: {FINAL_DATA_DIR}\n")

    for name in datasets_to_install:
        print(f"=== Preparing dataset: {name} ===")
        try:
            prepare_dataset(
                dataset_name=name,
                should_download=True, 
                download_dir=DOWNLOAD_TEMP_DIR, # Temporary folder for .zip files
                data_dir=FINAL_DATA_DIR         # Final folder for .h5 files
            )
            print(f"Success: {name}.h5 is ready in {FINAL_DATA_DIR}\n")
            
        except Exception as e:
            print(f"Error during preparation of {name}: {e}")
            print(f"Check if {name} is correctly defined in configs/dataset_config.yaml\n")

if __name__ == "__main__":
    main()