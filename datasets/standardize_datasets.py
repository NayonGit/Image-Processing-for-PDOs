import h5py
import torch
import numpy as np
from torch.utils.data import random_split
from pathlib import Path
from collections import defaultdict
import re

SEED = 42

def copy_h5_group(src_grp, dst_grp):
    """Recursively copy HDF5 group structure and data."""
    for key in src_grp.keys():
        if isinstance(src_grp[key], h5py.Dataset):
            dst_grp[key] = src_grp[key][()]
        else:
            sub_group = dst_grp.create_group(key)
            copy_h5_group(src_grp[key], sub_group)

def process_tellu(input_path, output_path):
    """Process TELLU dataset with 90/10 train/test split."""
    print("="*60)
    print("TELLU: Creating 90/10 split")
    print("="*60)
    
    with h5py.File(input_path, "r") as src:
        all_keys = []
        for split in src.keys():
            all_keys.extend(list(src[split]["images"].keys()))
        
        n_total = len(all_keys)
        n_train = int(0.9 * n_total)
        n_test = n_total - n_train
        
        generator = torch.Generator().manual_seed(SEED)
        train_subset, test_subset = random_split(
            all_keys, [n_train, n_test], generator=generator
        )
        
        train_keys = list(train_subset)
        test_keys = list(test_subset)
        
        print(f"Total images: {n_total}")
        print(f"Train: {len(train_keys)} (90%)")
        print(f"Test: {len(test_keys)} (10%)")
        
        with h5py.File(output_path, "w") as dst:
            for split_name, keys in [("train", train_keys), ("test", test_keys)]:
                split_group = dst.create_group(split_name)
                img_group = split_group.create_group("images")
                lbl_group = split_group.create_group("labels")
                
                for key in keys:
                    for src_split in src.keys():
                        if key in src[src_split]["images"]:
                            img_group[key] = src[src_split]["images"][key][()]
                            lbl_group[key] = src[src_split]["labels"][key][()]
                            break
                
                print(f"{split_name}: {len(keys)} images copied")
    
    print(f"Tellu saved to {output_path}\n")

def process_orgaquant(input_path, output_path):
    """Copy ORGAQUANT dataset as-is."""
    print("="*60)
    print("ORGAQUANT: Copying as-is")
    print("="*60)
    
    with h5py.File(input_path, "r") as src:
        with h5py.File(output_path, "w") as dst:
            copy_h5_group(src, dst)
    
    print(f"OrgaQuant copied to {output_path}\n")

def process_multiorg(input_path, output_path, train_size=5000, test_size=600):
    """Process MULTIORG dataset with stratified sampling by plate."""
    print("="*60)
    print("MULTIORG: Stratified sampling by plate")
    print("="*60)
    
    pattern = r'^(Normal|Macros)_Plate_(\d+)_image_(\d+)_patch_(\d+)_(\d+)$'
    
    with h5py.File(input_path, "r") as src:
        split_data = defaultdict(lambda: defaultdict(list))
        plate_counts = defaultdict(lambda: defaultdict(int))
        
        for split in src.keys():
            grp = src[split]["images"]
            for key in grp.keys():
                m = re.match(pattern, key)
                if m:
                    mod, plate_id, img_id, p1, p2 = m.groups()
                    split_data[split][plate_id].append(key)
                    plate_counts[split][plate_id] += 1
        
        multiorg_generator = torch.Generator().manual_seed(SEED)
        
        with h5py.File(output_path, "w") as dst:
            for split in sorted(src.keys()):
                target_size = train_size if split == "train" else test_size
                
                total_plates = sum(plate_counts[split].values())
                proportions = {
                    plate: count / total_plates 
                    for plate, count in plate_counts[split].items()
                }
                
                samples_per_plate = {}
                sampled = 0
                for plate in sorted(proportions.keys(), key=lambda x: -proportions[x]):
                    if plate == sorted(proportions.keys())[-1]:
                        samples_per_plate[plate] = target_size - sampled
                    else:
                        samples_per_plate[plate] = int(proportions[plate] * target_size)
                        sampled += samples_per_plate[plate]
                
                sampled_keys = []
                for plate in sorted(samples_per_plate.keys()):
                    n_sample = samples_per_plate[plate]
                    available_keys = split_data[split][plate]
                    n_available = len(available_keys)
                    n_pick = min(n_sample, n_available)
                    
                    indices = torch.randperm(
                        n_available, generator=multiorg_generator
                    )[:n_pick].tolist()
                    sampled_plate_keys = [available_keys[i] for i in indices]
                    sampled_keys.extend(sampled_plate_keys)
                
                print(f"\n{split}: sampling {len(sampled_keys)} images (target: {target_size})")
                for plate in sorted(samples_per_plate.keys()):
                    print(f"  Plate {plate}: {samples_per_plate[plate]} images")
                
                split_group = dst.create_group(split)
                img_group = split_group.create_group("images")
                lbl_group = split_group.create_group("labels")
                
                for key in sampled_keys:
                    img_group[key] = src[split]["images"][key][()]
                    lbl_group[key] = src[split]["labels"][key][()]
    
    print(f"\nMultiOrg saved to {output_path}\n")

def verify_datasets(datasets_config):
    """Verify processed datasets and print statistics."""
    def count_objects(lbl):
        arr = np.asarray(lbl)
        if arr.size == 0:
            return 0
        if arr.ndim == 0:
            return int(arr) if np.issubdtype(arr.dtype, np.integer) else 1
        return arr.shape[0]
    
    for name, path in datasets_config.items():
        p = Path(path)
        if not p.exists():
            print(f"{name}: file not found ({path})")
            continue
        
        with h5py.File(path, "r") as hdf:
            print("\n" + "="*60)
            print(f"DATASET: {name}")
            for split in sorted(hdf.keys()):
                grp = hdf[split]
                n_images = len(grp.get("images", {}))
                if "labels" in grp:
                    total_objects = 0
                    counts = []
                    for key in grp["labels"].keys():
                        lbl = grp["labels"][key][()]
                        n = count_objects(lbl)
                        total_objects += n
                        counts.append(n)
                    avg = total_objects / len(counts) if counts else 0
                    median = int(np.median(counts)) if counts else 0
                    zeros = sum(1 for c in counts if c == 0)
                    print(f"\nSplit: {split}  images={n_images}  images_with_labels={len(counts)}  total_objects={total_objects}  avg={avg:.2f}  median={median}  images_with_0_objects={zeros}")
                else:
                    print(f"\nSplit: {split}  images={n_images}  (no 'labels' group)")

if __name__ == "__main__":
    datasets = {
        "tellu": ("data/tellu.h5", "data/tellu_processed.h5"),
        "orgaquant": ("data/orgaquant.h5", "data/orgaquant_processed.h5"),
        "multiorg": ("data/multiorg.h5", "data/multiorg_processed.h5"),
    }
    
    # Process datasets
    process_tellu(datasets["tellu"][0], datasets["tellu"][1])
    process_orgaquant(datasets["orgaquant"][0], datasets["orgaquant"][1])
    process_multiorg(datasets["multiorg"][0], datasets["multiorg"][1])
    
    # Verify results
    verify_config = {
        "tellu": datasets["tellu"][1],
        "orgaquant": datasets["orgaquant"][1],
        "multiorg": datasets["multiorg"][1],
    }
    verify_datasets(verify_config)
    
    print("\n" + "="*60)
    print("All datasets processed successfully!")
    print("="*60)