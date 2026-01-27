import os
import csv
import h5py
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def prepare_orgaquant(in_dir: str,
                      out_path: str,
                      folder_name: str = 'Intestinal Organoid Dataset',
                      extension: str = '.jpg') -> None:
    """Prepare the OrgaQuant dataset for training and testing.
        
        Args:
            in_dir [str]: path to the directory containing the OrgaQuant dataset
            out_path [str]: path to save the prepared dataset
            folder_name [str]: name of the folder in the downloaded dataset
            extension [str]: extension of the images in the dataset
        
        Returns:
            None
    """
    in_dir = os.path.join(in_dir, folder_name)
    with h5py.File(out_path, 'w') as hdf:
        for split in ['train', 'test']:
            if split == 'train':
                labels_file = os.path.join(in_dir, 'train_labels.csv')
            else:
                labels_file = os.path.join(in_dir, 'test_labels.csv')
            
            with open(labels_file, 'r') as f:
                reader = csv.reader(f)
                labels = list(reader) if split == 'train' else list(reader)[1:]
            
            labels = [[os.path.basename(row[0])] + row[1:-1] for row in labels]

            group = hdf.create_group(split)
            group_img = group.create_group('images')
            group_lbl = group.create_group('labels')
            images = [os.path.join(in_dir, split, img) for img in os.listdir(os.path.join(in_dir, split)) if img.endswith(extension)]
            for img_path in tqdm(images,
                                 desc=split,
                                 leave=False):
                img_name = os.path.basename(img_path)
                img = np.array(Image.open(img_path).convert('L'))
                img_labels = np.array(
                    [row[1:] for row in labels if row[0] == img_name], 
                    dtype='int'
                )
                group_img.create_dataset(os.path.splitext(img_name)[0], data=img)
                group_lbl.create_dataset(os.path.splitext(img_name)[0], data=img_labels)

def prepare_tellu(in_dir: str,
                  out_path: str,
                  folder_name: str = 'OrganoidDataset',
                  img_extension: str = '.jpeg',
                  lbl_extension: str = '.txt') -> None:
    """Prepare the Tellu dataset for training and testing.
        
        Args:
            in_dir [str]: path to the directory containing the Tellu dataset
            out_path [str]: path to save the prepared dataset
            folder_name [str]: name of the folder in the downloaded dataset
            img_extension [str]: extension of the images in the dataset
            lbl_extension [str]: extension of the labels in the dataset
        
        Returns:
            None
    """
    in_dir = os.path.join(in_dir, folder_name)
    with h5py.File(out_path, 'w') as hdf:
        group = hdf.create_group('train')
        img_group = group.create_group('images')
        lbl_group = group.create_group('labels')
        for split in ['train', 'val']:
            images = [os.path.join(in_dir, split, 'images', img) for img in os.listdir(os.path.join(in_dir, split, 'images')) if img.endswith(img_extension)]
            labels = [os.path.join(in_dir, split, 'labels', os.path.basename(img.replace(img_extension, lbl_extension))) for img in images]
            for img_path, lbl_path in tqdm(zip(images, labels),
                                           desc=split,
                                           total=len(images),
                                           leave=False):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img = np.array(Image.open(img_path).convert('L'))
                with open(lbl_path, 'r') as f:
                    lbl = np.array([[float(x) for x in line.strip().split()] for line in f])
                
                H, W = img.shape
                cls, center_x, center_y, box_w, box_h = lbl.T
                center_x, box_w = center_x*W, box_w*W
                center_y, box_h = center_y*H, box_h*H
                start_x, end_x = (center_x - box_w/2).astype(int), (center_x + box_w/2).astype(int)
                start_y, end_y = (center_y - box_h/2).astype(int), (center_y + box_h/2).astype(int)

                if any(start_x == end_x) or any(start_y == end_y):
                    continue

                lbl = np.stack([cls+1, start_x, start_y, end_x, end_y]).T
                
                img_group.create_dataset(img_name, data=img)
                lbl_group.create_dataset(img_name, data=lbl)


def prepare_multiorg(in_dir: str,
                     out_path: str) -> None:
    """Prepare the MultiOrg dataset for training and testing.
        
        Args:
            in_dir [str]: path to the directory containing the MultiOrg dataset
            out_path [str]: path to save the prepared dataset
        
        Returns:
            None
    """
    def get_corresponding_json(tiff_path):
        dir_path = os.path.dirname(tiff_path)
        json_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]
        assert len(json_path) == 1, f'Found {len(json_path)} json files for {tiff_path}.'
        return json_path[0]
    
    def preprocess_label(label):
        lbl = []
        for key in label:
            corners = np.array(label[key])
            start_y, start_x = corners.min(0)
            end_y, end_x = corners.max(0)
            lbl.append([start_x, start_y, end_x, end_y])
        return np.array(lbl)

    with h5py.File(out_path, 'w') as hdf:
        for split in ['train', 'test']:
            group = hdf.create_group(split)
            img_group = group.create_group('images')
            lbl_group = group.create_group('labels')
            images = [path for path in Path(os.path.join(in_dir, 'multiorg', split)).rglob('*.tiff')]
            labels = list(map(get_corresponding_json, images))

            for img_path, lbl_path in tqdm(zip(images, labels),
                                           desc=split,
                                           total=len(images),
                                           leave=False):
                img_name = '_'.join(os.path.dirname(img_path).split('/')[-3:])
                img = np.array(Image.open(img_path))
                with open(lbl_path, 'r') as f:
                    lbl = json.load(f)
                lbl = preprocess_label(lbl)
                img_group.create_dataset(img_name, data=img)
                lbl_group.create_dataset(img_name, data=lbl)