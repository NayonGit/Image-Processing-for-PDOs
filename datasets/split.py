import os
from configs import *
import h5py
from sklearn.model_selection import KFold, StratifiedKFold

def get_dataset_split(dataset_name: str,
                      split: int,
                      num_splits: int,
                      data_dir: str = default_data_dir,
                      seed: int = 0) -> None:
    """Get the names of the images in the dataset splits.
        
        Args:
            dataset_name [str]: name of the dataset to prepare
            split [int]: index of the split to get
            num_splits [int]: number of splits to create
            data_dir [str]: directory to save the prepared dataset to
            seed [int]: seed for the random number generator
        
        Returns:
            None
    """

    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    dataset_path = os.path.join(data_dir, f'{dataset_name}.h5')
    dataset_type = dataset_config[dataset_name].task

    with h5py.File(dataset_path, 'r') as hdf:
        has_test = 'test' in hdf.keys()
        if dataset_type == 'classification':
            train_names = [os.path.join(cls, img) for cls in list(hdf.get('train').keys())
                           for img in list(hdf.get(os.path.join('train', cls)).keys())]
            test_names = [os.path.join(cls, img) for cls in list(hdf.get('test').keys())
                          for img in list(hdf.get(os.path.join('test', cls)).keys())] if has_test else None
            train_labels = [os.path.dirname(name) for name in train_names]
            splitter = StratifiedKFold(num_splits, random_state=seed, shuffle=True)
            train_idx, val_idx = list(splitter.split(train_names, train_labels))[split]
        else:
            train_names = list(hdf.get('train/images').keys())
            test_names = list(hdf.get('test/images').keys()) if has_test else None
            splitter = KFold(num_splits, random_state=seed, shuffle=True)
            train_idx, val_idx = list(splitter.split(train_names))[split]

    val_names = [train_names[i] for i in val_idx]
    train_names = [train_names[i] for i in train_idx]
    
    return train_names, val_names, test_names