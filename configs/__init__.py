import os
import yaml
from pathlib import Path
from easydict import EasyDict as edict

with open(os.path.join(os.path.dirname(__file__), 'dataset_config.yaml'), 'r') as file:
    dataset_config = edict(yaml.safe_load(file))

with open(os.path.join(os.path.dirname(__file__), 'train_config.yaml'), 'r') as file:
    train_config = edict(yaml.safe_load(file))

default_data_dir = os.path.join(Path(os.path.dirname(__file__)).parent.parent, 'Image-Processing-for-PDOs/data')