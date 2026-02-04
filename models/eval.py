from torch.nn import Module
from ..configs import default_data_dir

def eval(model: Module,
         dataset_name: str,
         split: int,
         checkpoint_path: str,
         download: bool,
         download_dir: str = '/tmp',
         data_dir: str = default_data_dir,
         num_splits: int = 4) -> None:
    """Evaluate a trained model on a specified dataset.
        
        Args:
            model [torch.nn.Module | L.LightningModule]: model to evaluate
            dataset_name [str]: name of the dataset to evaluate on
            split [int]: index of the split to evaluate on
            checkpoint_path [str]: path to the model checkpoint
            download [bool]: whether to download the dataset
            download_dir [str]: directory to download the dataset to
            data_dir [str]: directory to save the prepared dataset to
            num_splits [int]: number of splits to create
        
        Returns:
            None
    """
    import torch
    import lightning as L
    from .models import get_model
    from ..datasets import get_dataset
    from ..datasets import prepare_dataset
    from torch.utils.data import DataLoader
    from ..datasets import get_dataset_split
    from .train_utils import seed_everything
    from ..configs import train_config, dataset_config

    seed_everything(train_config.seed)

    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    if download:
        prepare_dataset(dataset_name=dataset_name, download_dir=download_dir, data_dir=data_dir)

    if not isinstance(model, L.LightningModule):
        model = get_model(dataset_config[dataset_name].task)(model,
                                                            train_config.optimizer,
                                                            train_config.optimizer_params,
                                                            dataset_config[dataset_name].num_classes)
    
    img_size = dataset_config[dataset_name].size
    _, val_names, test_names = get_dataset_split(dataset_name=dataset_name,
                                                 split=split,
                                                 num_splits=num_splits,
                                                 data_dir=data_dir,
                                                 seed=train_config.seed)
    
    test_names = test_names if test_names is not None else val_names
    test_dataset = get_dataset(dataset_config[dataset_name].task,
                               dataset_name=dataset_name,
                               data_dir=data_dir,
                               names=test_names,
                               mode='test',
                               img_size=img_size)
    collate_fn = getattr(model, 'collate_fn', None)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=train_config.batch_size,
                                 shuffle=False,
                                 num_workers=train_config.num_workers,
                                 collate_fn=collate_fn)
    
    trainer = L.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                        devices=1)
    
    trainer.validate(model,
                     test_dataloader,
                     ckpt_path=checkpoint_path)