from torch.utils.data import Dataset
from configs import default_data_dir
from datasets.detection_datasets import Tellu, OrgaQuant, MultiOrg

def get_dataset(name: str,
                data_dir: str = default_data_dir,
                **kwargs) -> Dataset:
    """Get the dataset for the given task.

        Args:
            name [str]: name of the dataset
            data_dir [str]: directory to the saved dataset
            kwargs: additional arguments to pass to the dataset
        
        Returns:
            dataset [torch.utils.data.Dataset]: dataset for the given task
    """
    if 'data_dir' in kwargs:
        data_dir = kwargs['data_dir']
        kwargs.pop('data_dir')

    if name == 'tellu':
        dataset = Tellu
    elif name == 'orgaquant':
        dataset = OrgaQuant
    elif name == 'multiorg':
        dataset = MultiOrg

    else:
        raise ValueError(f'Name {name} not supported.')
    return dataset(data_dir=data_dir, **kwargs)