import os
from configs import default_data_dir, dataset_config
from datasets import prepare_utils
from datasets.download import download, unzip

def prepare_dataset(dataset_name: str,
                    should_download: bool = True,
                    download_dir: str = '/tmp',
                    data_dir: str = default_data_dir) -> None:
    """Download and prepare the dataset with the given name.
        
        Args:
            dataset_name [str]: name of the dataset to prepare
            should_download [bool]: whether to download the dataset
            download_dir [str]: directory to download the dataset to
            data_dir [str]: directory to save the prepared dataset to

        Returns:
            None
    """


    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    download_url = dataset_config[dataset_name]['url']
    if isinstance(download_url, str):
        download_name = os.path.join(download_dir, f'{dataset_name}.zip')
        unziped_name = download_name.replace('.zip', '')
        if should_download:
            download(dataset_config[dataset_name].url, download_name)
            unzip(download_name, unziped_name)
    else:
        unziped_name = []
        for url in dataset_config[dataset_name].url:
            url, download_name = url
            download_name = os.path.join(download_dir, download_name)
            unziped_n = download_name.replace('.zip', '')
            unziped_name.append(unziped_n)
            if should_download:
                download(url, download_name)
                unzip(download_name, unziped_n)

    output_path = os.path.join(data_dir, f'{dataset_name}.h5')
    prepare_func = getattr(prepare_utils, f'prepare_{dataset_name}')
    prepare_func(unziped_name, output_path)
