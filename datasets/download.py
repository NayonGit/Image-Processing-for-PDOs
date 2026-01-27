import gdown
import zipfile
import requests
from tqdm import tqdm

# adapted from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str,
             fname: str,
             chunk_size: int=1024) -> None:
    """Download a file from the given URL and save it to the given file name.
        
        Args:
            url [str]: URL to download the file from
            fname [str]: file name to save the downloaded file to
            chunk_size [int]: size of the chunks to download the file in
        
        Returns:
            None
    """
    if 'drive.google.com' in url:
        gdown.download(url, fname)
    else:
        headers = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/120.0 Safari/537.36')}
        resp = requests.get(url, headers=headers, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(desc=fname,
                                            total=total,
                                            unit='iB',
                                            unit_scale=True,
                                            unit_divisor=1024,
                                            leave=False) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

def unzip(zip_path: str,
          unzip_path: str) -> None:
    """Unzip the file at the given path to the given directory.
        
        Args:
            zip_path [str]: path to the zip file
            unzip_path [str]: path to the directory to unzip the file to
        
        Returns:
            None
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)