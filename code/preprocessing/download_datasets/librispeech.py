'''
cannot use this script because spp bans connection to other sites.
had to download librispeech manually on dung's laptop.
'''

import os
import tarfile
import requests
from tqdm import tqdm

# Base URL for downloading the LibriSpeech dataset
_DL_URL = "http://www.openslr.org/resources/12/"

# Define the dataset subsets and their corresponding URLs
_DL_URLS = {
    "clean": {
        "dev": _DL_URL + "dev-clean.tar.gz",
        "test": _DL_URL + "test-clean.tar.gz",
        "train.100": _DL_URL + "train-clean-100.tar.gz",
        "train.360": _DL_URL + "train-clean-360.tar.gz",
    },
    "other": {
        "test": _DL_URL + "test-other.tar.gz",
        "dev": _DL_URL + "dev-other.tar.gz",
        "train.500": _DL_URL + "train-other-500.tar.gz",
    },
    "all": {
        "dev.clean": _DL_URL + "dev-clean.tar.gz",
        "dev.other": _DL_URL + "dev-other.tar.gz",
        "test.clean": _DL_URL + "test-clean.tar.gz",
        "test.other": _DL_URL + "test-other.tar.gz",
        "train.clean.100": _DL_URL + "train-clean-100.tar.gz",
        "train.clean.360": _DL_URL + "train-clean-360.tar.gz",
        "train.other.500": _DL_URL + "train-other-500.tar.gz",
    },
}

def download_file(url, save_path):
    """Download a file with progress bar."""
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0))
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")
    
    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            bar.update(size)

def extract_tarfile(filepath, extract_path):
    """Extract a tar.gz file with progress bar."""
    with tarfile.open(filepath, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=f"Extracting {os.path.basename(filepath)}") as bar:
            for member in members:
                tar.extract(member, path=extract_path)
                bar.update(1)

def download_librispeech(dl_urls, output_dir="librispeech_data", extract=True):
    """Download and optionally extract LibriSpeech dataset files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use only the "all" subset for downloading
    urls = dl_urls["all"]
    downloaded_files = []
    
    for name, url in urls.items():
        # Create the download path
        save_path = os.path.join(output_dir, f"{name}.tar.gz")
        
        # Download if file doesn't exist
        if not os.path.exists(save_path):
            print(f"Downloading {name} from {url}...")
            try:
                download_file(url, save_path)
                downloaded_files.append(save_path)
            except Exception as e:
                print(f"Error downloading {name}: {str(e)}")
                if os.path.exists(save_path):
                    os.remove(save_path)
                continue
        else:
            print(f"{name} already exists. Skipping download.")
            downloaded_files.append(save_path)
        
        # Extract if requested
        if extract:
            extract_dir = os.path.join(output_dir, name)
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    extract_tarfile(save_path, extract_dir)
                except Exception as e:
                    print(f"Error extracting {name}: {str(e)}")
                    continue
            else:
                print(f"Directory {name} already exists. Skipping extraction.")

    return downloaded_files

if __name__ == "__main__":
    # Download and extract all datasets
    output_dir = '../data/speech_prompts/librispeech'
    downloaded_files = download_librispeech(_DL_URLS, output_dir=output_dir, extract=True)
