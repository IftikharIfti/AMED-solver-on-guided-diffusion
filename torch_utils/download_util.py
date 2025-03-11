import os
from tqdm import tqdm
import urllib.request

urls = {
    "imagenet256": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
    "imagenet256-classifier": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
}

def search_local_model(key, subsubdir="src"):
    contents = os.listdir('../')
    subdirs = [item for item in contents if os.path.isdir(os.path.join('../', item))]
    url = urls[key]

    for subdir in subdirs:
        target_dir = os.path.join('../', subdir, subsubdir, key)
        if os.path.exists(target_dir):
            download_path = model_path = os.path.join(target_dir, url.split("/")[-1])
            if os.path.exists(model_path):
                return True, download_path, model_path
    download_path = os.path.join('./', subsubdir, key, url.split("/")[-1])
    return False, download_path, None

def download_model(url, download_path):
    target_dir = os.path.dirname(download_path)
    os.makedirs(target_dir, exist_ok=True)
    with open(download_path, 'wb') as file, tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=int(urllib.request.urlopen(url).getheader('Content-Length').strip()), desc=download_path) as bar:
        urllib.request.urlretrieve(url, download_path, reporthook=lambda block_num, block_size, _: bar.update(block_size))
    return download_path

def check_file_by_key(key, subsubdir="src"):
    if key not in urls:
        raise ValueError(f"Unknown key: {key}")
    exist_local_model, download_path, model_path = search_local_model(key, subsubdir)
    if exist_local_model:
        print(f'Model already exists: {model_path}')
    else:
        url = urls[key]
        print(f'Downloading from {url}')
        model_path = download_model(url, download_path)

    model_path_extra = None
    if key == "imagenet256" and "classifier" not in key:
        key_extra = "imagenet256-classifier"
        exist_local_model, download_path, model_path_extra = search_local_model(key_extra, subsubdir)
        if exist_local_model:
            print(f'Classifier already exists: {model_path_extra}')
        else:
            url = urls[key_extra]
            print(f'Downloading classifier from {url}')
            model_path_extra = download_model(url, download_path)
    return model_path, model_path_extra