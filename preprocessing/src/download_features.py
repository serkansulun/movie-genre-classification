from utils import download

url = "https://zenodo.org/records/13909366/files/features.pkl?download=1"
target_path = 'preprocessing/data/features.pkl'
download(url, target_path, is_zenodo=True)