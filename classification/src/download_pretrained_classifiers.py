from utils import download
from pathlib import Path

main_dir = Path('classification/output')

urls = {
    'mlp': {
        'model': 'https://zenodo.org/records/13909366/files/mlp.pt?download=1',
        'config': 'https://zenodo.org/records/13909366/files/mlp_config.pt?download=1'
    },
    'single_transformer': {
        'model': 'https://zenodo.org/records/13909366/files/mlp.pt?download=1',
        'config': 'https://zenodo.org/records/13909366/files/mlp_config.pt?download=1'
    },
    'multi_transformer': {
        'model': 'https://zenodo.org/records/13909366/files/mlp.pt?download=1',
        'config': 'https://zenodo.org/records/13909366/files/mlp_config.pt?download=1'
    },
}

# MLP model
for model in ('mlp', 'single_transformer', 'multi_transformer'):
    folder = main_dir / model
    folder.mkdir(exist_ok=True)

    model_url = f'https://zenodo.org/records/13909366/files/{model}.pt?download=1'
    model_target_path = folder / 'model.pt'
    download(model_url, model_target_path, is_zenodo=True)

    config_url = f'https://zenodo.org/records/13909366/files/{model}_config.pt?download=1'
    config_target_path = folder / 'config.pt'
    download(config_url, config_target_path, is_zenodo=True)

