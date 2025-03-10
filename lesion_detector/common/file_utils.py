from pathlib import Path

import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def extract_filename(path: str):
    full_path = Path(path)
    return full_path.name
