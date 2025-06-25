import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

import yaml


def create_run_dir(config) -> Path:
    """
    Creates a timestamped directory for run's logs and outputs.
    """

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config["train"]["output_dir"]) / run_id
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_log(path: str, header: List[str]):
    """
    Initializes the CSV file used to store metrics and results.
    """

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_log(path: str, ep: int, iou: float, dist: float, loss: float, reward: float):
    """
    Appends a row of given values to the CSV file.
    """

    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep, iou, dist, loss, reward])


def save_config(path: str, config):
    """
    Saves the config file with all hyperparameters used in a given run.
    """

    with open(Path(path) / "config.yaml", "w") as f:
        yaml.dump(config, f)
