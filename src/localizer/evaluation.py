import logging

import numpy as np
import pandas as pd
import torch
from stable_baselines3.dqn import DQN

from common.file_utils import extract_directory, load_metadata
from common.image_utils import create_image_paths, get_image_names
from localizer.environment import LocalizerEnv
from localizer.wrappers import CoordWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("LESION-DETECTOR")


def evaluate_localizer(model_path: str, algorithm: str, config, seed=42):
    """
    Runs a deterministic policy on all training
    images on a fresh environment and returns
    arrays of final IoUs and steps counts.
    """

    logger.info("Starting localizer evaluation.")

    # Load metadata
    dataset_metadata = load_metadata(config.get("metadata_path", ""))

    # Prepare image paths
    image_names = get_image_names("test", dataset_metadata)
    image_paths = create_image_paths(image_names, config.get("images_dir", ""))

    # Create and seed new environment
    render = config["environment"].get("render", False)
    threshold = config["environment"].get("iou_threshold", 0.5)
    env = LocalizerEnv(config, image_paths, dataset_metadata, seed)
    env = CoordWrapper(env)
    ious, steps = [], []

    model = DQN.load(model_path, env=env)

    for _ in image_paths:
        obs, _ = env.reset(seed=seed)
        if render:
            env.render()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if render:
                env.render()
            done = terminated or truncated

        ious.append(info["iou"])
        steps.append(info["steps"])
        logger.info(f"Episode ended — terminated={terminated} info={info}")

    logger.info("Localizer evaluation complete.")

    ious_arr = np.array(ious)
    steps_arr = np.array(steps)

    metrics = {}
    metrics["mean_iou"] = round(ious_arr.mean(), 4)
    metrics["std_iou"] = round(ious_arr.std(), 4)
    metrics["mean_steps"] = round(steps_arr.mean(), 4)
    metrics["std_steps"] = round(steps_arr.std(), 4)
    metrics["success_rate"] = round((ious_arr >= threshold).mean(), 4)

    run_dir = extract_directory(model_path)
    csv_log_path = run_dir / f"summary_{algorithm}_seed_{seed}.csv"
    summary = pd.DataFrame([metrics])
    summary.to_csv(csv_log_path, index=False)

    logger.info(
        f"Evaluation metrics (algorithm: '{algorithm}', seed: '{seed}') saved to {csv_log_path}."
    )
    logger.info(
        f"Evaluation metrics (algorithm: '{algorithm}', seed: '{seed}'):\n  {metrics}"
    )

    return metrics
