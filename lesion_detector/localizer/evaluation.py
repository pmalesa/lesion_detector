import logging

import numpy as np
from common.file_utils import load_metadata
from common.image_utils import create_image_paths, get_image_names
from localizer.environment import LocalizerEnv
from localizer.wrappers import CoordWrapper

logger = logging.getLogger("LESION-DETECTOR")


def evaluate_localizer(model, algorithm: str, config, seed=42, run_dir=None):
    """
    Runs a deterministic policy on all training
    images on a fresh environment and returns
    arrays of final IoUs and steps counts.
    """

    logger.info("Starting localizer evaluation.")

    if not run_dir:
        logger.info("Run directory must be specified.")

    # Load metadata
    metadata_path = config.get("metadata_path", "")
    dataset_metadata = load_metadata(metadata_path)

    # Prepare image paths
    val_image_names = get_image_names("val", dataset_metadata)
    test_image_names = get_image_names("test", dataset_metadata)
    image_names = val_image_names + test_image_names
    image_paths = create_image_paths(image_names, "../data/deeplesion/key_slices/")

    # Create and seed new environment
    render = config["environment"].get("render", False)
    threshold = config["environment"].get("iou_threshold", 0.5)
    env = LocalizerEnv(config, image_paths, dataset_metadata, seed)
    env = CoordWrapper(env)
    ious, steps = [], []

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

    ious = np.array(ious)
    steps = np.array(steps)

    metrics = {}
    metrics["mean_iou"] = round(ious.mean(), 4)
    metrics["std_iou"] = round(ious.std(), 4)
    metrics["mean_steps"] = round(steps.mean(), 4)
    metrics["std_steps"] = round(steps.std(), 4)
    metrics["success_rate"] = round((ious >= threshold).mean(), 4)

    logger.info(f"Evaluation metrics (algorithm: '{algorithm}', seed: '{seed}'):\n  {metrics}")

    return metrics
