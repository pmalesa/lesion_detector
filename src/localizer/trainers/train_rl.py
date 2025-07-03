import logging
import random
import warnings

import numpy as np
import pandas as pd
import torch
from common.file_utils import load_metadata
from common.image_utils import create_image_paths, get_image_names
from common.logging_utils import create_run_dir, init_log, save_config
from localizer.callbacks import RenderCallback
from localizer.environment import LocalizerEnv
from localizer.evaluation import evaluate_localizer
from localizer.networks.common import ResNet50CoordsExtractor
from localizer.wrappers import CoordWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy

warnings.filterwarnings(
    "ignore",
    message="It seems that your observation .* is an image but its `dtype` is",
    module="stable_baselines3.common.env_checker",
)
warnings.filterwarnings(
    "ignore",
    message="It seems that your observation space .* is an image but the upper and lower bounds",
    module="stable_baselines3.common.env_checker",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("LESION-DETECTOR")


def run_complete_training(config, algorithm: str):
    logger.info("Starting complete localizer training.")

    seeds = [0, 42, 123, 999, 2025]
    all_metrics = []

    # Initialize run's logs
    run_dir = create_run_dir(config, algorithm)
    header = ["mean_iou", "mean_steps", "success_rate"]
    csv_log_path = run_dir / "summary.csv"
    init_log(csv_log_path, header)
    save_config(run_dir, config)

    for seed in seeds:
        model, _, _ = train_single_localizer("dqn", config, seed, run_dir)
        metrics = evaluate_localizer(model, "dqn", config, seed, run_dir)
        all_metrics.append(
            {
                "seed": seed,
                "mean_iou": metrics["mean_iou"],
                "std_iou": metrics["std_iou"],
                "mean_steps": metrics["mean_steps"],
                "std_steps": metrics["std_steps"],
                "success_rate": metrics["success_rate"],
            }
        )

    df = pd.DataFrame(all_metrics)
    summary = df[header].agg(["mean", "std"])
    summary.to_csv(csv_log_path)
    logger.info(f"\n{summary}")

    logger.info("Localizer full training finished.")


def train_single_localizer(algorithm: str, config, seed=42, run_dir=None):

    logger.info(
        f"Starting localizer training (algorithm: '{algorithm}', seed: '{seed}')."
    )

    # Load metadata
    dataset_metadata = load_metadata(config.get("metadata_path", ""))

    # Create run directory if not given
    if not run_dir:
        run_dir = create_run_dir(config, algorithm)

    # Prepare image paths
    image_names = get_image_names(split_type="train", metadata=dataset_metadata)
    image_paths = create_image_paths(image_names, config.get("images_dir", ""))

    # Create and seed new environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = LocalizerEnv(config, image_paths, dataset_metadata, seed)
    env = CoordWrapper(env)
    check_env(env)
    monitor_logs_path = run_dir / f"{algorithm}_seed_{seed}_monitor.csv"
    env = Monitor(
        env,
        str(monitor_logs_path),
        info_keywords=("iou", "dist")    
    )

    # Set hyperparameters
    train_steps = config["train"].get("train_steps", 1_000_000)
    weights_path = config.get("backbone_cnn_path", "")
    config = config["agent"]
    learning_rate = config.get("learning_rate", 1e-4)
    n_steps = config.get("n_steps", 3)
    discount_factor = config.get("discount_factor", 0.9)
    tau = config.get("tau", 1.0)
    epsilon_start = config.get("epsilon_start", 1.0)
    epsilon_end = config.get("epsilon_end", 0.01)
    epsilon_decay = config.get("epsilon_decay", 0.25)
    replay_buffer_size = config.get("replay_buffer_size", 100_000)
    batch_size = config.get("batch_size", 32)
    target_update_steps = config.get("target_update_steps", 1000)
    train_freq = config.get("train_freq", 1)

    policy_kwargs = dict(
        features_extractor_class=ResNet50CoordsExtractor,
        features_extractor_kwargs={
            "features_dim": 2048 + 4,
            "weights_path": weights_path,
            "device": device
        },
        net_arch=[512, 256],  # Q-Network architecture
        normalize_images=False,
    )

    model = None
    if algorithm == "dqn":
        model = DQN(
            policy=MultiInputPolicy,
            env=env,
            seed=seed,
            exploration_initial_eps=epsilon_start,
            exploration_final_eps=epsilon_end,
            exploration_fraction=epsilon_decay,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=1,
            buffer_size=replay_buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=discount_factor,
            target_update_interval=target_update_steps,
            train_freq=train_freq,
            verbose=1,
            tensorboard_log=None,
        )
    else:
        raise Exception(f"There is no such algorithm as '{algorithm}'")

    model.learn(
        total_timesteps=train_steps,
        callback=RenderCallback(render_freq=1)
    )
    model_path = run_dir / f"{algorithm}_seed_{seed}_dynamic"
    model.save(model_path)

    logger.info(
        f"Localizer training complete (aglorithm: '{algorithm}', seed: '{seed}')."
    )
    logger.info(f"Localizer model saved to '{run_dir}'.")

    return model, run_dir, seed