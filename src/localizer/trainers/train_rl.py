import logging
import random
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy

from common.file_utils import load_metadata
from common.image_utils import create_image_paths, get_image_names
from common.logging_utils import create_run_dir, init_log, save_config
from localizer.callbacks import RenderCallback
from localizer.environment import LocalizerEnv
from localizer.evaluation import evaluate_localizer
from localizer.networks.common import ResNet50CoordsExtractor, ResNet50Extractor
from localizer.wrappers import CoordWrapper

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


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_available_actions()


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
        model_path, _ = train_single_localizer("dqn", config, seed, run_dir)
        metrics = evaluate_localizer(model_path, "dqn", config, seed)
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
    save_config(run_dir, config)

    # Prepare image paths
    image_names = get_image_names(split_type="train", metadata=dataset_metadata)
    image_names_val = get_image_names(
        split_type="validation", metadata=dataset_metadata
    )
    image_paths = create_image_paths(image_names, config.get("images_dir", ""))
    image_paths_val = create_image_paths(image_names_val, config.get("images_dir", ""))
    image_names.extend(image_names_val)
    image_paths.extend(image_paths_val)

    # Create and seed new environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = None
    env = LocalizerEnv(config, image_paths, dataset_metadata, seed)

    # Set hyperparameters
    train_steps = config["train"].get("train_steps", 1_000_000)
    weights_path = Path(config.get("backbone_cnn_path", ""))
    agent_config = config["agent"]
    learning_rate = agent_config.get("learning_rate", 1e-4)
    # n_steps = agent_config.get("n_steps", 3)
    discount_factor = agent_config.get("discount_factor", 0.9)
    tau = agent_config.get("tau", 1.0)
    epsilon_start = agent_config.get("epsilon_start", 1.0)
    epsilon_end = agent_config.get("epsilon_end", 0.01)
    epsilon_decay = agent_config.get("epsilon_decay", 0.25)
    replay_buffer_size = agent_config.get("replay_buffer_size", 100_000)
    batch_size = agent_config.get("batch_size", 32)
    target_update_steps = agent_config.get("target_update_steps", 1000)
    train_freq = agent_config.get("train_freq", 1)

    if algorithm == "dqn":
        env = CoordWrapper(env)
        policy_kwargs = dict(
            features_extractor_class=ResNet50CoordsExtractor,
            features_extractor_kwargs={
                "features_dim": 2048 + 4,  # ResNet50 output size + 4 bbox coords
                "weights_path": weights_path,
                "device": device,
            },
            net_arch=[512, 256],  # Q-Network architecture
            normalize_images=False,
        )
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
            device=device,
        )
    elif algorithm == "ppo":
        env = LocalizerEnv(config, image_paths, dataset_metadata, seed)
        env = ActionMasker(env, mask_fn)
        policy_kwargs = dict(
            features_extractor_class=ResNet50Extractor,
            features_extractor_kwargs={
                "features_dim": 2048,
                "weights_path": weights_path,
                "device": device,
            },
            net_arch=[512, 256],  # Q-Network architecture
            normalize_images=False,
        )
        model = MaskablePPO(
            policy=MaskableActorCriticCnnPolicy,
            env=env,
            seed=seed,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            ent_coef=1e-3,
            n_steps=256,
            n_epochs=4,
            batch_size=batch_size,
            gamma=discount_factor,
            verbose=1,
            tensorboard_log=None,
            device=device,
        )
    else:
        raise Exception(f"There is no such algorithm as '{algorithm}'")

    check_env(env)
    monitor_logs_path = run_dir / f"{algorithm}_seed_{seed}_monitor.csv"
    env = Monitor(env, str(monitor_logs_path), info_keywords=("iou", "dist"))

    # TODO - ADD CALLBACK TO SAVE CHECKPOINTS EVERY 10k steps
    model.learn(total_timesteps=train_steps, callback=RenderCallback(render_freq=1))
    model_path = run_dir / f"{algorithm}_seed_{seed}_dynamic"
    model.save(model_path)

    logger.info(
        f"Localizer training complete (aglorithm: '{algorithm}', seed: '{seed}')."
    )
    logger.info(f"Localizer model saved to '{run_dir}'.")

    return model_path, seed
