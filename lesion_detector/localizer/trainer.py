import logging
import warnings

# import numpy as np
from common.file_utils import load_metadata
from common.image_utils import get_image_names, get_image_paths
from localizer.callbacks import RenderCallback

# from common.logging_utils import append_log, create_run_dir, init_log, save_config
from localizer.environment import LocalizerEnv
from localizer.networks.common import ResNet50Extractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN

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

logger = logging.getLogger("LESION-DETECTOR")


def train_localizer(config):
    logger.info("Starting localizer training.")

    # Load metadata
    metadata_path = config.get("metadata_path", "")
    dataset_metadata = load_metadata(metadata_path)

    # Prepare image paths
    image_names = get_image_names("train", dataset_metadata)
    image_paths = get_image_paths(image_names, "../data/deeplesion/key_slices/")

    env = LocalizerEnv(config, image_paths, dataset_metadata)
    check_env(env)
    env = Monitor(env, "logs/")

    train_steps = config["train"].get("train_steps", 200_000)
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
        features_extractor_class=ResNet50Extractor,
        features_extractor_kwargs=dict(features_dim=2048),
        net_arch=[512, 256],  # Q-Network architecture
        normalize_images=False,
    )

    model = DQN(
        policy="CnnPolicy",
        env=env,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        exploration_fraction=epsilon_decay,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        buffer_size=replay_buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=discount_factor,
        target_update_interval=target_update_steps,
        train_freq=train_freq,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(
        total_timesteps=train_steps,
        callback=RenderCallback(render_freq=1),
    )
    model.save("zoom_in_dqn")

    logger.info("Localizer training finished.")


# def train_localizer_custom(config):
#     logger.info("Starting localizer training (CUSTOM).")

#     metadata_path = config.get("metadata_path", "")
#     dataset_metadata = load_metadata(metadata_path)

#     env = LocalizerEnv(config)
#     agent = LocalizerAgent(config)
#     log_interval = config["train"]["log_interval"]
#     num_episodes = config["train"]["train_episodes"]

#     # Initialize run's logs
#     run_dir = create_run_dir(config)
#     csv_log_path = run_dir / "training_log.csv"
#     init_log(csv_log_path)
#     save_config(run_dir, config)

#     # Iterate over all training key slice images
#     image_names = get_image_names("train", dataset_metadata)
#     for i, image_name in enumerate(image_names):
#         # image_name = image_names[1666]  # TODO
#         image_name = "000001_03_01_088.png"
#         logger.info(f"Loaded image {i + 1}: {image_name}.")
#         image_path = f"../data/deeplesion/key_slices/{image_name}"
#         agent.reset()

#         for episode in range(num_episodes):
#             image_metadata = get_image_metadata(dataset_metadata, image_name)
#             obs = env.reset(image_path, image_metadata)
#             env.render()

#             episode_reward = 0.0
#             losses = []
#             done = False
#             info = {}

#             while not done:
#                 mask = env.get_available_actions()
#                 action = agent.select_action(obs, mask)
#                 next_obs, reward, done, info = env.step(action)
#                 env.render()
#                 agent.store_experience((obs, action, reward, next_obs, done))
#                 loss = agent.update()
#                 episode_reward += reward
#                 if loss:
#                     losses.append(float(loss))
#                 obs = next_obs

#             mean_loss = round(np.mean(losses), 2) if losses else 0.0
#             episode_reward = round(episode_reward, 2)
#             append_log(
#                 csv_log_path,
#                 episode,
#                 info["iou"],
#                 info["dist"],
#                 mean_loss,
#                 episode_reward,
#             )

#             if episode % log_interval == 0:
#                 logger.info(
#                     f"Episode {episode + 1} | "
#                     f"Mean Loss: {mean_loss} | "
#                     f"Reward: {episode_reward} | "
#                     f"Steps: {info['steps']}"
#                 )

#         break  # TODO

#     # Save run's results
#     agent.save_model(str(run_dir / "model.keras"))
#     logger.info("Localizer training finished (CUSTOM).")
