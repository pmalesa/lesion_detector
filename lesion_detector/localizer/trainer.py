import logging

from common.file_utils import extract_filename, load_metadata
from localizer.agent import LocalizerAgent
from localizer.environment import LocalizerEnv
from tensorflow.config import list_physical_devices

logger = logging.getLogger("LESION-DETECTOR")


def train_localizer(config):
    show_available_devices()
    logger.info("Starting localizer training.")

    metadata_path = config.get("metadata_path", "")
    dataset_metadata = load_metadata(metadata_path)
    env = LocalizerEnv(config, render=True)
    agent = LocalizerAgent(config)
    config = config["train"]
    num_episodes = config.get("train_episodes", 1000)

    # TODO - change so that it iterates over some/all training images
    image_path = "../data/000001_03_01_088.png"
    image_name = extract_filename(image_path)

    for episode in range(num_episodes):
        image_metadata = dataset_metadata.loc[
            dataset_metadata["File_name"] == image_name
        ].iloc[0]

        obs = env.reset(image_path, image_metadata)

        episode_reward = 0.0
        done = False
        info = {}
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_experience((obs, action, reward, next_obs, done))
            agent.update()
            episode_reward += reward
            obs = next_obs
        logger.info(f"Episode {episode + 1} - Reward: {round(episode_reward, 2)}")

    logger.info("Localizer training finished.")


def test_localizer(config):
    logger = logging.getLogger("LESION-DETECTOR")
    logger.info("Starting localizer testing.")

    env = LocalizerEnv(config)
    agent = LocalizerAgent(config)
    config = config["test"]

    num_episodes = config.get("test_episodes", 10)
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        logger.info(f"Test Episode {episode} - Reward: {episode_reward}")

    logger.info("Localizer testing finished.")


def show_available_devices():
    message = f"\n  Number of GPUs available: {len(list_physical_devices('GPU'))}"
    message += "\n  Available devices:"
    for device in list_physical_devices():
        message += f"\n   - {device}"
    logger.info(message)
