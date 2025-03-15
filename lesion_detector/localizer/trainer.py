import logging

from common.file_utils import extract_filename, load_metadata
from localizer.agent import LocalizerAgent
from localizer.environment import LocalizerEnv


# TODO
def train_localizer(config):
    logger = logging.getLogger("LESION-DETECTOR")
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
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_experience((obs, action, reward, next_obs, done))
            agent.update()
            episode_reward += reward
            obs = next_obs
        logger.info(f"Episode {episode} - Reward: {episode_reward}")

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
