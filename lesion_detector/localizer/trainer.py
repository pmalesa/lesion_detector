import logging

from localizer.agent import LocalizerAgent
from localizer.environment import LocalizerEnv


# TODO
def train_localizer(config):
    logger = logging.getLogger("LESION-DETECTOR")
    logger.info("Starting localizer training.")

    env = LocalizerEnv(config)
    agent = LocalizerAgent(config)
    config = config["train"]

    num_episodes = config.get("train_episodes", 100)
    for episode in range(num_episodes):
        obs = env.reset()
        break  # TODO
        done = True
        # done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_experience((obs, action, reward, next_obs, done))
            agent.update()
            episode_reward += reward
            obs = next_obs
        logger.info(f"Episode {episode} - Reward: {episode_reward}")

    logger.info("Localizer training finished.")

    # -----

    env.test()

    # -----


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
