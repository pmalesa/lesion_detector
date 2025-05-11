import logging

import numpy as np
from common.file_utils import load_metadata
from common.image_utils import get_image_metadata, get_image_names
from common.logging_utils import append_log, create_run_dir, init_log, save_config
from localizer.agent import LocalizerAgent
from localizer.environment import LocalizerEnv

logger = logging.getLogger("LESION-DETECTOR")


def train_localizer(config):
    logger.info("Starting localizer training.")

    metadata_path = config.get("metadata_path", "")
    dataset_metadata = load_metadata(metadata_path)

    env = LocalizerEnv(config)
    agent = LocalizerAgent(config)
    log_interval = config["train"]["log_interval"]
    num_episodes = config["train"]["train_episodes"]

    # Initialize run's logs
    run_dir = create_run_dir(config)
    csv_log_path = run_dir / "training_log.csv"
    init_log(csv_log_path)
    save_config(run_dir, config)

    # Iterate over all training key slice images
    image_names = get_image_names("train", dataset_metadata)
    for i, image_name in enumerate(image_names):
        # image_name = image_names[1666]  # TODO
        image_name = "000001_03_01_088.png"
        logger.info(f"Loaded image {i + 1}: {image_name}.")
        image_path = f"../data/deeplesion/key_slices/{image_name}"
        agent.reset()

        for episode in range(num_episodes):
            image_metadata = get_image_metadata(dataset_metadata, image_name)
            obs = env.reset(image_path, image_metadata)
            env.render()

            episode_reward = 0.0
            losses = []
            done = False
            info = {}

            while not done:
                mask = env.get_available_actions()
                action = agent.select_action(obs, mask)
                next_obs, reward, done, info = env.step(action)
                env.render()
                agent.store_experience((obs, action, reward, next_obs, done))
                loss = agent.update()
                episode_reward += reward
                if loss:
                    losses.append(float(loss))
                obs = next_obs

            mean_loss = round(np.mean(losses), 2) if losses else 0.0
            episode_reward = round(episode_reward, 2)
            append_log(
                csv_log_path,
                episode,
                info["iou"],
                info["dist"],
                mean_loss,
                episode_reward,
            )

            if episode % log_interval == 0:
                logger.info(
                    f"Episode {episode + 1} | "
                    f"Mean Loss: {mean_loss} | "
                    f"Reward: {episode_reward} | "
                    f"Steps: {info['steps']}"
                )

        break  # TODO

    # Save run's results
    agent.save_model(str(run_dir / "model.keras"))
    logger.info("Localizer training finished.")
