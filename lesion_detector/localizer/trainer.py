import logging

import numpy as np
import pandas as pd
from common.file_utils import load_metadata
from common.logging_utils import append_log, create_run_dir, init_log, save_config
from localizer.agent import LocalizerAgent
from localizer.environment import LocalizerEnv
from tensorflow.config import list_physical_devices

logger = logging.getLogger("LESION-DETECTOR")


def train_localizer(config):
    show_available_devices()
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
        image_name = image_names[1666]  # TODO
        logger.info(f"Loaded image {i + 1}: {image_name}.")
        image_path = f"../data/deeplesion/key_slices/{image_name}"
        agent.reset()

        for episode in range(num_episodes):
            image_metadata = get_image_metadata(dataset_metadata, image_name)
            obs = env.reset(image_path, image_metadata)

            episode_reward = 0.0
            losses = []
            done = False
            info = {}

            while not done:
                mask = env.get_available_actions()
                action = agent.select_action(obs, mask)
                next_obs, reward, done, info = env.step(action)
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


def evaluate_localizer(config, model_weights_path: str):
    logger.info("Starting localizer testing.")

    metadata_path = config.get("metadata_path", "")
    dataset_metadata = load_metadata(metadata_path)

    env = LocalizerEnv(config)
    agent = LocalizerAgent(config)
    agent.load_model(model_weights_path)
    success_iou = config["test"]["success_iou"]

    iou_values = []
    dist_values = []
    steps_values = []
    success_episodes = 0

    # Iterate over all test key slice images
    image_names = get_image_names("test", dataset_metadata)

    for i, image_name in enumerate(image_names):
        image_path = f"../data/deeplesion/key_slices/{image_name}"
        image_metadata = get_image_metadata(dataset_metadata, image_name)

        obs = env.reset(image_path, image_metadata)
        done = False
        while not done:
            mask = env.get_available_actions()
            action = agent.select_action(obs, mask, training=False)
            next_obs, _, done, info = env.step(action)
            obs = next_obs

        if info["iou"] >= success_iou:
            success_episodes += 1

        iou_values.append(info["iou"])
        dist_values.append(info["dist"])
        steps_values.append(info["steps"])

        logger.info(
            f"Image {i + 1} ({image_name}) | "
            f"Final IoU: {info['iou']} | "
            f"Total steps: {info['steps']}"
        )

    logger.info("Localizer evaluation finished.")

    success_rate = round(success_episodes / len(image_names), 2)
    average_iou = round(np.mean(iou_values), 4)
    average_dist = round(np.mean(dist_values), 4)
    average_steps = np.mean(steps_values)

    logger.info(
        f"\n    Success rate: {success_rate}"
        f"\n    Average IoU: {average_iou}"
        f"\n    Average distance: {average_dist}"
        f"\n    Average steps: {average_steps}"
    )


def show_available_devices():
    """
    Prints the list of available devices.
    """

    message = f"\n  Number of GPUs available: {len(list_physical_devices('GPU'))}"
    message += "\n  Available devices:"
    for device in list_physical_devices():
        message += f"\n   - {device}"
    logger.info(message)


# TODO - move it to the image_utils.py file
def get_image_names(split_type_str: str, metadata: pd.DataFrame):
    """
    Returns a list of key slices' image names, given the split
    type (train, validation or test).
    """

    split_type = None
    match split_type_str:
        case "train":
            split_type = 1
        case "validation":
            split_type = 2
        case "test":
            split_type = 3
        case _:
            logger.error(
                "Wrong image type selected. Must be 'train', 'validation' or 'test'."
            )

    image_names = []
    for i in range(len(metadata)):
        if metadata["Train_Val_Test"][i] == split_type:
            image_names.append(metadata["File_name"][i])

    return image_names


# TODO - there can be multiple rows with the same image name (fix it)!
def get_image_metadata(metadata: pd.DataFrame, image_name: str):
    """
    Returns the dataframe of a single row from the whole
    metadata dataframe, given the image name.
    """

    return metadata.loc[metadata["File_name"] == image_name].iloc[0]
