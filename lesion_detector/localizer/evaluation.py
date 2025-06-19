import logging

# import numpy as np
# from common.file_utils import load_metadata
# from common.image_utils import get_image_metadata, get_image_names

# from localizer.agent import LocalizerAgent
# from localizer.environment import LocalizerEnv

logger = logging.getLogger("LESION-DETECTOR")


def evaluate_localizer(config, model_weights_path: str):
    pass


# def evaluate_localizer_custom(config, model_weights_path: str):
#     logger.info("Starting localizer evaluation.")

#     metadata_path = config.get("metadata_path", "")
#     dataset_metadata = load_metadata(metadata_path)

#     env = LocalizerEnv(config)
#     agent = LocalizerAgent(config)
#     agent.load_model(model_weights_path)
#     success_iou = config["test"]["success_iou"]

#     iou_values = []
#     dist_values = []
#     steps_values = []
#     success_episodes = 0

#     # Iterate over all test key slice images
#     image_names = get_image_names("test", dataset_metadata)

#     for i, image_name in enumerate(image_names):
#         image_path = f"../data/deeplesion/key_slices/{image_name}"
#         image_metadata = get_image_metadata(dataset_metadata, image_name)

#         obs = env.reset(image_path, image_metadata)
#         env.render()
#         done = False
#         while not done:
#             mask = env.get_available_actions()
#             action = agent.select_action(obs, mask, training=False)
#             next_obs, _, done, info = env.step(action)
#             env.render()
#             obs = next_obs

#         if info["iou"] >= success_iou:
#             success_episodes += 1

#         iou_values.append(info["iou"])
#         dist_values.append(info["dist"])
#         steps_values.append(info["steps"])

#         logger.info(
#             f"Image {i + 1} ({image_name}) | "
#             f"Final IoU: {info['iou']} | "
#             f"Total steps: {info['steps']}"
#         )

#     logger.info("Localizer evaluation finished.")

#     success_rate = round(success_episodes / len(image_names), 2)
#     average_iou = round(np.mean(iou_values), 4)
#     average_dist = round(np.mean(dist_values), 4)
#     average_steps = np.mean(steps_values)

#     logger.info(
#         f"\n    Success rate: {success_rate}"
#         f"\n    Average IoU: {average_iou}"
#         f"\n    Average distance: {average_dist}"
#         f"\n    Average steps: {average_steps}"
#     )
