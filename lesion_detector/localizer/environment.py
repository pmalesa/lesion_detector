import cv2
import numpy as np
import pandas as pd
from common.image_utils import load_image
from common.metrics import dist, iou
from numpy.typing import NDArray


class LocalizerEnv:
    ACTIONS = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "INCREASE_WIDTH",
        5: "DECREASE_WIDTH",
        6: "INCREASE_HEIGHT",
        7: "DECREASE_HEIGHT",
        8: "FINISH",
    }

    def __init__(self, config):
        self._config = config["environment"]
        self._render = self._config["render"]
        self._image_metadata = None

        self._max_steps = self._config["max_steps"]
        self._initial_bbox_width = self._config["initial_bbox_width"]
        self._initial_bbox_height = self._config["initial_bbox_height"]
        self._bbox_pixel_margin = self._config["bbox_pixel_margin"]
        self._min_bbox_length = self._config["min_bbox_length"]
        self._move_step_factor = self._config["bbox_move_step_factor"]
        self._resize_factor = self._config["bbox_resize_factor"]
        self._iou_threshold = self._config["iou_threshold"]

        # Reward function weights
        self._alpha = self._config["reward"]["alpha"]
        self._beta = self._config["reward"]["beta"]
        self._step_penalty = self._config["reward"]["step_penalty"]
        self._iou_final_reward = self._config["reward"]["iou_final_reward"]

        self._image_data = None
        self._image_name = None
        self._image_height = None
        self._image_width = None

        # Maximal distance between two points in the image
        self._max_dist = None

        self._current_step = None

        # Bounding boxes are np.arrays of the form [x, y, width, height]
        self._bbox = None
        self._target_bbox = None

    def reset(self, image_path: str, image_metadata: pd.DataFrame):
        """
        Resets the environment and returns the image pixel data
        and the coordinates of the target bounding box
        """

        self._init_image_data(image_path, image_metadata)
        self._init_target_bbox()
        self._reset_bbox()
        self._current_step = 1

        if self._render:
            cv2.namedWindow("Rendered Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rendered Image", 600, 600)

        return self._get_observation()

    def step(self, action_id):
        """
        Performs a step, given a specific action ID.
        """
        action = self.ACTIONS[action_id]

        match action:
            case "MOVE_UP":
                move_step = self._calculate_move_step(self._bbox[3])
                self._bbox[1] = max(0, self._bbox[1] - move_step)
            case "MOVE_DOWN":
                move_step = self._calculate_move_step(self._bbox[3])
                self._bbox[1] = min(self._image_height - 1, self._bbox[1] + move_step)
            case "MOVE_LEFT":
                move_step = self._calculate_move_step(self._bbox[2])
                self._bbox[0] = max(0, self._bbox[0] - move_step)
            case "MOVE_RIGHT":
                move_step = self._calculate_move_step(self._bbox[2])
                self._bbox[0] = min(self._image_width - 1, self._bbox[0] + move_step)
            case "INCREASE_WIDTH":
                bbox_resize_step = self._calculate_resize_step(self._bbox[2])
                self._bbox[2] = min(
                    self._bbox[2] + bbox_resize_step,
                    self._image_width - self._bbox[0],
                )
            case "DECREASE_WIDTH":
                bbox_resize_step = self._calculate_resize_step(self._bbox[2])
                self._bbox[2] = max(
                    self._bbox[2] - bbox_resize_step, self._min_bbox_length
                )
            case "INCREASE_HEIGHT":
                bbox_resize_step = self._calculate_resize_step(self._bbox[3])
                self._bbox[3] = min(
                    self._bbox[3] + bbox_resize_step,
                    self._image_height - self._bbox[1],
                )
            case "DECREASE_HEIGHT":
                bbox_resize_step = self._calculate_resize_step(self._bbox[3])
                self._bbox[3] = max(
                    self._bbox[3] - bbox_resize_step, self._min_bbox_length
                )
            case "FINISH":
                done = True
                iou_val = iou(self._bbox, self._target_bbox)

                iou_additional_reward = 0.0
                if iou_val >= self._iou_threshold:
                    iou_additional_reward += self._iou_final_reward
                else:
                    iou_additional_reward -= self._iou_final_reward

                # Maybe add additional big reward for distance
                # Or a negative reward for IoU below threshold
                # ...

                info = {"bbox": self._bbox, "steps": self._current_step}

                # if self._render:
                #     cv2.destroyAllWindows()

                return (
                    self._get_observation(),
                    self._compute_reward() + iou_additional_reward,
                    done,
                    info,
                )

        next_obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {} if not done else {"bbox": self._bbox, "steps": self._current_step}
        self._current_step += 1

        if self._render:
            self._render_bboxes()

        return next_obs, reward, done, info

    def get_available_actions(self) -> np.ndarray:
        """
        Returns a 1D vector/mask of actions that are avaiable to
        be perfomed by the agent at a given step. It consists of
        values either 0 or 1, where 1 at a given index denotes an
        action that can be performed and 0 denotes an unavailable
        action. This vector/mask has length equal to the number
        of possible actions.
        """

        possible_actions = [
            self._can_move_up(),
            self._can_move_down(),
            self._can_move_left(),
            self._can_move_right(),
            self._can_increase_width(),
            self._can_decrease_width(),
            self._can_increase_height(),
            self._can_decrease_height(),
            True,
        ]

        return np.array(possible_actions, dtype=np.int32)

    def _render_bboxes(self):
        """
        Renders the image with both current and target bounding boxes.
        """

        # Rescale the image to 8-bit precision
        gray_8bit = np.clip(self._image_data * 255, 0, 255).astype(np.uint8)

        # Convert 1-channel 8-bit image to a 3-channel color image
        rgb = cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR)

        # Extract bounding box coordinates and draw boxes
        x, y, w, h = self._target_bbox
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)

        x, y, w, h = self._bbox
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        cv2.imshow("Rendered Image", rgb)
        cv2.waitKey(10)

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Returns the observation as a tuple containing
        cropped image from the current bounding box
        and a vector of 10 previous actions.
        """

        (x, y, w, h) = self._bbox

        x_1 = max(0, x - self._bbox_pixel_margin)
        y_1 = max(0, y - self._bbox_pixel_margin)

        x_2 = min(self._image_width - 1, x + w + self._bbox_pixel_margin)
        y_2 = min(self._image_height - 1, y + h + self._bbox_pixel_margin)

        return self._image_data[y_1:y_2, x_1:x_2]

    def _compute_reward(self) -> float:
        """
        Computes the reward, that includes three components:
        * Component proportional to the IoU metric,
        * Component inversely proportional to the distance between centers,
        * Step penalty component.
        """

        iou_val = iou(self._bbox, self._target_bbox)

        # Compute and normalize distance between centers
        dist_val = dist(self._bbox, self._target_bbox)
        dist_val_norm = dist_val / self._max_dist

        iou_reward = self._alpha * iou_val
        center_reward = self._beta * (1.0 - dist_val_norm)

        return iou_reward + center_reward + self._step_penalty

    def _init_image_data(self, image_path: str, image_metadata: pd.DataFrame):
        """
        Initializes the image data (pixel data, name and dimensions)
        """

        self._image_metadata = image_metadata
        self._image_data = load_image(image_path, norm=True)
        self._image_name = self._image_metadata["File_name"]
        self._image_height = self._image_data.shape[0]
        self._image_width = self._image_data.shape[1]
        self._max_dist = np.sqrt(self._image_width**2 + self._image_height**2)

    def _reset_bbox(self):
        """
        Resets current bounding box' parameters.
        """

        w = self._initial_bbox_width
        h = self._initial_bbox_height
        x = int((self._image_width - w) / 2)
        y = int((self._image_height - h) / 2)
        self._bbox = np.array([x, y, w, h])

    def _check_done(self) -> bool:
        """
        Checks whether the episode should end.
        """

        return self._current_step >= self._max_steps

    def _init_target_bbox(self):
        """
        Calculates ground truth bounding box' parameters.
        """

        target_bbox_str = self._image_metadata["Bounding_boxes"]
        target_bbox_coords = [float(val) for val in target_bbox_str.split(",")]
        x1, y1, x2, y2 = [int(c) for c in target_bbox_coords]
        w, h = x2 - x1, y2 - y1
        self._target_bbox = np.array([x1, y1, w, h])

    def _normalize_bbox(self, bbox: np.ndarray) -> NDArray[np.float32]:
        """
        Normalizes the bounding box' coordinates and sizes.
        """

        x_norm = bbox[0] / self._image_width
        y_norm = bbox[1] / self._image_height
        w_norm = bbox[2] / self._image_width
        h_norm = bbox[3] / self._image_height

        return np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32)

    def _calculate_move_step(self, bbox_length: int) -> int:
        """
        Calculates the move step according to the defined move step
        factor parameter and given bbox_length (either width or height).
        Smallest possible move step is equal to 1 pixel.
        """

        return max(1, round(bbox_length * self._move_step_factor))

    def _calculate_resize_step(self, bbox_length: int) -> int:
        """
        Calculates the resize step according to the defined resize step
        factor parameter and given bbox_length (either width or height).
        Smallest possible resize step is equal to 1 pixel.
        """

        return max(1, round(bbox_length * self._resize_factor))

    def _can_move_up(self) -> bool:
        return self._bbox[1] > 0

    def _can_move_down(self) -> bool:
        return self._bbox[1] + self._bbox[3] < self._image_height - 1

    def _can_move_left(self) -> bool:
        return self._bbox[0] > 0

    def _can_move_right(self) -> bool:
        return self._bbox[0] + self._bbox[2] < self._image_width - 1

    def _can_increase_width(self) -> bool:
        return self._bbox[0] + self._bbox[2] < self._image_width - 1

    def _can_decrease_width(self) -> bool:
        return self._bbox[2] > self._min_bbox_length

    def _can_increase_height(self) -> bool:
        return self._bbox[1] + self._bbox[3] < self._image_height - 1

    def _can_decrease_height(self) -> bool:
        return self._bbox[3] > self._min_bbox_length
