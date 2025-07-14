import random

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray

from common.file_utils import extract_filename
from common.image_utils import get_image_metadata, load_image
from common.metrics import dist, iou


class LocalizerEnv(gym.Env):
    ACTIONS = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "INCREASE_SIZE",
        5: "DECREASE_SIZE",
        6: "INCREASE_ASPECT_RATIO",
        7: "DECREASE_ASPECT_RATIO",
        8: "FINISH",
    }

    def __init__(self, config, image_paths, dataset_metadata, seed=42):
        super().__init__()
        self._image_paths = image_paths
        self._dataset_metadata = dataset_metadata
        self._image_metadata = None
        self._idx = -1
        self._seed = seed

        self._fixed_patch_length = config["agent"].get("fixed_patch_length", 128)
        self._config = config["environment"]
        self._render = self._config.get("render", False)

        self._max_steps = self._config.get("max_steps", 100)
        self._initial_bbox_width = self._config.get("initial_bbox_width", 50)
        self._initial_bbox_height = self._config.get("initial_bbox_height", 50)
        self._bbox_pixel_margin = self._config.get("bbox_pixel_margin", 10)
        self._bbox_min_length = self._config.get("bbox_min_length", 10)
        self._bbox_max_length = self._config.get("bbox_max_length", 64)
        self._bbox_max_aspect_ratio = self._config.get("bbox_max_aspect_ratio", 3.0)
        self._bbox_randomize = self._config.get("bbox_randomize", True)
        self._bbox_position_shift_range = self._config.get(
            "bbox_position_shift_range", 50.0
        )
        self._bbox_size_shift_range = self._config.get("bbox_size_shift_range", 10.0)
        self._move_step_factor = self._config.get("bbox_move_step_factor", 0.1)
        self._resize_factor = self._config.get("bbox_resize_factor", 0.1)
        self.iou_terminate_threshold = self._config.get("iou_terminate_threshold", 0.8)

        # Reward function weights
        self._config = self._config["reward"]
        self._alpha_1 = self._config.get("alpha_1", 5.0)
        self._alpha_2 = self._config.get("alpha_2", 2.0)
        self._beta = self._config.get("beta", 2.0)
        self._step_penalty = self._config.get("step_penalty", 0.05)
        self._iou_final_reward = self._config.get("iou_final_reward", 10.0)
        self._illegal_action_penalty = self._config.get("illegal_action_penalty", 2.0)

        self._prev_dist = None
        self._prev_iou = None

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

        # Action and observation space
        self.action_space = spaces.Discrete(len(self.ACTIONS), seed=self._seed)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, self._fixed_patch_length, self._fixed_patch_length),
            dtype=np.float32,
            seed=self._seed,
        )

    def reset(self, *, seed=None, options=None) -> NDArray[np.float32]:
        """
        Resets the environment by choosing the
        next image and returns the image pixel
        data and the coordinates of the target bounding box.
        """

        super().reset(seed=self._seed)

        self._idx = (self._idx + 1) % len(self._image_paths)
        image_path = self._image_paths[self._idx]

        image_name = extract_filename(image_path)
        image_metadata = get_image_metadata(self._dataset_metadata, image_name)

        self._init_image_data(image_path, image_metadata)
        self._reset_bbox()

        self._current_step = 0
        self._prev_dist = None
        self._prev_iou = None

        if self._render:
            cv2.namedWindow("Rendered Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rendered Image", 600, 600)

        return self._get_observation(), {}

    def step(self, action_id):
        """
        Performs a step, given a specific action ID.
        """

        self._current_step += 1

        # Handle illegal actions
        mask = self.get_available_actions()

        action = None
        if mask[action_id] != 0:
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
            case "INCREASE_SIZE":
                self._increase_size()
            case "DECREASE_SIZE":
                self._decrease_size()
            case "INCREASE_ASPECT_RATIO":
                self._increase_aspect_ratio()
            case "DECREASE_ASPECT_RATIO":
                self._decrease_aspect_ratio()
            case "FINISH":
                obs = self._get_observation()
                reward = self._compute_reward(final=True, timeout=False)
                terminated = True
                truncated = False
                info = {
                    "bbox": self._bbox,
                    "steps": self._current_step,
                    "iou": round(self._get_iou(), 4),
                    "dist": round(self._get_distance(), 4),
                }
                return obs, reward, terminated, truncated, info

        obs = self._get_observation()
        terminated = self._check_done()
        truncated = False
        action_valid = True if action else False
        reward = self._compute_reward(
            action_valid=action_valid, final=False, timeout=terminated
        )
        info = {}
        if terminated:
            info = {
                "bbox": self._bbox,
                "steps": self._current_step,
                "iou": round(self._get_iou(), 4),
                "dist": round(self._get_distance(), 4),
            }

        return obs, reward, terminated, truncated, info

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
            self._can_increase_size(),
            self._can_decrease_size(),
            self._can_increase_aspect_ratio(),
            self._can_decrease_aspect_ratio(),
            True,
        ]

        return np.array(possible_actions, dtype=np.int32)

    def render(self):
        """
        Renders the image with both current and target bounding boxes.
        """

        if not self._render:
            return

        # Rescale the image to 8-bit precision
        gray_8bit = np.clip(self._image_data * 255, 0, 255).astype(np.uint8)

        # Convert 1-channel 8-bit image to a 3-channel color image
        rgb = cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR)

        # Extract bounding box coordinates and draw boxes
        x, y, w, h = self._target_bbox
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        x, y, w, h = self._bbox
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        cv2.imshow("Rendered Image", rgb)
        cv2.waitKey(1)

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Returns the observation as a tuple containing cropped
        image of shape (1, H, W) from the current bounding box (+ margin).
        """

        return np.expand_dims(self._get_patch_with_margin(), axis=0)

    def _get_patch_with_margin(self) -> NDArray[np.float32]:
        """
        Returns 2D np.array of pixel data from within the current
        bounding box with additional margin (10 pixels by default),
        resized to a fixed size (128 by 128 pixels by default).
        """

        (x, y, w, h) = self._bbox

        x_1 = max(0, x - self._bbox_pixel_margin)
        y_1 = max(0, y - self._bbox_pixel_margin)

        x_2 = min(self._image_width - 1, x + w + self._bbox_pixel_margin)
        y_2 = min(self._image_height - 1, y + h + self._bbox_pixel_margin)

        patch_data = self._image_data[y_1:y_2, x_1:x_2]

        return self._resize_patch(patch_data)

    def _resize_patch(self, img_patch: np.ndarray) -> NDArray[np.float32]:
        """
        Resizes the cropped image patch into a fixed sized patch.
        If the size is already correct, then the input image
        patch is returned with no resizing.
        """

        fixed_shape = (self._fixed_patch_length, self._fixed_patch_length)
        if img_patch.shape == fixed_shape:
            return img_patch
        resized = cv2.resize(
            img_patch, fixed_shape, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        return np.clip(resized, 0.0, 1.0)

    def _compute_reward(
        self, action_valid: bool = True, final: bool = False, timeout: bool = False
    ) -> float:
        """
        Computes the reward (including bonus), that is based on
        the IoU metric (+1/-1) and step penalty.
        """

        iou_val = self._get_iou()
        reward = 0.0
        step_penalty = self._step_penalty if iou_val < 0.5 else 5 * self._step_penalty
        if action_valid:
            delta_iou_reward = 0.0
            if self._prev_iou:
                delta_iou_reward = iou_val - self._prev_iou
            reward = np.sign(delta_iou_reward)
        else:
            reward -= self._illegal_action_penalty

        reward -= step_penalty

        # Compute the bonus component
        if iou_val >= 0.8:
            final = True
        if final:
            if iou_val >= 0.75:
                reward += (
                    self._iou_final_reward if timeout else 2 * self._iou_final_reward
                )
            elif iou_val >= 0.5:
                reward += (
                    0.5 * self._iou_final_reward if timeout else self._iou_final_reward
                )
            # else:
            #     reward -= self._iou_final_reward

        return reward

    def _get_iou(self) -> float:
        """
        Computes and returns the current IoU metric value.
        """

        return iou(self._bbox, self._target_bbox)

    def _get_distance(self) -> float:
        """
        Computes and returns the current normalized distance metric value.
        """

        dist_val = dist(self._bbox, self._target_bbox)
        return dist_val / self._max_dist

    def _init_image_data(self, image_path: str, image_metadata: pd.DataFrame):
        """
        Initializes the image data (pixel data, name and dimensions)
        """

        self._image_metadata = image_metadata
        self._image_data = load_image(image_path, image_metadata, norm=True)
        self._image_name = self._image_metadata["File_name"]
        self._image_height = self._image_data.shape[0]
        self._image_width = self._image_data.shape[1]

        # Initialize target bounding box
        target_bbox_str = self._image_metadata["Bounding_boxes"]
        target_bbox_coords = [float(val) for val in target_bbox_str.split(",")]
        x1, y1, x2, y2 = [round(c) for c in target_bbox_coords]

        # Rescale to (512 x 512) if necessary
        if (self._image_height, self._image_width) != (512, 512):
            self._image_data = cv2.resize(
                self._image_data, (512, 512), interpolation=cv2.INTER_AREA
            )
            scale_x = 512 / self._image_width
            scale_y = 512 / self._image_height
            x1 = round(x1 * scale_x)
            y1 = round(y1 * scale_y)
            x2 = round(x2 * scale_x)
            y2 = round(y2 * scale_y)
            self._image_height = 512
            self._image_width = 512

        self._max_dist = np.sqrt(self._image_width**2 + self._image_height**2)
        w, h = x2 - x1, y2 - y1
        self._target_bbox = np.array([x1, y1, w, h])

    def _reset_bbox(self):
        """
        Resets current bounding box' parameters.
        """

        # Calculate random size and position shifts
        random_size_shift = (
            random.randint(-self._bbox_size_shift_range, self._bbox_size_shift_range)
            if self._bbox_randomize
            else 0
        )
        random_x_shift = (
            random.randint(
                -self._bbox_position_shift_range, self._bbox_position_shift_range
            )
            if self._bbox_randomize
            else 0
        )
        random_y_shift = (
            random.randint(
                -self._bbox_position_shift_range, self._bbox_position_shift_range
            )
            if self._bbox_randomize
            else 0
        )

        w = self._initial_bbox_width + random_size_shift
        h = self._initial_bbox_height + random_size_shift

        x = round((self._image_width - w) / 2) + random_x_shift
        y = round((self._image_height - h) / 2) + random_y_shift
        self._bbox = np.array([x, y, w, h])

    def _check_done(self) -> bool:
        """
        Checks whether the episode should end.
        """

        condition_1 = self._current_step >= self._max_steps
        condition_2 = self._get_iou() >= self.iou_terminate_threshold

        return bool(condition_1 or condition_2)

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
        Smallest possible resize step is equal to 1 pixel. It is
        multiplied by 0.5, because this resize step is used two times,
        once to move the top edge and senond time to move the bottom one
        (or right edge and left edge).
        """

        return max(1, round(0.5 * bbox_length * self._resize_factor))

    def _increase_size(self):
        w, h = self._bbox[2], self._bbox[3]
        if w < self._bbox_max_length and h < self._bbox_max_length:
            self._increase_width()
            self._increase_height()

    def _decrease_size(self):
        w, h = self._bbox[2], self._bbox[3]
        if w > self._bbox_min_length and h > self._bbox_min_length:
            self._decrease_width()
            self._decrease_height()

    def _increase_aspect_ratio(self):
        w, h = self._bbox[2], self._bbox[3]
        if w / h < self._bbox_max_aspect_ratio:
            if w < self._bbox_max_length:
                self._increase_width()
            elif h > self._bbox_min_length:
                self._decrease_height()

    def _decrease_aspect_ratio(self):
        w, h = self._bbox[2], self._bbox[3]
        if w / h > (1 / self._bbox_max_aspect_ratio):
            if w > self._bbox_min_length:
                self._decrease_width()
            elif h < self._bbox_max_length:
                self._increase_height()

    def _increase_width(self):
        x, w = self._bbox[0], self._bbox[2]
        step = self._calculate_resize_step(w)

        x_new = max(0, x - step)
        w_new = min(self._bbox_max_length, w + 2 * step)

        if x_new + w_new > self._image_width:
            w_new = self._image_width - x_new

        self._bbox[0] = x_new
        self._bbox[2] = w_new

    def _decrease_width(self):
        x, w = self._bbox[0], self._bbox[2]
        step = self._calculate_resize_step(w)

        x_new = x + step
        w_new = max(self._bbox_min_length, w - 2 * step)

        if x_new + w_new > self._image_width:
            w_new = self._image_width - x_new

        self._bbox[0] = x_new
        self._bbox[2] = w_new

    def _increase_height(self):
        y, h = self._bbox[1], self._bbox[3]
        step = self._calculate_resize_step(h)

        y_new = max(0, y - step)
        h_new = min(self._bbox_max_length, h + 2 * step)

        if y_new + h_new > self._image_height:
            h_new = self._image_height - y_new

        self._bbox[1] = y_new
        self._bbox[3] = h_new

    def _decrease_height(self):
        y, h = self._bbox[1], self._bbox[3]
        step = self._calculate_resize_step(h)

        y_new = y + step
        h_new = max(self._bbox_min_length, h - 2 * step)

        if y_new + h_new > self._image_height:
            h_new = self._image_height - y_new

        self._bbox[1] = y_new
        self._bbox[3] = h_new

    def _can_move_up(self) -> bool:
        return self._bbox[1] > 0

    def _can_move_down(self) -> bool:
        return self._bbox[1] + self._bbox[3] < self._image_height - 1

    def _can_move_left(self) -> bool:
        return self._bbox[0] > 0

    def _can_move_right(self) -> bool:
        return self._bbox[0] + self._bbox[2] < self._image_width - 1

    def _can_increase_size(self) -> bool:
        condition_1 = self._bbox[2] < self._bbox_max_length
        condition_2 = self._bbox[3] < self._bbox_max_length
        return condition_1 and condition_2

    def _can_decrease_size(self) -> bool:
        condition_1 = self._bbox[2] > self._bbox_min_length
        condition_2 = self._bbox[3] > self._bbox_min_length
        return condition_1 and condition_2

    # TODO - POPRAW
    def _can_increase_aspect_ratio(self) -> bool:
        w, h = self._bbox[2], self._bbox[3]
        return w / h < self._bbox_max_aspect_ratio

    # TODO - POPRAW
    def _can_decrease_aspect_ratio(self) -> bool:
        w, h = self._bbox[2], self._bbox[3]
        return w / h > (1 / self._bbox_max_aspect_ratio)
