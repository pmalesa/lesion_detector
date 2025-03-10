import numpy as np
from common.file_utils import extract_filename
from common.image_utils import load_image, load_metadata, show_image
from common.metrics import iou
from numpy.typing import NDArray


class LocalizerEnv:
    ACTIONS = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "RESIZE_LARGER",  # TODO
        5: "RESIZE_SMALLER",  # TODO
        6: "MARK",
        7: "FINISH",
    }

    def __init__(self, config):
        self._config = config["environment"]
        self._metadata_path = self._config.get("metadata_file", "")
        self._dataset_path = self._config.get("dataset_path", "")
        self._metadata = load_metadata(self._metadata_path)

        self._initial_bbox_width = self._config["initial_bbox_width"]
        self._initial_bbox_height = self._config["initial_bbox_height"]
        self._bbox_move_step = self._config["bbox_move_step"]

        self._image_data = None
        self._image_name = None
        self._img_height = None
        self._img_width = None

        # Bounding boxes are np.arrays of the form [x, y, width, height]
        self._current_bbox = None
        self._target_bbox = None

    def reset(self, image_path: str):
        self._init_image_data(image_path)
        self._reset_bbox()
        self._calculate_target_bbox()
        return self._get_observation()

    def step(self, action_id):
        action = self.ACTIONS[action_id]

        if action == "MOVE_UP":
            self._bbox_y = max(0, self._bbox_y - self._bbox_move_step)
        elif action == "MOVE_DOWN":
            pass
        elif action == "MOVE_LEFT":
            pass
        elif action == "MOVE_RIGHT":
            pass
        elif action == "RESIZE_LARGER":
            pass
        elif action == "RESIZE_SMALLER":
            pass
        elif action == "MARK":
            pass
        elif action == "FINISH":
            done = True
            return self._get_observation(), self._compute_reward(), done, {}

        next_obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}  # TODO - Maybe store there info about marked bounding boxes

        return next_obs, reward, done, info

    def _get_observation(self) -> dict[str, NDArray[np.float32]]:
        return {"image_data": self._image_data, "bbox": self._current_bbox}

    # TODO
    def _compute_reward(self) -> float:
        # ...

        return 1.0

    def _init_image_data(self, image_path: str):
        """
        Initializes the image data (pixel data, name and dimensions)
        """

        self._image_data = load_image(image_path)  # TODO - Add normalization
        self._image_name = extract_filename(image_path)
        self._img_height = self._image_data.shape[0]
        self._img_width = self._image_data.shape[1]

    def _reset_bbox(self):
        """
        Resets current bounding box' parameters.
        """

        w = self._initial_bbox_width
        h = self._initial_bbox_height
        x = (self._img_width - w) / 2
        y = (self._img_height - h) / 2
        self._current_bbox = np.array([x, y, w, h], dtype=np.float32)

    def _calculate_target_bbox(self):
        """
        Calculates ground truth bounding box' parameters.
        """

        target_bbox_str = self._metadata.loc[
            self._metadata["File_name"] == self._image_name, "Bounding_boxes"
        ].iloc[0]
        target_bbox_coords = [float(val) for val in target_bbox_str.split(",")]
        x1, y1, x2, y2 = [int(c) for c in target_bbox_coords]
        w, h = x2 - x1, y2 - y1
        self._target_bbox = np.array([x1, y1, w, h], dtype=np.float32)

    # TODO
    def _check_done(self) -> bool:
        return False

    # TODO
    def render(self):
        show_image(self._image_data)

    # -----------------------------------------------------------------
    # TODO - TO REMOVE
    def test(self):
        print(self._image_name)
        print(self._target_bbox)
        print(self._current_bbox)
        print(f"IoU: {iou(self._current_bbox, self._target_bbox)}")

    # -----------------------------------------------------------------
