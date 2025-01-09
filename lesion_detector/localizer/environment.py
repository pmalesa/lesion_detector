import numpy as np
from common.image_utils import load_image, load_metadata
from numpy.typing import NDArray


class LocalizerEnv:
    ACTIONS = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "RESIZE_LARGER",  # TODO - INCREASE_HEIGHT and INCREASE_WIDTH
        5: "RESIZE_SMALLER",  # TODO - DECREASE_HEIGHT and DECREASE_WIDTH
        6: "MARK",
        7: "FINISH",
    }

    INITIAL_BBOX_HEIGHT = 20
    INITIAL_BBOX_WIDTH = 20

    def __init__(self, config):
        self.__config = config["environment"]
        self.__metadata_path = self.__config.get("metadata_file", "")
        self.__dataset_path = self.__config.get("dataset_path", "")
        self.__metadata = load_metadata(self.__metadata_path)

        self.__image = None
        self.__img_height = None
        self.__img_width = None

        self.__bbox_w = None
        self.__bbox_h = None
        self.__bbox_x = None
        self.__bbox_y = None

    def reset(self):
        # Load image
        self.__image = load_image("../data/test_img.png")  # TODO
        self.__img_height = self.__image.shape[0]
        self.__img_width = self.__image.shape[1]

        # Reset bounding box parameters
        self.__bbox_w = self.INITIAL_BBOX_WIDTH
        self.__bbox_h = self.INITIAL_BBOX_HEIGHT
        self.__bbox_x = self.__img_width / 2 - self.__bbox_w / 2
        self.__bbox_y = self.__img_height / 2 - self.__bbox_h / 2

        return self.__get_observation()

    def step(self, action):
        # Apply action (move bounding box, etc.)
        # Return next_observation, reward, done, info

        # -----
        # TO REMOVE
        next_obs = self.__obs + 0.5
        reward = 1.0
        done = False
        info = {}
        # -----

        next_obs = self.__get_observation()

        return next_obs, reward, done, info

    def __get_observation(self) -> NDArray[np.float32]:
        pass

    # TODO
    def render(self):
        pass

    # -----
    # TO REMOVE
    def test(self):
        # test_df = self.__metadata["Image_size"].apply(
        #     lambda x: [int(item.strip()) for item in x.split(",")]
        # )

        print(self.__bbox_x)
        print(self.__bbox_y)

    # -----
