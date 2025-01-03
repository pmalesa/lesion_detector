import numpy as np


class LocalizerEnv:
    def __init__(self, config):
        # Load images, metadata, etc.
        self.__config = config["environment"]
        self.__initial_obs = np.array([])
        self.__obs = np.array([])

        pass

    def reset(self):
        # Reset environment state
        # Return initial observation
        self.__obs = self.__initial_obs
        return self.__obs

    def step(self, action):
        # Apply action (move bounding box, etc.)
        # Return next_observation, reward, done, info

        # TO REMOVE
        next_obs = self.__obs + 0.5
        reward = 1.0
        done = False
        info = {}

        return next_obs, reward, done, info

    # TODO
    def render(self):
        pass
