import numpy as np
from gymnasium import ObservationWrapper, spaces

class CoordWrapper(ObservationWrapper):
    """
    Takes an environment whose observation is a single array and
    turns it into a Dict(image=array, coords=array) with additional
    normalized bounding box coordinates.
    """

    def __init__(self, env):
        super().__init__(env)

        # Original observation space
        img_space = env.observation_space

        # New observation space
        coord_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"image": img_space, "coords": coord_space}
        )

    def observation(self, obs):
        coords = self.env._normalize_bbox(self.env._bbox)
        return {"image": obs, "coords": coords}
