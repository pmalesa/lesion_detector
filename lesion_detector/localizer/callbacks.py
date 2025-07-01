import cv2
from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    """
    A callback that periodically calls env.render()
    """

    def __init__(self, render_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_freq == 0:
            env = self.training_env.envs[0]
            env.render()
            cv2.waitKey(1)
        return True
