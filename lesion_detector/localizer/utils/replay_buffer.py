import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        A simple buffer to store experiences as
        (obs, action, reward, next_obs, done) tuples.
        """

        self._buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store single experience tuple (obs, action, reward, next_obs, done).
        """

        self._buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences randomly.
        Returns a list of tuples (obs, action, reward, next_obs, done).
        """

        batch = random.sample(self._buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            np.array(observations, dtype=object),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_observations, dtype=object),
            np.array(dones, dtype=np.int32),
        )

    def __len__(self):
        return len(self._buffer)
