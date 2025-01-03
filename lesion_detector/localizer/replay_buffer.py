import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        A simple buffer to store experiences as
        (obs, action, reward, next_obs, done) tuples.
        """

        self.__buffer = deque(maxlen=capacity)

    def store(self, experience):
        """
        Store single experience tuple (obs, action, reward, next_obs, done).
        """

        self.__buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences randomly.
        Returns a list of tuples (obs, action, reward, next_obs, done).
        """

        batch = random.sample(self.__buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(next_observations),
            np.array(dones),
        )

    def __len__(self):
        return len(self.__buffer)
