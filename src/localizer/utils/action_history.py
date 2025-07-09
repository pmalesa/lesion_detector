from collections import deque

import numpy as np


class ActionHistory:
    def __init__(self, length=10, n_actions=9):
        self._length = length
        self._n_actions = n_actions
        self._history = deque(maxlen=length)
        self._one_hot_actions = np.eye(self._n_actions, dtype=np.float32)

    def add(self, action: int):
        if action < 0 or action > self._n_actions - 1:
            return
        one_hot_action = self._one_hot_actions[action]
        self._history.append(one_hot_action)

    def get_history_vector(self) -> np.ndarray:
        """
        Returns a flattened vector of concatenated one-hot encoded
        actions or zero-pads if there are fewer actions than the
        given length of the history.
        """

        one_hot_vectors = list(self._history)
        while len(one_hot_vectors) < self._length:
            one_hot_vectors.insert(0, np.zeros(self._n_actions, dtype=np.float32))

        return np.concatenate(one_hot_vectors, axis=0)

    def clear(self):
        self._history.clear()
