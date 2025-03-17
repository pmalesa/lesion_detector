import numpy as np
from keras.optimizers import Adam
from localizer.image_cache import ImageCache
from localizer.networks.dqn import DQN
from localizer.replay_buffer import ReplayBuffer
from numpy.typing import NDArray


class LocalizerAgent:
    def __init__(self, config):
        data_dir = config.get("data_dir", "")
        self._config = config["agent"]

        # Initialize hyperparameters
        self._gamma = self._config.get("gamma", 0.99)
        self._batch_size = self._config.get("batch_size", 64)
        self._learning_rate = self._config.get("learning_rate", 1e-4)
        self._epsilon_start = self._config.get("epsilon_start", 1.0)
        self._epsilon_end = self._config.get("epsilon_end", 0.1)
        self._epsilon_decay = self._config.get("epsilon_decay", 10000)

        # Initialize replay buffer and cache
        capacity = self._config.get("replay_buffer_size", 100000)
        self._replay_buffer = ReplayBuffer(capacity)

        cache_size = self._config.get("image_cache_size", 1000)
        self._image_cache = ImageCache(data_dir, cache_size)

        self._global_step = 0
        self._epsilon = self._epsilon_start

        # Initialize agent's network
        self._q_network = DQN(action_dim=9)
        self._target_network = DQN(action_dim=9)
        self._optimizer = Adam(learning_rate=1e-4)

    def select_action(self, obs: tuple[NDArray[np.float32], str]):
        (bbox, image_name) = obs

        # Epsilon-greedy policy
        if np.random.rand() < self._epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
        else:
            # Image data shape is (512, 512)
            image_data = self._image_cache.get(image_name)

            # Convert into (512, 512, 3), by replicating single channel
            image_data = np.repeat(image_data[..., np.newaxis], 3, axis=-1)

            # Convert into (1, 512, 512, 3), by adding batch dimension
            image_input = np.expand_dims(image_data, axis=0)

            bbox_input = np.array(bbox)[None, ...]
            q_values = self._q_network((image_input, bbox_input))
            action = np.argmax(q_values)

        return action

    def store_experience(self, experience):
        self._replay_buffer.store(experience)

    # TODO
    def update(self):
        """
        Updates policy networks using transitions from replay buffer
        """

        return

        if len(self._replay_buffer) < self._batch_size:
            return

        observations, actions, rewards, next_obesrvations, dones = (
            self._replay_buffer.sample(self._batch_size)
        )

        # for i in range(self._batch_size):
        #     (bbox, image_name) =

        # TODO
        # Perform DQN-style update
        # 1) current Q = Q_network(observations)[range(batch_size), actions]
        # 2) target Q =
        #     rewards + gamma * max Q_network(next_observations) * (1 - dones)
        # 3) MSE loss between current Q and target Q
        # 4) optimizer step

        # Update epsilon
        self._global_step += 1
        fraction = min(float(self._global_step) / self._epsilon_decay, 1.0)
        self._epsilon = self._epsilon_start + fraction * (
            self._epsilon_end - self._epsilon_start
        )
