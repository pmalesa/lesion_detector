import logging

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from localizer.image_cache import ImageCache
from localizer.networks.dqn import DQN
from localizer.replay_buffer import ReplayBuffer
from numpy.typing import NDArray

logger = logging.getLogger("LESION-DETECTOR")


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
        self._target_update_steps = self._config.get("target_update_steps", 100)

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
        self._optimizer = Adam(learning_rate=self._learning_rate)

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

    def update(self):
        """
        Updates policy networks using transitions from replay buffer
        """

        if len(self._replay_buffer) < self._batch_size:
            return

        observations, actions, rewards, next_observations, dones = (
            self._replay_buffer.sample(self._batch_size)
        )

        # Prepare batch tensors
        observations = self._prepare_batch(observations)
        next_observations = self._prepare_batch(next_observations)

        # Compute DQN loss
        with tf.GradientTape() as tape:
            q = self._q_network(observations)

            # Create a 1D vector of integers from 0 to batch_size - 1
            batch_indices = tf.range(self._batch_size, dtype=tf.int32)

            # Stack the batch_indices vector with the corresponding performed actions
            indices = tf.stack([batch_indices, actions], axis=1)

            # Creates a vector of Q-values corresponding to performed actions
            chosen_q = tf.gather_nd(q, indices)

            q_next = self._target_network(next_observations)

            # Create a vector of highest Q-values from q_next batch
            q_next_max = tf.reduce_max(q_next, axis=1)

            target_q = rewards + (1 - dones) * self._gamma * q_next_max

            # Calculate MSE between the Q-value of the action taken and target Q-value
            loss = tf.reduce_mean((chosen_q - target_q) ** 2)

        # Backprop and update the network
        grads = tape.gradient(loss, self._q_network.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_network.trainable_variables))

        self._global_step += 1

        # Update target network once every x steps
        if self._global_step % self._target_update_steps == 0:
            self._target_network.set_weights(self._q_network.get_weights())

        # Update epsilon
        fraction = min(float(self._global_step) / self._epsilon_decay, 1.0)
        self._epsilon = self._epsilon_start + fraction * (
            self._epsilon_end - self._epsilon_start
        )

        return loss

    def _prepare_batch(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes observations as an array of tuples (bbox, img_name),
        and returns them as an array of tuples (img_tensor, bbox_tensor), where:
        - img_tensor has shape (batch, 512, 512, 3)
        - bbox_tensor has shape (batch, 4)
        """

        bbox_batch = []
        img_batch = []

        for i in range(len(observations)):
            (bbox, img_name) = observations[i]

            # Get image data from cache
            img_data = self._image_cache.get(img_name)

            # Replicate grayscale channel across 3 channels
            img_data = np.repeat(img_data[..., np.newaxis], 3, axis=-1)

            bbox_batch.append(bbox)
            img_batch.append(img_data)

        # Convert to numpy arrays
        img_tensor = np.array(img_batch, dtype=np.float32)
        bbox_tensor = np.array(bbox_batch, dtype=np.float32)

        return img_tensor, bbox_tensor
