import logging

import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from localizer.networks.dqn import DQN
from localizer.replay_buffer import ReplayBuffer
from numpy.typing import NDArray

logger = logging.getLogger("LESION-DETECTOR")


class LocalizerAgent:
    def __init__(self, config):
        self._config = config["agent"]

        # Initialize hyperparameters
        self._gamma = self._config.get("gamma", 0.99)
        self._batch_size = self._config.get("batch_size", 64)
        self._learning_rate = self._config.get("learning_rate", 1e-4)
        self._epsilon_start = self._config.get("epsilon_start", 1.0)
        self._epsilon_end = self._config.get("epsilon_end", 0.1)
        self._epsilon_decay = self._config.get("epsilon_decay", 10000)
        self._target_update_steps = self._config.get("target_update_steps", 100)
        self._image_patch_size = self._config.get("image_patch_size", 128)

        # Initialize replay buffer and cache
        capacity = self._config.get("replay_buffer_size", 100000)
        self._replay_buffer = ReplayBuffer(capacity)

        self._global_step = 0
        self._epsilon = self._epsilon_start

        # Initialize agent's network
        image_shape = (self._image_patch_size, self._image_patch_size, 3)
        self._q_network = DQN(action_dim=9, image_shape=image_shape)
        self._target_network = DQN(action_dim=9, image_shape=image_shape)
        self._optimizer = Adam(learning_rate=self._learning_rate)

    def select_action(self, obs: NDArray[np.float32], mask: np.ndarray):
        # Epsilon-greedy policy
        if np.random.rand() < self._epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
        else:
            # Image data shape is (512, 512)
            resized_patch = self._resize_patch(obs)

            # Replicate patch across three channels
            patch_data = np.repeat(resized_patch[..., np.newaxis], 3, axis=-1)

            # Convert into (1, patch_width, patch_height, 3), by adding batch dimension
            patch_input = np.expand_dims(patch_data, axis=0)

            # TODO - Add vector of 10 previous acitons
            q_values = self._q_network(patch_input)

            # Convert the tensorflow tensor into numpy array
            q_values = q_values.numpy()[0]

            # Apply the mask to the q_values vector
            q_values[mask == 0] = -1.0e9

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

    def _prepare_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Takes observations as an array of image patches of various size,
        and returns them as an array of fixed-sized image patches (128 x 128 pixels).
        """

        resized_observations = []
        for img_patch in observations:
            resized_patch = self._resize_patch(img_patch)

            # Replicate grayscale channel across 3 channels
            resized_patch = np.repeat(resized_patch[..., np.newaxis], 3, axis=-1)

            resized_observations.append(resized_patch)

        return np.array(resized_observations, dtype=np.float32)

    def _resize_patch(self, img_patch: np.ndarray) -> np.ndarray:
        """
        Resizes the cropped image patch into a fixed sized patch.
        """

        patch_shape = (self._image_patch_size, self._image_patch_size)
        return cv2.resize(img_patch, patch_shape, interpolation=cv2.INTER_AREA)
