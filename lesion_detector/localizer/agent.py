import logging

import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from localizer.networks.dqn import DQN
from localizer.utils.replay_buffer import ReplayBuffer
from numpy.typing import NDArray
from tensorflow.keras.models import load_model

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
        self._fixed_patch_length = self._config.get("fixed_patch_length", 64)

        # Initialize replay buffer and cache
        capacity = self._config.get("replay_buffer_size", 10000)
        self._replay_buffer = ReplayBuffer(capacity)

        self._global_step = 0
        self._epsilon = self._epsilon_start

        # Initialize agent's network
        image_shape = (self._fixed_patch_length, self._fixed_patch_length, 3)
        self._q_network = DQN(action_dim=9, image_shape=image_shape)
        self._target_network = DQN(action_dim=9, image_shape=image_shape)
        self._optimizer = Adam(learning_rate=self._learning_rate)

    def reset(self):
        """
        Resets the agent's epsilon value and clears the replay buffer.
        """

        self._global_step = 0
        self._epsilon = self._epsilon_start

    def select_action(
        self, obs: NDArray[np.float32], mask: np.ndarray, training: bool = True
    ):
        # Epsilon-greedy policy (only during training)
        if training and np.random.rand() < self._epsilon:
            allowed_actions = np.where(mask == 1)[0]
            action = np.random.choice(allowed_actions)
        else:
            (img_patch, prev_actions) = obs

            # Resize patch to a fixed size (64, 64)
            resized_patch = self._resize_patch(img_patch)

            # Replicate patch across three channels (64, 64, 3)
            patch_data = np.repeat(resized_patch[..., np.newaxis], 3, axis=-1)

            # Convert into (1, 64, 64, 3), by adding batch dimension
            patch_input = np.expand_dims(patch_data, axis=0)

            # Convert into (1, 90), by adding batch dimension
            prev_actions_input = np.expand_dims(prev_actions, axis=0)

            q_values = self._q_network((patch_input, prev_actions_input))

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
            # loss = tf.reduce_mean((chosen_q - target_q) ** 2)

            # Calculate the Huber loss between the Q-value of the action taken and target Q-value
            loss = tf.reduce_mean(tf.keras.losses.huber(chosen_q, target_q))

        # Backprop, clip gradients and update the network
        grads = tape.gradient(loss, self._q_network.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
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

        if self._global_step % 100 == 0 and self._global_step <= self._epsilon_decay:
            logger.info(f"* EPSILON: {round(self._epsilon, 2)} *")

        return loss

    def save_model(self, path: str):
        self._q_network.save(path)

    def load_model(self, path: str):
        self._q_network = load_model(path)
        self._target_network.set_weights(self._q_network.get_weights())

    def _prepare_batch(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes observations as an array of tuples with image patch of various size
        and a vector of history of previous actions. It returns them as tuple with
        two arrays (resized_patch_tensor, prev_actions_tensor), where the resized
        patch has a fixed-sized (128 x 128 pixels by default).
        """

        resized_patch_batch = []
        prev_actions_batch = []

        for obs in observations:
            (img_patch, prev_actions) = obs

            resized_patch = self._resize_patch(img_patch)

            # Replicate grayscale channel across 3 channels
            resized_patch = np.repeat(resized_patch[..., np.newaxis], 3, axis=-1)

            resized_patch_batch.append(resized_patch)
            prev_actions_batch.append(prev_actions)

        resized_patch_tensor = np.array(resized_patch_batch, dtype=np.float32)
        prev_actions_tensor = np.array(prev_actions_batch, dtype=np.float32)

        return resized_patch_tensor, prev_actions_tensor

    def _resize_patch(self, img_patch: np.ndarray) -> np.ndarray:
        """
        Resizes the cropped image patch into a fixed sized patch.
        If the size is already correct, then the input image
        patch is returned with no resizing.
        """

        fixed_shape = (self._fixed_patch_length, self._fixed_patch_length)
        if img_patch.shape == fixed_shape:
            return img_patch
        return cv2.resize(img_patch, fixed_shape, interpolation=cv2.INTER_AREA)
