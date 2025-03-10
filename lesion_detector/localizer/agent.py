import numpy as np
from localizer.replay_buffer import ReplayBuffer


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

        # Initialize replay buffer
        capacity = self._config.get("replay_buffer_size", 100000)
        self._replay_buffer = ReplayBuffer(capacity)

        self._global_step = 0
        self._epsilon = self._epsilon_start

    def select_action(self, state):

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        else:
            # TODO
            # Use Q-network to pick best action
            action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])

        return action

    def store_experience(self, experience):
        self._replay_buffer.store(experience)

    def update(self):
        """
        Updates policy networks using transitions from replay buffer
        """

        if len(self.replay_buffer) < self._batch_size:
            return

        observations, actions, rewards, next_obesrvations, dones = (
            self._replay_buffer.sample(self._batch_size)
        )

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
