import numpy as np
from localizer.replay_buffer import ReplayBuffer


class LocalizerAgent:
    def __init__(self, config):
        self.__config = config["agent"]

        # Initialize hyperparameters
        self.__gamma = self.__config.get("gamma", 0.99)
        self.__batch_size = self.__config.get("batch_size", 64)
        self.__learning_rate = self.__config.get("learning_rate", 1e-4)
        self.__epsilon_start = self.__config.get("epsilon_start", 1.0)
        self.__epsilon_end = self.__config.get("epsilon_end", 0.1)
        self.__epsilon_decay = self.__config.get("epsilon_decay", 10000)

        # Initialize replay buffer
        capacity = self.__config.get("replay_buffer_size", 100000)
        self.__replay_buffer = ReplayBuffer(capacity)

        self.__global_step = 0
        self.__epsilon = self.__epsilon_start

    def select_action(self, state):

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        else:
            # TODO
            # Use Q-network to pick best action
            action = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])

        return action

    def store_experience(self, experience):
        self.__replay_buffer.store(experience)

    def update(self):
        """
        Updates policy networks using transitions from replay buffer
        """

        if len(self.replay_buffer) < self.__batch_size:
            return

        observations, actions, rewards, next_obesrvations, dones = (
            self.__replay_buffer.sample(self.__batch_size)
        )

        # TODO
        # Perform DQN-style update
        # 1) current Q = Q_network(observations)[range(batch_size), actions]
        # 2) target Q =
        #     rewards + gamma * max Q_network(next_observations) * (1 - dones)
        # 3) MSE loss between current Q and target Q
        # 4) optimizer step

        # Update epsilon
        self.__global_step += 1
        fraction = min(float(self.__global_step) / self.__epsilon_decay, 1.0)
        self.__epsilon = self.__epsilon_start + fraction * (
            self.__epsilon_end - self.__epsilon_start
        )
