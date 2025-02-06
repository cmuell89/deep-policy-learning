


class NPG:
    def __init__(self, policy, value_network, optimizer, gamma, lam):
        self.policy = policy
        self.value_network = value_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam

    def train(self, env, num_episodes, max_steps, batch_size, seed=None):
        pass 