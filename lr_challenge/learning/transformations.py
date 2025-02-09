import torch


class ObservationNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = None
        self.std = None
        self.epsilon = epsilon
        self.clipping_epsilon = 5

    def update(self, obs):
        # Online mean and variance calculation
        if self.mean is None:
            self.mean = obs.mean(0)
            self.std = obs.std(0) + self.epsilon
        else:
            self.mean = 0.99 * self.mean + 0.01 * obs.mean(0)
            self.std = 0.99 * self.std + 0.01 * (obs.std(0) + self.epsilon)

    def normalize(self, obs):
        normalized = (obs - self.mean) / self.std
        return torch.clamp(normalized, -self.clipping_epsilon, self.clipping_epsilon)


def normalize_tensors(tensors: torch.Tensor) -> torch.Tensor:
    return (tensors - tensors.mean()) / (tensors.std() + 1e-8)
