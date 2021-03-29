import numpy as np


class Noise:
    def __init__(self, mean=0, std=0.2, clip=0.5):
        self.mean = mean
        self.std = std
        self.clip = clip

    def sample(self, shape):
        return np.random.normal(self.mean, self.std, *shape).clip(-self.clip, self.clip)

    def schedule(self, step):
        pass


class ScheduledNoise(Noise):
    def __init__(self, mean=0, std=0.1, clip=0.5, num_steps=int(1e5), alpha=0.5):
        super().__init__(mean, std, clip)
        self.num_steps = num_steps
        self.alpha = alpha

    def schedule(self, step):
        d = 1 + self.num_steps - step
        self.std -= self.alpha * self.std / d
        self.clip -= self.alpha * self.clip / d

