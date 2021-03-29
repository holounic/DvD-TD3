import random
from collections import deque


class Buffer:
    def __init__(self, maxlen=int(1e5)):
        self.data = deque(maxlen=maxlen)

    def save_transition(self, transition):
        self.data.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(zip(*batch))
