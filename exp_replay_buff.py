import numpy as np
from numpy import sum as npsum
from numpy.random import choice, sample

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append([state, action, reward, next_state])

    def sample(self, batch_size):
        sample_batch = sample(self.buffer, batch_size)
        return sample_batch
    

    class PriorityReplayBuffer:
        def __init__(self, size):
            self.size = size
            self.buffer = []

        def add(self, state, action, reward, next_state):
            if len(self.buffer) >= self.size:
                self.buffer.pop(0)
            self.buffer.append([state, action, reward, next_state])

        def sample(self, batch_size):
            priorities = [abs(experience[2]) for experience in self.buffer]

            probabilities = priorities / npsum(priorities)
            sample_indices = choice(range(len(self.buffer)), batch_size, p = probabilities)

            return [self.buffer[i] for i in sample_indices]