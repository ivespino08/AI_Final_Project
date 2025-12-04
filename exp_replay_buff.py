from numpy import sum as npsum
from numpy.random import choice, sample

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        sample_batch = sample(self.buffer, batch_size)
        return sample_batch
    

    class PriorityReplayBuffer:
        def __init__(self, size):
            self.size = size
            self.buffer = []

        def add(self, state, action, reward, next_state, done, td_error):
            if len(self.buffer) >= self.size:
                self.buffer.pop(0)
            self.buffer.append([state, action, reward, next_state, done, td_error])

        def sample(self, batch_size):
            prios = [abs(exp[2]) for exp in self.buffer]
            probs = prios / npsum(prios)

            sample_indices = choice(range(len(self.buffer)), batch_size, p = probs)
            sample_batch = [self.buffer[i] for i in sample_indices]
            return sample_batch