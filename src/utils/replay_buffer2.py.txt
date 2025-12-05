import numpy as np
import random


# BASIC REPLAY BUFFER


class ReplayBuffer:
    def __init__(self, size):
        # Maximum number of stored experiences
        self.size = size
        
        # Internal list used to store experience tuples
        # Each entry will be: (state, action, reward, next_state, done)
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        # If the buffer is full, remove the oldest stored experience (FIFO behavior)
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)

        # Add the new experience tuple to the end of the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly pick 'batch_size' number of experiences from the buffer
        # random.sample chooses WITHOUT replacement
        return random.sample(self.buffer, batch_size)


# PRIORITY REPLAY BUFFER 
class PriorityReplayBuffer:
    def __init__(self, size):
        # Maximum number of stored experiences
        self.size = size
        self.buffer = []

    def add(self, state, action, reward, next_state, done, td_error):
        # If full, remove oldest (same FIFO behavior)
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)

        # Store td_error as the last element (used as priority)
        self.buffer.append((state, action, reward, next_state, done, td_error))

    def sample(self, batch_size):
        # Extract all td_error values from stored experiences
        # td_error = how "surprising" or important the transition was
        td_errors = np.array([abs(exp[5]) for exp in self.buffer])

        # If all td_errors are zero → assign equal probability
        if td_errors.sum() == 0:
            probs = np.ones(len(td_errors)) / len(td_errors)
        else:
            # Convert td_errors into probabilities by normalizing them
            probs = td_errors / td_errors.sum()

        # Select indices according to priority probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Return the sampled experiences
        return [self.buffer[i] for i in indices]
