import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer
from agents.model import DQNModel


# DQN Agent Class
# This class:
#  - Chooses actions using epsilon-greedy
#  - Stores experiences into ReplayBuffer
#  - Learns Q-values by training the neural network
#  - Maintains a target network for stable learning

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size=50000,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=100
    ):
        # Main Q-network (the one we train)
        self.q_network = DQNModel(state_dim, action_dim)

        # Target network (more stable learning)
        self.target_network = DQNModel(state_dim, action_dim)

        # Copy weights from q_network -> target_network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Stores transitions for training
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Adam optimizer for neural network training
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Discount factor for Bellman equation
        self.gamma = gamma

        # Exploration variables for epsilon-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Environment information
        self.action_dim = action_dim

        # How often to update target network
        self.target_update_freq = target_update_freq

        # Count steps for syncing target network
        self.learn_step_counter = 0


    # Choosing an action (epsilon-greedy)
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        return action


    # Store experience in replay buffer
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


    # Train the Q-network
    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network every N learning steps
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
