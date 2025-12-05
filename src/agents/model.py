
import torch
import torch.nn as nn

# We define a class called DQNModel
# This represents our neural network model for the agent
# It inherits from nn.Module, which is the base class for all neural network models in PyTorch
class DQNModel(nn.Module):
    # The __init__ method = the constructor
    # state_dim: how many numbers are in the input state ( 4 for CartPole)
    # action_dim: how many possible actions there are (2 for CartPole: left or right)
    def __init__(self, state_dim, action_dim):

        # Call the parent class (nn.Module) constructor so PyTorch can set up this model 
        super().__init__()

        # Define the actual neural network structure and store it in self.net
        # nn.Sequential lets us chain layers one after another in a straight line
        self.net = nn.Sequential(
            # First layer: a fully connected (linear) layer.
            # It takes an input of size state_dim (4)
            # and outputs a vector of size 128.
            nn.Linear(state_dim, 128),

            # Apply  ReLU activation function
            # This introduces non-linearity so the network can learn complex patterns.
            nn.ReLU(),

            # Second hidden layer: takes 128 inputs and outputs 128 values.
            nn.Linear(128, 128),

            # Another ReLU activation
            nn.ReLU(),

            # Output layer: takes 128 inputs and outputs action_dim values.
            # Each output corresponds to a Q-value for one possible action.
            nn.Linear(128, action_dim)
        )

    # The forward method defines how data moves through the network
    # x is the input (for example, a batch of states from the environment).
    def forward(self, x):
        # Pass the input x through the self.net sequence of layers we defined above.
        # Whatever comes out is returned as the output (Q-values).
        return self.net(x)
