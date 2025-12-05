import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer, PriorityReplayBuffer


def train_dqn(
    num_episodes=500,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    use_reward_shaping=False,
    use_priority=False
):
    # Create environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    agent = DQNAgent(state_dim, action_dim, buffer_size=buffer_size, gamma=gamma)

    # Initialize correct replay buffer
    if use_priority:
        replay_buffer = PriorityReplayBuffer(buffer_size)
    else:
        replay_buffer = ReplayBuffer(buffer_size)

    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Reward shaping
            if use_reward_shaping and done:
                reward = reward - 5

            # Store experience
            if use_priority:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    current_q = agent.q_network(state_tensor)[0][action]
                    next_q = agent.target_network(next_state_tensor).max() * (1 - done)
                    td_error = (reward + agent.gamma * next_q) - current_q

                replay_buffer.add(state, action, reward, next_state, done, float(td_error))
            else:
                replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

            # Training step
            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)

                if use_priority:
                    batch = [(s, a, r, ns, d) for (s, a, r, ns, d, _) in batch]

                agent.train(batch)

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

    # Summary Statistics
    avg_last50 = np.mean(rewards_history[-50:])
    best_reward = np.max(rewards_history)
    print(f"\nBest Reward: {best_reward:.2f}")
    print(f"Average Last 50 Episodes: {avg_last50:.2f}\n")

    # Save raw data only (no plotting)
    np.save("rewards_history.npy", np.array(rewards_history))

    return rewards_history


if __name__ == "__main__":
    train_dqn(
        use_reward_shaping=False,
        use_priority=True
    )
