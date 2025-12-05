import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run_random_agent(num_episodes=500):
    env = gym.make("CartPole-v1")
    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards_history.append(total_reward)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            last10 = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Last 10 Rewards: {last10:.2f}")

    env.close()

    print(f"\nBest Reward: {np.max(rewards_history):.2f}")
    print(f"Average Reward over All Episodes: {np.mean(rewards_history):.2f}")

    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Random Agent - CartPole")
    plt.show()

    return rewards_history

if __name__ == "__main__":
    run_random_agent()
