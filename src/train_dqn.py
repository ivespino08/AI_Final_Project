import gymnasium as gym
import numpy as np
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

def train_dqn(
    num_episodes=500,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    use_reward_shaping=False
):
    #Create environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent and replay buffer
    agent = DQNAgent(state_dim, action_dim, buffer_size=buffer_size, gamma=gamma)
    replay_buffer = ReplayBuffer(buffer_size)

    rewards_history = []  # for plotting

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            #Select action with epsilon-greedy
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            ## Variation #2: Reward Shaping (turn ON only when testing variation)
            if use_reward_shaping and done:
                reward = reward - 5

            #Store experience
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

            #TRAIN STEP IF ENOUGH MEMORY
            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.train(batch)

        #End of episode
        agent.decay_epsilon()
        rewards_history.append(total_reward)

        # Logging
        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

    # Summary Statistics (for report)
    avg_last50 = np.mean(rewards_history[-50:])
    best_reward = np.max(rewards_history)
    print(f"\nBest Reward: {best_reward:.2f}")
    print(f"Average Last 50 Episodes: {avg_last50:.2f}\n")

    # Save results to files
    np.save("rewards_history.npy", np.array(rewards_history))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Learning Curve - CartPole")
    plt.savefig("dqn_learning_curve.png")
    plt.show()

    return rewards_history


if __name__ == "__main__":
    # To toggle reward shaping:
    #  True  = penalty for falling
    #  False = standard
    train_dqn(use_reward_shaping= True)
