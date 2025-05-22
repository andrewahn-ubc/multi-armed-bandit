import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import MultiArmedBandit
from agent_sample_avg import RLAgentSampleAvg
from agent_constant_step import RLAgentConstantStepSize

    
if __name__ == "__main__":
    num_runs = 1000
    time_steps = 10000
    all_rewards_1 = np.empty((0, time_steps))
    all_rewards_2 = np.empty((0, time_steps))
    stationary = False
    method_1_name = "Sample-Average Method"
    method_2_name = "Recency-Weighted-Average Method"
    graph_title = "Comparison of Action-Value Methods for Non-Stationary Problems"

    for j in range(num_runs):
        bandit = MultiArmedBandit(num_arms=10, mean_start=0, sd=1, walk_mean=0, walk_sd=0.01, stationary=stationary)

        # Initialize the two RL agents
        agent_1 = RLAgentSampleAvg(bandit, start_estimate=0, epsilon=0.1)
        agent_2 = RLAgentConstantStepSize(bandit, start_estimate=0, epsilon=0.1, stepSize=0.1)

        # Iterate through 10,000 time steps, performing an action for each RL agent and incrementing the bandit 
        rewards_1 = np.array([])
        rewards_2 = np.array([])

        for i in range(time_steps):
            reward_1 = agent_1.action()
            reward_2 = agent_2.action()

            rewards_1 = np.append(rewards_1, reward_1)
            rewards_2 = np.append(rewards_2, reward_2)
            
            if not stationary:
                bandit.increment()
        
        all_rewards_1 = np.vstack([all_rewards_1, rewards_1])
        all_rewards_2 = np.vstack([all_rewards_2, rewards_2])

    # Visualize the average reward against the time steps, across all runs for both methods
    mean_rewards_1 = np.mean(all_rewards_1, axis=0)
    mean_rewards_2 = np.mean(all_rewards_2, axis=0)
    x = np.arange(1, time_steps + 1)
    plt.plot(x, mean_rewards_1, label=method_1_name, color="blue")
    plt.plot(x, mean_rewards_2, label=method_2_name, color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title(graph_title)
    plt.legend()
    plt.show()


