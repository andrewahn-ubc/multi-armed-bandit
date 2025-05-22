import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import MultiArmedBandit
from agent_sample_avg import RLAgentSampleAvg
from agent_constant_step import RLAgentConstantStepSize

    
if __name__ == "__main__":
    num_runs = 1000
    time_steps = 10000
    all_rewards_sample_avg = np.empty((0, time_steps))
    all_rewards_constant_step = np.empty((0, time_steps))
    stationary = False

    for j in range(num_runs):
        bandit = MultiArmedBandit(num_arms=10, mean_start=0, sd=1, walk_mean=0, walk_sd=0.01, stationary=stationary)

        # Initialize the two RL agents
        agent_sample_avg = RLAgentSampleAvg(bandit, start_estimate=0, epsilon=0.1)
        agent_constant_step = RLAgentConstantStepSize(bandit, start_estimate=0, epsilon=0.1, stepSize=0.1)

        # Iterate through 10,000 time steps, performing an action for each RL agent and incrementing the bandit 
        rewards_sample_avg = np.array([])
        rewards_constant_step = np.array([])

        for i in range(time_steps):
            reward_sa = agent_sample_avg.action()
            reward_cs = agent_constant_step.action()

            rewards_sample_avg = np.append(rewards_sample_avg, reward_sa)
            rewards_constant_step = np.append(rewards_constant_step, reward_cs)
            
            if not stationary:
                bandit.increment()
        
        all_rewards_sample_avg = np.vstack([all_rewards_sample_avg, rewards_sample_avg])
        all_rewards_constant_step = np.vstack([all_rewards_constant_step, rewards_constant_step])

    # Visualize the average reward against the time steps, across all runs for both methods
    mean_rewards_sample_avg = np.mean(all_rewards_sample_avg, axis=0)
    mean_rewards_constant_step = np.mean(all_rewards_constant_step, axis=0)
    x = np.arange(1, time_steps + 1)
    plt.plot(x, mean_rewards_sample_avg, label="Sample-Average Method", color="blue")
    plt.plot(x, mean_rewards_constant_step, label="Constant Step Size Method", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Comparison of Action-Value Methods for Non-Stationary Problems")
    plt.legend()
    plt.show()


