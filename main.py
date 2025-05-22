import numpy as np
import random
import matplotlib.pyplot as plt

# A single arm of a multi-armed bandit that can be "pulled", after which it must return a randomly-sampled reward
class Arm:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def pull(self):
        reward = np.random.normal(loc=self.mean, scale=self.sd)
        return reward
    
    # Increment the action value (mean) by some amount to make this problem non-stationary
    def increment(self, amount):
        self.mean += amount

# Special case of the multi-armed bandit problem where the true action values start out at the same value and slowly scatter
class MultiArmedBandit:
    def __init__(self, num_arms, mean_start, sd, walk_mean, walk_sd):
        self.num_arms = num_arms
        self.mean_start = mean_start
        self.sd = sd
        self.walk_mean = walk_mean
        self.walk_sd = walk_sd
        self.arms = []

        for i in range(num_arms):
            # mean_i = self.mean_start
            # sd_i = self.sd

            mean_i = np.random.normal(loc=0, scale=1)
            sd_i = self.sd

            arm = Arm(mean_i, sd_i)
            self.arms.append(arm)

    # Changes the true action values of the arms to make this problem non-stationary. Run after each time step
    def increment(self):
        amounts = np.random.normal(loc=self.walk_mean, scale=self.walk_sd, size=(self.num_arms,))

        for i in range(self.num_arms):
            self.arms[i].increment(amounts[i])

    # Pull one of the arms and return the reward 
    def pull(self, index):
        reward = self.arms[index].pull()
        return reward
    
class RLAgentSampleAvg:
    def __init__(self, bandit, start_estimate, epsilon):
        self.bandit = bandit
        self.estimates = [start_estimate] * self.bandit.num_arms
        self.pull_counts = [0] * self.bandit.num_arms
        self.epsilon = epsilon

    # Pull an arm and update the estimate for that arm
    def action(self):
        x = random.random() # returns a float between 0 and 1
        reward = 0 # initialize the reward
        greedy_arm_index = np.argmax(self.estimates)
        index_of_arm_pulled = 0

        if (x > self.epsilon):
            # Greedy 
            reward = self.bandit.pull(greedy_arm_index)
            index_of_arm_pulled = greedy_arm_index
        else:
            # Non-greedy (assumes no duplicate estimates)
            random_index = round(random.random() * (self.bandit.num_arms - 2))
            if (random_index >= greedy_arm_index):
                random_index += 1
            reward = self.bandit.pull(random_index)
            index_of_arm_pulled = random_index

        # Update the estimate for that arm using the sample-averages method
        self.pull_counts[index_of_arm_pulled] += 1
        n = self.pull_counts[index_of_arm_pulled]
        prev_reward = self.estimates[index_of_arm_pulled]
        self.estimates[index_of_arm_pulled] = prev_reward + (1/n)*(reward - prev_reward)

        return reward

class RLAgentConstantStepSize:
    def __init__(self, bandit, start_estimate, epsilon, stepSize):
        self.bandit = bandit
        self.estimates = [start_estimate] * self.bandit.num_arms
        self.epsilon = epsilon
        self.stepSize = stepSize
    
    # Pull an arm and update the estimate for that arm
    def action(self):
        x = random.random() # returns a float between 0 and 1
        reward = 0 # initialize the reward
        greedy_arm_index = np.argmax(self.estimates)
        index_of_arm_pulled = 0

        if (x > self.epsilon):
            # Greedy 
            reward = self.bandit.pull(greedy_arm_index)
            index_of_arm_pulled = greedy_arm_index
        else:
            # Non-greedy (assumes no duplicate estimates)
            random_index = round(random.random() * (self.bandit.num_arms - 2))
            if (random_index >= greedy_arm_index):
                random_index += 1
            reward = self.bandit.pull(random_index)
            index_of_arm_pulled = random_index

        # Update the estimate for that arm using the constant step size method
        prev_reward = self.estimates[index_of_arm_pulled]
        self.estimates[index_of_arm_pulled] = prev_reward + self.stepSize*(reward - prev_reward)

        return reward

    
if __name__ == "__main__":
    num_runs = 1000
    time_steps = 10000
    all_rewards_sample_avg = np.empty((0, time_steps))
    all_rewards_constant_step = np.empty((0, time_steps))

    for j in range(num_runs):
        bandit = MultiArmedBandit(num_arms=10, mean_start=0, sd=1, walk_mean=0, walk_sd=0.01)

        # Initialize the two RL agents
        agent_sample_avg = RLAgentSampleAvg(bandit, start_estimate=0, epsilon=0.1)
        agent_constant_step = RLAgentConstantStepSize(bandit, start_estimate=0, epsilon=0.1, stepSize=0.1)
        # agent_constant_step = RLAgentSampleAvg(bandit, start_estimate=0, epsilon=0.01)

        # Iterate through 10,000 time steps, performing an action for each RL agent and incrementing the bandit 
        rewards_sample_avg = np.array([])
        rewards_constant_step = np.array([])

        for i in range(time_steps):
            reward_sa = agent_sample_avg.action()
            reward_cs = agent_constant_step.action()

            rewards_sample_avg = np.append(rewards_sample_avg, reward_sa)
            rewards_constant_step = np.append(rewards_constant_step, reward_cs)

            # bandit.increment()
        
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
    plt.title("Comparison of Action-Value Methods for Stationary Problems")
    # plt.plot(x, mean_rewards_sample_avg, label="0.1-greedy", color="blue")
    # plt.plot(x, mean_rewards_constant_step, label="0.01-greedy", color="red")
    # plt.xlabel("Time Step")
    # plt.ylabel("Reward")
    # plt.title("Comparison of Epsilon-Greedy Methods for Stationary Problems")

    plt.legend()
    plt.show()


