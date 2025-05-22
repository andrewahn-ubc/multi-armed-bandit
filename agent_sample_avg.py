import numpy as np
import random

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