from arm import Arm
import numpy as np

# Special case of the multi-armed bandit problem where the true action values start out at the same value and slowly scatter
class MultiArmedBandit:
    def __init__(self, num_arms, mean_start, sd, walk_mean, walk_sd, stationary=True):
        self.num_arms = num_arms
        self.mean_start = mean_start
        self.sd = sd
        self.walk_mean = walk_mean
        self.walk_sd = walk_sd
        self.stationary = stationary
        self.arms = []

        for i in range(num_arms):
            mean_i = self.mean_start
            sd_i = self.sd

            if (self.stationary):
                mean_i = np.random.normal(loc=0, scale=1)

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
