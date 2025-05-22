import numpy as np

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