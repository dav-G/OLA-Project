from UCB1_Learner import *
import numpy as np
from math import log

class SWUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, arms, window_size, margin, clicks, cost):
        super().__init__(n_arms, arms, margin, clicks, cost)
        self.window_size = window_size
        self.last_rewards = np.zeros(window_size)
        self.last_choices = np.full(window_size, -1)
        self.arms = np.zeros(n_arms)
    
    def pull_arm(self):
        if np.any(self.arms == 0):
            return np.where(self.arms == 0)[0][0]
        
        for arm in range(self.n_arms):
            self.empirical_means[arm] = np.sum(
                self.last_rewards[self.last_choices == arm]
            ) / self.arms[arm]
            self.confidence = np.sqrt(
                (2 * log(min(self.t, self.window_size))) / self.arms[arm]
            )
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf==upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / np.max(self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        now = self.t % self.window_size
        self.last_choices[now] = pulled_arm
        self.arms[pulled_arm] = np.count_nonzero(self.last_choices == pulled_arm)
        self.last_rewards[now] = normalized_reward
        self.t += 1
        super().update_observations(pulled_arm, reward)