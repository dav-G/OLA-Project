import numpy as np
from UCB1_Learner import *

"""
class EXP3_Learner(UCB1_Learner):
    def __init__(self, n_arms, arms, margin, clicks, cost):
        super().__init__(n_arms, arms, margin, clicks, cost)
        self.gamma = 0.8
        self.weights = np.ones(n_arms)
        self.estimated_rewards = np.zeros(n_arms)
        self.p = np.full(n_arms,1/n_arms)

    def pull_arm(self):
        self.p = (1-self.gamma)*self.weights/(np.sum(self.weights)) + self.gamma/self.n_arms
        return np.random.choice(np.where(self.p==self.p.max())[0])

    def update(self, pulled_arm, reward):
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / (self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        self.estimated_rewards[pulled_arm] = normalized_reward/self.p[pulled_arm]
        self.weights[pulled_arm] = self.weights[pulled_arm]*np.exp(
            self.estimated_rewards[pulled_arm]*self.gamma/self.n_arms
        )
        self.t += 1
        super().update_observations(pulled_arm, reward)     
"""

class EXP3_Learner(UCB1_Learner):
    def __init__(self, n_arms, arms, margin, clicks, cost):
        super().__init__(n_arms, arms, margin, clicks, cost)
        self.gamma = np.sqrt(np.log(n_arms) / n_arms)
        self.weights = np.full(n_arms, 1. / n_arms)
        self._initial_exploration = np.random.permutation(n_arms)
    
    @property
    def trusts(self):
        trusts = ((1 - self.gamma) * self.weights) + (self.gamma / self.n_arms)
        if not np.all(np.isfinite(trusts)):
            trusts[~np.isfinite(trusts)] = 0 
        if np.isclose(np.sum(trusts), 0):
            trusts[:] = 1.0 / self.n_arms
        return trusts / np.sum(trusts)
    
    def pull_arm(self):
        if self.t < self.n_arms:
            return self._initial_exploration[self.t]
        else:
            return np.random.choice(self.n_arms, p=self.trusts)

    
    def update(self, pulled_arm, reward):
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / (self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        self.weights[pulled_arm] *= np.exp(normalized_reward * (self.gamma / self.n_arms))
        self.weights /= np.sum(self.weights)
        self.t += 1
        super().update_observations(pulled_arm, reward)