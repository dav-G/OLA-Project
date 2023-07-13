import numpy as np
from Learner import *
from math import sqrt, log

class EXP3_Learner(Learner):
    def __init__(self, n_arms, gamma):
        super().__init__(n_arms)
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.estimated_rewards = np.zeros(n_arms)
        self.p = np.full(n_arms,1/n_arms)

    def pull_arm(self):
        self.p = (1-self.gamma)*self.weights/(np.sum(self.weights)) + self.gamma/self.n_arms
        return np.random.choice(np.where(self.p==self.p.max())[0])

    def update(self, pulled_arm, reward):
        self.estimated_rewards[pulled_arm] = reward/self.p[pulled_arm]
        self.weights[pulled_arm] = self.weights[pulled_arm]*np.exp(
            self.estimated_rewards[pulled_arm]*self.gamma/self.n_arms
        )
        self.t += 1
        super().update_observations(pulled_arm, reward)