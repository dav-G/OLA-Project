from UCB1_Learner import *
import numpy as np
from CUSUM import *


class CDUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, arms, M, eps, h, alpha, margin, clicks, cost):
        super().__init__(n_arms, arms, margin, clicks, cost)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha
    
    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e-3
            return np.random.choice(np.where(upper_conf==upper_conf.max())[0])
        else:
            return np.random.choice(self.n_arms)
    
    def update(self, pulled_arm, reward):
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / (self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        self.t += 1
        if self.change_detection[pulled_arm].update(normalized_reward):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arm[pulled_arm]=[]
            self.change_detection[pulled_arm].reset()
        self.update_observations(pulled_arm, normalized_reward, reward)
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arm[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arm])
        for a in range(self.n_arms):
           n_samples = len(self.valid_rewards_per_arm[a])
           self.confidence[a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples>0 else np.inf

    def update_observations(self, pulled_arm, normalized_reward, reward):
       self.rewards_per_arm[pulled_arm].append(reward)
       self.valid_rewards_per_arm[pulled_arm].append(normalized_reward)
       self.collected_rewards = np.append(self.collected_rewards, reward)