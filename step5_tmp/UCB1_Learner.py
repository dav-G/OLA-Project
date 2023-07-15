import math
from Learner import *

class UCB1_Learner(Learner):
    def __init__(self, n_arms,margin,clicks,cost):
        super().__init__(n_arms)
        #self.pulled_arms_counts = [0] * n_arms
        #self.ucb_values = [0] * n_arms
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        self.margin=margin
        self.clicks=clicks
        self.cost=cost

    def pull_arm(self):
        # if self.t < self.n_arms:
        #     return self.t
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf==upper_conf.max())[0]) 

        # for arm in range(self.n_arms):
        #     average_reward = np.mean(self.rewards_per_arm[arm]*self.margin[arm]-self.clicks*self.cost)
        #     exploration_term = math.sqrt(2 * math.log(self.t) / self.pulled_arms_counts[arm])
        #     self.ucb_values[arm] = average_reward + exploration_term

        # return np.argmax(self.ucb_values)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm]= (
            self.empirical_means[pulled_arm]*(self.t-1) + self.margin[pulled_arm]*reward-self.clicks*self.cost
        )/self.t
        for a in range(self.n_arms):
            n_samples= len(self.rewards_per_arm[a])
            self.confidence[a]= np.sqrt(2*np.log(self.t)/n_samples) if n_samples>0 else np.inf
        super().update_observations(pulled_arm, self.margin[pulled_arm]*reward-self.clicks*self.cost)
        