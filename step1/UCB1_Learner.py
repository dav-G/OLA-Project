import math
from Learner import *

class UCB1_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.pulled_arms_counts = [0] * n_arms
        self.ucb_values = [0] * n_arms

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t

        for arm in range(self.n_arms):
            average_reward = np.mean(self.rewards_per_arm[arm])
            exploration_term = math.sqrt(2 * math.log(self.t) / self.pulled_arms_counts[arm])
            self.ucb_values[arm] = average_reward + exploration_term

        return np.argmax(self.ucb_values)

    def update(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms_counts[pulled_arm] += 1
        self.t += 1