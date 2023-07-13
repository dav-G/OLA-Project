from UCB1_Learner import *
import numpy as np
 
class SWUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
    
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        tmp = np.where(upper_conf==upper_conf.max())[0]
        if tmp.size > 0:
            return np.random.choice(tmp, size=int(len(tmp)>0)).item() 
        else:
            return 0

    def update(self, pulled_arm, reward):
        self.t +=1
        self.empirical_means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm][-self.window_size:])
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:]==arm)
            self.confidence[arm]= np.sqrt((2 * np.log(self.t) / n_samples)) if n_samples>0 else np.inf            
        super().update_observations(pulled_arm, reward)