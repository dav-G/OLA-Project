from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner):
    def __init__(self,n_arms,arms):
        super().__init__(n_arms)
        self.arms=arms
        self.means=np.zeros(self.n_arms)
        self.sigmas=np.ones(self.n_arms)*15
        self.pulled_arms=[]
		self.pulled_arms_idx=[]
		self.pulled_features=[]
		self.contexts=[]
        alpha=1
        kernel=C(1.0,(1e-3,1e3))*RBF(1.0,(1e-3,1e3))
        self.gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self,arm_idx, reward, features):
        super().update_observations(arm_idx,reward)
        self.pulled_arms.append(self.arms[arm_idx])
		self.pulled_arms_idx.append(arm_idx)
		self.pulled_features.append(features)
		for i,context in enumerate(self.contexts):
			if context.items() < features.items():
				self.contexted_collected_rewards[i].append(reward)
				self.contexted_pulled_arms[i].append(self.arms[arm_idx])
				break

    def update_model(self, features):
		if self.contexts != []:
			for i,context in enumerate(self.contexts):
				if context.items() < features.items():
					x=np.atleast_2d(self.contexted_pulled_arms[i]).T
					y=self.contexted_collected_rewards[i]
					break
		else
			x=np.atleast_2d(self.pulled_arms).T
			y=self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas=np.maximum(self.sigmas,1e-2)

    def update(self,pulled_arm,reward, features):
        self.t+=1
        self.update_observations(pulled_arm,reward,features)
        self.update_model(features)

    def pull_arm(self):
        sampled_values=np.random.normal(self.means,self.sigmas)
        return np.argmax(sampled_values)
		
	def updateContexts(self, contexts):
		self.contexts = contexts
		self.contexted_pulled_arms=[]
		self.contexted_collected_rewards=[]
		
		for context in contexts:
			self.contexted_collected_rewards.append([])
			self.contexted_pulled_arms.append([])
			
			for i,feature in enumerate(self.pulled_features):
				if context.items() < feature.items():
					self.contexted_collected_rewards[-1].append(self.collected_rewards[i])
					self.contexted_pulled_arms[-1].append(self.pulled_arms[i])
    
    
class GPTS_Learner_Lo(Learner):
    def __init__(self,n_arms,arms):
        super().__init__(n_arms)
        self.arms=arms
        self.means=np.zeros(self.n_arms)
        self.sigmas=np.ones(self.n_arms)*0.5
        self.pulled_arms=[]
        alpha=1
        kernel=C(1.0,(1e-3,1e3))*RBF(1.0,(1e-3,1e3))
        self.gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self,arm_idx, reward):
        super().update_observations(arm_idx,reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x=np.atleast_2d(self.pulled_arms).T
        y=self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas=np.maximum(self.sigmas,1e-2)

    def update(self,pulled_arm,reward):
        self.t+=1
        self.update_observations(pulled_arm,reward)
        self.update_model()

    def pull_arm(self):
        sampled_values=np.random.normal(self.means,self.sigmas)
        return np.argmin(sampled_values)
