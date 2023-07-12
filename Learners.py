import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class Learner:
	def __init__(self, n_arms, arms):
		self.n_arms = n_arms
		self.arms = arms
		self.t = 0
		self.rewards_per_arm = x = [[] for i in range(n_arms)]
		self.collected_rewards = np.array([])
		self.pulled_arms=[]

	def update_observations(self, pulled_arm, reward):
		self.rewards_per_arm[pulled_arm].append(reward)
		self.collected_rewards = np.append(self.collected_rewards, reward)
		self.pulled_arms.append(self.arms[pulled_arm])
		

		
class TS_Learner(Learner):
	def __init__(self, n_arms, arms):
		super().__init__(n_arms, arms)
		self.beta_parameters = np.ones((n_arms, 2))

	def pull_arm(self):
		idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
		return idx

	def update(self, pulled_arm, reward):
		self.t+=1
		self.update_observations(pulled_arm, reward)
		self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
		self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

class UCB1_Learner(Learner):
	def __init__(self, n_arms, arms):
		super().__init__(n_arms, arms)
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



class GPLearner(Learner):
	def __init__(self,n_arms,arms):
		super().__init__(n_arms, arms)
		self.arms=arms
		self.means=np.zeros(self.n_arms)
		self.sigmas=np.ones(self.n_arms)*5
		alpha=1
		kernel=C(1.0,(1e-3,1e3))*RBF(1.0,(1e-3,1e3))
		self.gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

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

class GPTS_Learner(GPLearner):
	def pull_arm(self):
		return np.argmax(np.random.normal(self.means, self.sigmas))

class GPUCB_Learner(GPLearner):
	def pull_arm(self,beta):
		return np.argmax(self.means + self.sigmas * np.sqrt(beta))
		
		

		
class GP_Context_Learner(GPLearner):
	def __init__(self,n_arms, arms):
		super().__init__(n_arms, arms)

		self.pulled_arms_idx=[]
		self.pulled_features=[]
		self.contexts=[]
		self.contexted_pulled_arms=[]
		self.contexted_collected_rewards=[]

	def update_observations(self,arm_idx, reward, features):
		super().update_observations(arm_idx,reward)
		self.pulled_arms_idx.append(arm_idx)
		self.pulled_features.append(features)
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				self.contexted_collected_rewards[i].append(reward)
				self.contexted_pulled_arms[i].append(self.arms[arm_idx])
				break

	def update_model(self, features):
		if self.contexts != []:
			for i,context in enumerate(self.contexts):
				if context.items() <= features.items():
					x=np.atleast_2d(self.contexted_pulled_arms[i]).T
					y=self.contexted_collected_rewards[i]
					break
		else:
			x=np.atleast_2d(self.pulled_arms).T
			y=self.collected_rewards
		
		self.gp.fit(x,y)
		self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
		self.sigmas=np.maximum(self.sigmas,1e-2)

	def update(self,pulled_arm,reward, features):
		self.t+=1
		self.update_observations(pulled_arm,reward,features)

	def updateContexts(self, contexts):
		self.contexts = contexts
		self.contexted_pulled_arms=[]
		self.contexted_collected_rewards=[]

		for context in contexts:
			self.contexted_collected_rewards.append([])
			self.contexted_pulled_arms.append([])

			for i,feature in enumerate(self.pulled_features):
				if context.items() <= feature.items():
					self.contexted_collected_rewards[-1].append(self.collected_rewards[i])
					self.contexted_pulled_arms[-1].append(self.pulled_arms[i])
				
class GPTS_Context_Learner(GP_Context_Learner):
	def pull_arm(self, features):
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items() and len(self.contexted_collected_rewards[i]) > 0:
				self.update_model(features)
		return np.argmax(np.random.normal(self.means, self.sigmas))

class GPUCB_Context_Learner(GP_Context_Learner):
	def pull_arm(self, beta, features):
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items() and len(self.contexted_collected_rewards[i]) > 0:
				self.update_model(features)
		return np.argmax(self.means + self.sigmas * np.sqrt(beta))