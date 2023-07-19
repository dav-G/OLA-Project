import numpy as np
import math
from CUSUM import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
import copy

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
	def __init__(self, n_arms, arms,margin,clicks,cost):
		super().__init__(n_arms, arms)
		self.beta_parameters = np.ones((n_arms, 2))
		self.margin=margin
		self.clicks=clicks
		self.cost=cost

	def pull_arm(self):
		idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])*self.margin[:]-self.clicks*self.cost)
		return idx

	def update(self, pulled_arm, reward):
		self.t+=1
		self.update_observations(pulled_arm, self.margin[pulled_arm]*reward-self.clicks*self.cost)
		self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
		self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + self.clicks - reward

class UCB1_Learner(Learner):
	def __init__(self, n_arms, arms,margin,clicks,cost):
		super().__init__(n_arms, arms)
		self.pulled_arms_counts = [0] * n_arms
		self.ucb_values = [0] * n_arms
		self.margin=margin
		self.clicks=clicks
		self.cost=cost

	def pull_arm(self):
		if self.t < self.n_arms:
			return self.t

		for arm in range(self.n_arms):
			average_reward = np.mean(self.rewards_per_arm[arm])
			exploration_term = math.sqrt(2 * math.log(self.t) / self.pulled_arms_counts[arm])
			self.ucb_values[arm] = average_reward + exploration_term

		return np.argmax(self.ucb_values)

	def update(self, pulled_arm, reward):
		super().update_observations(pulled_arm, self.margin[pulled_arm]*reward-self.clicks*self.cost)
		self.pulled_arms_counts[pulled_arm] += 1
		self.t += 1

class UCB1_Learner_ns(Learner):
    def __init__(self, n_arms, arms, margin, clicks, cost):
        super().__init__(n_arms, arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        self.margin = margin
        self.clicks = clicks
        self.cost = cost

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf==upper_conf.max())[0]) 

    def update(self, pulled_arm, reward):
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / (self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        self.t+=1
        self.empirical_means[pulled_arm]= (self.empirical_means[pulled_arm]*(self.t-1) + normalized_reward)/self.t
        for a in range(self.n_arms):
            n_samples= len(self.rewards_per_arm[a])
            self.confidence[a]= np.sqrt(2*np.log(self.t)/n_samples) if n_samples>0 else np.inf
        super().update_observations(pulled_arm, reward)


		
class TS_Context_Learner(Learner):
	def __init__(self, n_arms, arms,margin):
		super().__init__(n_arms, arms)
		self.beta_parameters = np.ones((1, n_arms, 2))
		self.margin=margin
		
		self.pulled_arms_idx=[]
		self.pulled_features=[]
		self.contexts=[{}]

	def pull_arm(self, features):
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				return np.argmax(np.random.beta(self.beta_parameters[i,:,0], self.beta_parameters[i,:,1])*self.margin[:])
		return None

	def update(self, pulled_arm, reward,sold,clicks, features):
		self.t+=1
		self.update_observations(pulled_arm, reward, features)
		
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				self.beta_parameters[i, pulled_arm, 0] = self.beta_parameters[i, pulled_arm, 0] + sold
				self.beta_parameters[i, pulled_arm, 1] = self.beta_parameters[i, pulled_arm, 1] + clicks - sold
		
	def update_observations(self,arm_idx, reward, features):
		super().update_observations(arm_idx,reward)
		self.pulled_arms_idx.append(arm_idx)
		self.pulled_features.append(features)

	def updateContexts(self, contexts):
		new_beta_parameters = np.ones((len(contexts), self.n_arms, 2))
		for i, new_context in enumerate(contexts):
			for j, old_context in enumerate(self.contexts):
				if old_context.items() <= new_context.items():
					new_beta_parameters[i] = copy.deepcopy(self.beta_parameters[j])
					
		self.contexts = contexts
		self.beta_parameters = new_beta_parameters
		
		
class UCB1_Context_Learner(Learner):
	def __init__(self, n_arms, arms):
		super().__init__(n_arms, arms)
		
		self.rewards_per_arm = np.empty((1, n_arms), dtype='O')
		for i in range(n_arms):
			self.rewards_per_arm[0, i] = []
		
		self.pulled_arms_counts = np.zeros((1, n_arms))
		self.ucb_values = [0] * n_arms
		
		self.pulled_arms_idx=[]
		self.pulled_features=[]
		self.contexts=[{}]

	def pull_arm(self, features):
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				
				for arm in range(self.n_arms):
					if len(self.rewards_per_arm[i, arm]) == 0:
						return arm
						
					average_reward = np.mean(self.rewards_per_arm[i, arm])
					exploration_term = math.sqrt(2 * math.log(max([1, self.t / len(self.contexts)])) / self.pulled_arms_counts[i, arm])
					self.ucb_values[arm] = average_reward + exploration_term

				return np.argmax(self.ucb_values)
		return None

	def update(self, pulled_arm, reward, features):
		self.update_observations(pulled_arm, reward, features)
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				self.rewards_per_arm[i, pulled_arm].append(reward)
				self.pulled_arms_counts[i, pulled_arm] += 1
		self.t += 1
	
	def update_observations(self, arm_idx, reward, features):
		self.collected_rewards = np.append(self.collected_rewards, reward)
		self.pulled_arms.append(self.arms[arm_idx])
		self.pulled_arms_idx.append(arm_idx)
		self.pulled_features.append(features)

	def updateContexts(self, contexts):
		new_rewards_per_arm = np.empty((len(contexts), self.n_arms), dtype='O')
		for j in range(self.n_arms):
			for i in range(len(contexts)):
				new_rewards_per_arm[i,j] = []
		new_pulled_arms_counts = np.zeros((len(contexts), self.n_arms))
		
		for i, new_context in enumerate(contexts):
			for j, old_context in enumerate(self.contexts):
				if old_context.items() <= new_context.items():
					new_rewards_per_arm[i] =  copy.deepcopy(self.rewards_per_arm[j])
					new_pulled_arms_counts[i] = copy.deepcopy(self.pulled_arms_counts[j])
					
		self.rewards_per_arm = new_rewards_per_arm
		self.pulled_arms_counts = new_pulled_arms_counts
		self.contexts = contexts
		
		# self.contexts = contexts
		# self.rewards_per_arm = np.empty((0, n_arms, 0))
		# self.pulled_arms_counts = np.zeros((0, n_arms))

		# for context in contexts:
			# self.rewards_per_arm = np.append(self.rewards_per_arm, np.empty(n_arms, 0))
			# self.pulled_arms_counts = np.append(self.pulled_arms_counts, np.zeros(n_arms))

			# for i,feature in enumerate(self.pulled_features):
				# if context.items() <= feature.items():
					# self.rewards_per_arm[-1, pulled_arms_idx[i]] = np.append(self.rewards_per_arm[-1, pulled_arms_idx[i]], self.collected_rewards[i])
					# self.pulled_arms_counts[i, pulled_arm] += 1
		
		
class TS_Learner3(Learner):
	def __init__(self, n_arms, arms,margin):
		super().__init__(n_arms, arms)
		self.beta_parameters = np.ones((n_arms, 2))
		self.margin=margin

	def pull_arm(self):
		idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])*self.margin[:])
		return idx

	def update(self, pulled_arm, reward,sold,clicks):
		self.t+=1
		self.update_observations(pulled_arm, reward)
		self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + sold
		self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + clicks - sold

class UCB1_Learner3(Learner):
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
		super().__init__(n_arms,arms)
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


	def plot(self, unknown_function, sigma_scale_factor=20, path=None):

		x_pred = np.atleast_2d(self.arms).T
		y_pred, sigma = self.gp.predict(x_pred, return_std=True)

		plt.figure(0)
		plt.title(f'Predicted clicks over budget')
		plt.plot(x_pred, unknown_function, ':', label=r'True n(x) function')
		plt.scatter(self.pulled_arms, self.collected_rewards, marker='o', label=r'Observed Clicks')
		plt.plot(x_pred, y_pred, '-', label=r'Predicted Clicks')
		plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_pred - 1.96 * sigma * sigma_scale_factor,
                                 (y_pred + 1.96 * sigma * sigma_scale_factor)[::-1]]),
                 alpha=.2, fc='C2', ec='None', label='95% conf interval')
		plt.xlabel('% Of Allocated Budget')
		plt.ylabel('$n(x)$')
		plt.legend(loc='lower right')
		plt.show()	

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
		self.contexts=[{}]
		self.contexted_pulled_arms=[[]]
		self.contexted_collected_rewards=[[]]

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
		for i,context in enumerate(self.contexts):
			if context.items() <= features.items():
				x=np.atleast_2d(self.contexted_pulled_arms[i]).T
				y=self.contexted_collected_rewards[i]
				break
		
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

class SWUCB_Learner(UCB1_Learner_ns):
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
            if not(arm in self.last_choices):
                 return arm
            self.empirical_means[arm] = np.sum(
                self.last_rewards[self.last_choices == arm]
            ) / self.arms[arm]
            self.confidence = np.sqrt(
                (2 * math.log(min(self.t, self.window_size))) / self.arms[arm]
            )
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf==upper_conf.max())[0])

    def update(self, pulled_arm, reward):        
        reward = self.margin[pulled_arm] * reward - self.clicks * self.cost
        normalized_reward = reward / (self.margin[pulled_arm] * self.clicks - self.clicks * self.cost)
        now = self.t % self.window_size
        self.last_choices[now] = pulled_arm
        self.arms[pulled_arm] = np.count_nonzero(self.last_choices == pulled_arm)
        self.last_rewards[now] = normalized_reward
        self.t += 1
        super().update_observations(pulled_arm, reward)

class CDUCB_Learner(UCB1_Learner_ns):
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

class EXP3_Learner(UCB1_Learner_ns):
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