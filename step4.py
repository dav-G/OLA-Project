def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt

from Environment import ContextEnvironment
from Learners import GPTS_Context_Learner, GPUCB_Context_Learner
from Customer import Customer

from plotResults import plot
from clairvoyant import getOptimal


def generateContext(rewards, pulled_arms, pulled_arms_features, features):
	rewards = np.array(rewards)
	pulled_arms = np.array(pulled_arms)
	pulled_arms_features = np.array(pulled_arms_features)

	delta = 0.9
	for feature in features:
		indexes1 = np.array( [ i for i,pulled_features in enumerate(pulled_arms_features) if pulled_features[feature] ] )
		indexes2 = np.array( [ i for i,pulled_features in enumerate(pulled_arms_features) if not pulled_features[feature] ] )
		
		if len(indexes1) == 0 or len(indexes2) == 0:
			continue
		
		rewards1, arms1, features1 = rewards[indexes1], pulled_arms[indexes1], pulled_arms_features[indexes1]
		rewards2, arms2, features2 = rewards[indexes2], pulled_arms[indexes2], pulled_arms_features[indexes2]
		
		tmp = [0] * n_arms
		for i in range(len(rewards1)):
			tmp[arms1[i]] += rewards1[i]
		expected_reward1 = np.max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards1)))
		
		tmp = [0] * n_arms
		for i in range(len(rewards2)):
			tmp[arms2[i]] += rewards2[i]
		expected_reward2 = np.max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards2)))
		
		tmp = [0] * n_arms
		for i in range(len(rewards)):
			tmp[pulled_arms[i]] += rewards[i]
		expected_reward = np.max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards)))
		
		
		prob1 = len(rewards1)/len(rewards) - np.sqrt(-np.log(delta)/(2*len(rewards1)))
		prob2 = len(rewards2)/len(rewards) - np.sqrt(-np.log(delta)/(2*len(rewards2)))
		
		if prob1*expected_reward1 + prob2*expected_reward2 >= expected_reward:
			new_features = features.copy()
			new_features.remove(feature)
			if new_features != {}:
				context1 = generateContext(rewards1, arms1, features1, new_features)
				context2 = generateContext(rewards2, arms2, features2, new_features)
			for i in range(len(context1)):
				context1[i][feature] = True
			for i in range(len(context2)):
				context2[i][feature] = False
			return context1 + context2
		
	return [{}]
	

n_arms = 100
min_bid = 0.0
max_bid = 2.0
bids = np.linspace(min_bid,max_bid,n_arms)

prices = [10, 20, 30, 40, 50]

sigma = 5

T = 60 #horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment = 3

customers = []
customers.append(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100, {'student': True, 'commuter': True}))
customers.append(Customer('C2', -0.079, 0.89, 46, 3.4, -0.5, -1.5, 80, {'student': False, 'commuter': True}))
customers.append(Customer('C3', -0.082, 0.79, 35, 3.3, -5, 0.3, 65, {'student': False, 'commuter': False}))

gpucb_rewards_per_experiment=[]
gpts_rewards_per_experiment=[]

for e in range(0,n_experiment):
	beta = 0
	env = ContextEnvironment(prices, bids, sigma, customers)
	gpts_learner = GPTS_Context_Learner(n_arms=n_arms, arms=bids)
	gpucb_learner = GPUCB_Context_Learner(n_arms=n_arms, arms=bids)
	
	print({'student':True}.items() <= {'student':True, 'commuter':False}.items())
	
	# gpts_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])
	# gpucb_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])

	for t in range (1,T+1):
		features = env.getFeatures()
		
		#GPTS Learner
		pulled_arm = gpts_learner.pull_arm(features)
		reward = env.getReward(2, pulled_arm)
		gpts_learner.update(pulled_arm, reward, features)

		if t%14 == 0:
			contexts = generateContext(gpts_learner.collected_rewards, gpts_learner.pulled_arms_idx, gpts_learner.pulled_features, ['student', 'commuter'])
			gpts_learner.updateContexts(contexts)

		#gpucb Learner
		beta = 2 * np.log(n_arms * t**2 * np.pi**2 /(6 * 0.05))
		pulled_arm = gpucb_learner.pull_arm(beta, features)
		reward = env.getReward(2, pulled_arm)
		gpucb_learner.update(pulled_arm, reward, features)

		if t%14 == 0:
			contexts = generateContext(gpucb_learner.collected_rewards, gpucb_learner.pulled_arms_idx, gpucb_learner.pulled_features, ['student', 'commuter'])
			gpucb_learner.updateContexts(contexts)

	gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
	gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)

		
gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)

opt = getOptimal()[2][0]

plot(opt, T, [gpts_rewards_per_experiment, gpucb_rewards_per_experiment], ['GPTS', 'GPUCB'])