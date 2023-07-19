def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt

from Environment import ContextEnvironment
from Learners import GPTS_Context_Learner, GPUCB_Context_Learner, TS_Context_Learner, UCB1_Context_Learner
from Customer import Customer

from plotResults import plot
from clairvoyant import getOptimal


def generateContext(n_arms, rewards, pulled_arms, pulled_arms_features, features):
	rewards = np.array(rewards)
	pulled_arms = np.array(pulled_arms)
	pulled_arms_features = np.array(pulled_arms_features)

	delta = 1
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
		for i in range(n_arms):
			tmp[i] = tmp[i] / (sum(arms1 == i) + 1)
		expected_reward1 = max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards1) + 1))
		
		tmp = [0] * n_arms
		for i in range(len(rewards2)):
			tmp[arms2[i]] += rewards2[i]
		for i in range(n_arms):
			tmp[i] = tmp[i] / (sum(arms2 == i) + 1)
		expected_reward2 = max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards2) + 1))
		
		tmp = [0] * n_arms
		for i in range(len(rewards)):
			tmp[pulled_arms[i]] += rewards[i]
		for i in range(n_arms):
			tmp[i] = tmp[i] / (sum(pulled_arms == i) + 1)
		expected_reward = max(tmp) - np.sqrt(-np.log(delta)/(2*len(rewards) + 1))
		
		prob1 = len(rewards1)/len(rewards) - np.sqrt(-np.log(delta)/(2*len(rewards1) + 1))
		prob2 = len(rewards2)/len(rewards) - np.sqrt(-np.log(delta)/(2*len(rewards2) + 1))
		
		print(f'prob1: {prob1}, prob2: {prob2}, exp1: {expected_reward1}, exp2: {expected_reward2}, exp: {expected_reward}')
		if prob1*expected_reward1 + prob2*expected_reward2 >= expected_reward:
			print('SPLIT')
			new_features = features.copy()
			new_features.remove(feature)
			context1 = [{}]
			context2 = [{}]
			if new_features != []:
				context1 = generateContext(n_arms, rewards1, arms1, features1, new_features)
				context2 = generateContext(n_arms, rewards2, arms2, features2, new_features)
			for i in range(len(context1)):
				context1[i][feature] = True
			for i in range(len(context2)):
				context2[i][feature] = False
			return context1 + context2
		
	return [{}]
	

n_arms_ad = 100
n_arms_pr = 5
min_bid = 0.0
max_bid = 2.0
bids = np.linspace(min_bid,max_bid,n_arms_ad)

prices = np.array([10, 20, 30, 40, 50])
margin=(prices-8)

sigma = 5

T = 365 #horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment = 10

customers = []
customers.append(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100, {'student': True, 'commuter': True}))
customers.append(Customer('C2', -0.079, 0.89, 46, 3.4, -0.5, -1.5, 80, {'student': False, 'commuter': True}))
customers.append(Customer('C3', -0.082, 0.79, 35, 3.3, -5, 0.3, 65, {'student': False, 'commuter': False}))

gpucb_rewards_per_experiment=[]
gpts_rewards_per_experiment=[]

for e in range(0,n_experiment):
	print(f'Experiment {e}')

	env = ContextEnvironment(prices, bids, sigma, customers)
	gpts_learner = GPTS_Context_Learner(n_arms = n_arms_ad, arms = bids)
	gpucb_learner = GPUCB_Context_Learner(n_arms = n_arms_ad, arms = bids)
	ucb1_learner = UCB1_Context_Learner(n_arms_pr,prices)
	ts_learner = TS_Context_Learner(n_arms_pr,prices,margin)

	# gpts_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])
	# gpucb_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])
	# ts_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])
	# ucb1_learner.updateContexts([{'student': True}, {'student': False, 'commuter': True}, {'student': False, 'commuter': False}])
	
	for t in range (1,T+1):
		features = env.getFeatures()
		
		#GPTS Learner
		pulled_bid = gpts_learner.pull_arm(features)
		pulled_price = ts_learner.pull_arm(features)
		sold, clicks, clicks_cost, features = env.getReward(pulled_price, pulled_bid)
		reward = sold * margin[pulled_price] - clicks * clicks_cost
		gpts_learner.update(pulled_bid, reward, features)
		ts_learner.update(pulled_price, reward, sold, clicks, features)
		
		#gpucb Learner
		beta = 2 * np.log(n_arms_ad * t**2 * np.pi**2 / (6 * 0.05))
		pulled_bid = gpucb_learner.pull_arm(beta, features)
		pulled_price = ucb1_learner.pull_arm(features)
		sold, clicks, clicks_cost, features = env.getReward(pulled_price, pulled_bid)
		reward = sold * margin[pulled_price] - clicks * clicks_cost
		gpucb_learner.update(pulled_bid, reward, features)
		ucb1_learner.update(pulled_price, reward, features)
		
		if t%14 == 0:
			contexts = generateContext(n_arms_pr, ucb1_learner.collected_rewards, ucb1_learner.pulled_arms_idx, ucb1_learner.pulled_features, ['student', 'commuter'])
			ucb1_learner.updateContexts(contexts)
			print(f'1: Context at time {t} for experiment {e}: {contexts}')
			contexts = generateContext(n_arms_pr, ts_learner.collected_rewards, ts_learner.pulled_arms_idx, ts_learner.pulled_features, ['student', 'commuter'])
			ts_learner.updateContexts(contexts)
			print(f'2: Context at time {t} for experiment {e}: {contexts}')
			
			contexts = generateContext(n_arms_ad,gpucb_learner.collected_rewards, gpucb_learner.pulled_arms_idx, gpucb_learner.pulled_features, ['student', 'commuter'])
			gpucb_learner.updateContexts(contexts)
			print(f'3: Context at time {t} for experiment {e}: {contexts}')
			contexts = generateContext(n_arms_ad, gpts_learner.collected_rewards, gpts_learner.pulled_arms_idx, gpts_learner.pulled_features, ['student', 'commuter'])
			gpts_learner.updateContexts(contexts)
			print(f'4: Context at time {t} for experiment {e}: {contexts}')

	gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
	gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)

		
gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment)
gpucb_rewards_per_experiment = np.array(gpucb_rewards_per_experiment)

opt = np.average(getOptimal()[2])

plot(opt, T, [gpucb_rewards_per_experiment, gpts_rewards_per_experiment], ['GPUCB', 'GPTS'])