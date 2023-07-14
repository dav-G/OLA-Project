from matplotlib import pyplot as plt
import numpy as np

from Environment import PricingEnvironment
from Learners import UCB1_Learner, TS_Learner
from Customer import Customer

from plotResults import plot
from clairvoyant import getOptimal

c1=(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100))

prices = [10, 20, 30, 40, 50]
prices=np.array(prices)
T = 365
n_experiments = 10

#prb=c1.conversion_probability(np.array(prices))
prb=np.array([0.95,0.75,0.53,0.28,0.14])
margin=(prices-8)
clicks=int(np.array(c1.num_clicks(2)))
cost=np.array(c1.click_cost(2))

ucb1_rewards_per_experiment = []
ts_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = PricingEnvironment(prices, [c1], prb)
    ucb1_learner = UCB1_Learner(len(prices), prices,margin,clicks,cost)
    ts_learner = TS_Learner(len(prices), prices,margin,clicks,cost)
	
    for t in range(0,T):
        # UCB1
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm,clicks)[0]
        ucb1_learner.update(pulled_arm, reward)

        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm,clicks)[0]
        ts_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)

ts_rewards_per_experiment = np.array(ts_rewards_per_experiment)
ucb1_rewards_per_experiment = np.array(ucb1_rewards_per_experiment)

opt=1008.17

plot(opt, T, [ucb1_rewards_per_experiment, ts_rewards_per_experiment], ['UCB1', 'TS'])