#%%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt
from Customer import Customer
from Environment import Environment
from Learners import GPUCB_Learner, GPTS_Learner,TS_Learner3,UCB1_Learner3

from plotResults import plot
from clairvoyant import getOptimal
# %%
n_arms_ad=10 #da cambiare bisogna mettere 100 utilizzare uno minure solo per vedere le convergenze
n_arms_pr=5

min_bid=0.0
max_bid=2.0
bids=np.linspace(min_bid,max_bid,n_arms_ad)
prices=np.linspace(10,50,n_arms_pr)
sigma=10
margin=(prices-8)
T=60 #horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment=3


customers = []
customers.append(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100))

# %%

gpucb_rewards_per_experiment=[]
gpts_rewards_per_experiment=[]

ucb1_rewards_per_experiment = []
ts_rewards_per_experiment = []
for e in range(0,n_experiment):
  beta=0
  env = Environment(prices, bids, sigma, customers)
  gpts_learner = GPTS_Learner(n_arms = n_arms_ad, arms = bids)
  gpucb_learner = GPUCB_Learner(n_arms = n_arms_ad, arms = bids)
  ucb1_learner = UCB1_Learner3(n_arms_pr,prices)
  ts_learner = TS_Learner3(n_arms_pr,prices,margin)

  for t in range (1,T+1):
    #GPTS Learner
    pulled_bid=gpts_learner.pull_arm()
    pulled_price = ts_learner.pull_arm()
    sold,clicks,clicks_cost, features=env.round(pulled_price,pulled_bid)
    reward=sold*margin[pulled_price]-clicks*clicks_cost
    gpts_learner.update(pulled_bid, reward)
    ts_learner.update(pulled_price, reward,sold,clicks)
    #gpucb Learner
    beta=2*np.log(n_arms_ad*t**2*np.pi**2/(6*0.05))
    pulled_bid=gpucb_learner.pull_arm(beta)
    pulled_price = ucb1_learner.pull_arm()
    sold,clicks,clicks_cost, features=env.round(pulled_price,pulled_bid)
    reward=sold*margin[pulled_price]-clicks*clicks_cost
    gpucb_learner.update(pulled_bid, reward)
    ucb1_learner.update(pulled_price, reward)

  gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
  gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)


# %%
#prendere opt dal clairovoiant
gpts_rewards_per_experiment=np.array(gpts_rewards_per_experiment)
gpucb_rewards_per_experiment=np.array(gpucb_rewards_per_experiment)

opt=1008.17

plot(opt, T, (gpucb_rewards_per_experiment, gpts_rewards_per_experiment), ('GPUCB', 'GPTS'))

# %%
