def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt

from Customer import Customer
from Environment import BiddingEnvironment
from Learners import GPUCB_Learner, GPTS_Learner

from plotResults import plot
from clairvoyant import getOptimal


n_arms = 100 
min_bid = 0.0
max_bid = 2.0
bids = np.linspace(min_bid,max_bid,n_arms)
sigma = 10
best_price = getOptimal()[0][0]

T = 60 # horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment = 5

customers = []
customers.append(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100))

gpucb_rewards_per_experiment=[]
gpts_rewards_per_experiment=[]

for e in range(0, n_experiment):
  env = BiddingEnvironment(best_price, bids, sigma, customers)
  gpts_learner = GPTS_Learner(n_arms = n_arms, arms = bids)
  gpucb_learner = GPUCB_Learner(n_arms = n_arms, arms = bids)
  
  for t in range (1, T+1):
    # GPTS Learner
    pulled_arm = gpts_learner.pull_arm()
    reward = env.round(pulled_arm)[0]
    gpts_learner.update(pulled_arm, reward)

    # GPUCB Learner
    beta = 2 * np.log(n_arms * t**2 * np.pi**2 /(6 * 0.05))
    pulled_arm = gpucb_learner.pull_arm(beta)
    reward = env.round(pulled_arm)[0]
    gpucb_learner.update(pulled_arm, reward)

  gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
  gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)


opt = getOptimal()[2][0]

plot(opt, T, (gpucb_rewards_per_experiment, gpts_rewards_per_experiment), ('GPUCB', 'GPTS'))