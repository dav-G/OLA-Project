# %%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import matplotlib.pyplot as plt
from BiddingEnvironment import*
from GPTS_Learner import*
from GPUCB_Learner import*

# %%
n_arms=100
min_bid=0.0
max_bid=1.0
bids=np.linspace(min_bid,max_bid,n_arms)
sigma=10
T=365 #horizon
n_experiment=1
best_price=30 #da cambiare vedere cosa esce da step 1

customers = []
customers.append(Customer('C1', -1.5, 0.5, 100))
customers.append(Customer('C2', -0.5, -1.5, 80))
customers.append(Customer('C3', -5, 0.3, 65))

gpucb_rewards_cost_per_experiment=[]
gpts_rewards_cost_per_experiment=[]

gpucb_rewards_num_per_experiment=[]
gpts_rewards_num_per_experiment=[]


# %%
for e in range(0,n_experiment):
  env=BiddingEnvironmentCost(bids,sigma,customers[0])
  gpts_learner_cost=GPTS_Learner_Lo(n_arms=n_arms,arms=bids)
  gpucb_learner_cost=GPUCB_LO_Learner(n_arms=n_arms,arms=bids)
  for t in range (0,T):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward=env.round(pulled_arm)
    gpts_learner_cost.update(pulled_arm, reward)

    #gpucb Learner
    pulled_arm=gpucb_learner_cost.pull_arm()
    reward=env.round(pulled_arm)
    gpucb_learner_cost.update(pulled_arm, reward)

  gpts_rewards_cost_per_experiment.append(gpts_learner_cost.collected_rewards)
  gpucb_rewards_cost_per_experiment.append(gpucb_learner_cost.collected_rewards)



  env=BiddingEnvironmentClicks(bids,sigma,customers[0])
  gpts_learner_clicks=GPTS_Learner_Lo(n_arms=n_arms,arms=bids)
  gpucb_learner_clicks=GPUCB_LO_Learner(n_arms=n_arms,arms=bids)
  for t in range (0,T):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward=env.round(pulled_arm)
    gpts_learner_cost.update(pulled_arm, reward)

    #gpucb Learner
    pulled_arm=gpucb_learner_cost.pull_arm()
    reward=env.round(pulled_arm)
    gpucb_learner_cost.update(pulled_arm, reward)

  gpts_rewards_num_per_experiment.append(gpts_learner_cost.collected_rewards)
  gpucb_rewards_num_per_experiment.append(gpucb_learner_cost.collected_rewards)


# %%
#prendere opt dal clairovoiant
opt=1.4

gpts_rewards_per_experiment=best_price*gpts_rewards_num_per_experiment-gpts_rewards_cost_per_experiment
gpucb_rewards_per_experiment=best_price*gpucb_rewards_num_per_experiment-gpucb_rewards_cost_per_experiment


x=list(range(1,T+1))
# %%
#Cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0)),'r')
GPTS,=plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0)),'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
# %%
#Standard deviation of cumulative regret (forse nello stesso grafico?)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(np.std(np.cumsum(opt-gpucb_rewards_per_experiment,axis=0),axis=0),'r')
GPTS,=plt.plot(np.std(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0)),'b'))
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%
#Cumulative reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(np.cumsum(np.mean(gpucb_rewards_per_experiment, axis=0)),'r')
GPTS,=plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)),'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%
#istantaneous regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(x,opt-gpucb_rewards_per_experiment,'r')
GPTS,=plt.plot(x,opt-gpts_rewards_per_experiment,'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%
#istantaneous reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(x,gpucb_rewards_per_experiment,'r')
GPTS,=plt.plot(x,gpts_rewards_per_experiment,'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()