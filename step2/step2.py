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
import pandas

# %%
n_arms=100
min_bid=0.0
max_bid=1.0
bids=np.linspace(min_bid,max_bid,n_arms)
sigma=10
T=30 #horizon
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
  env1=BiddingEnvironmentCost(bids,sigma,customers[0])
  gpts_learner_cost=GPTS_Learner_Lo(n_arms=n_arms,arms=bids)
  gpucb_learner_cost=GPUCB_LO_Learner(n_arms=n_arms,arms=bids)
  for t in range (0,T):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward=env1.round(pulled_arm)
    gpts_learner_cost.update(pulled_arm, reward)

    #gpucb Learner
    pulled_arm=gpucb_learner_cost.pull_arm()
    reward=env1.round(pulled_arm)
    gpucb_learner_cost.update(pulled_arm, reward)

  gpts_rewards_cost_per_experiment.append(gpts_learner_cost.collected_rewards)
  gpucb_rewards_cost_per_experiment.append(gpucb_learner_cost.collected_rewards)



  env2=BiddingEnvironmentClicks(bids,sigma,customers[0])
  gpts_learner_clicks=GPTS_Learner_Lo(n_arms=n_arms,arms=bids)
  gpucb_learner_clicks=GPUCB_LO_Learner(n_arms=n_arms,arms=bids)
  for t in range (0,T):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward=env2.round(pulled_arm)
    gpts_learner_clicks.update(pulled_arm, reward)

    #gpucb Learner
    pulled_arm=gpucb_learner_cost.pull_arm()
    reward=env2.round(pulled_arm)
    gpucb_learner_clicks.update(pulled_arm, reward)

  gpts_rewards_num_per_experiment.append(gpts_learner_clicks.collected_rewards)
  gpucb_rewards_num_per_experiment.append(gpucb_learner_clicks.collected_rewards)


# %%
#prendere opt dal clairovoiant
opt=2500


gpts_rewards_per_experiment=best_price*np.array(gpts_rewards_num_per_experiment)-np.array(gpts_rewards_cost_per_experiment)
gpucb_rewards_per_experiment=best_price*np.array(gpucb_rewards_num_per_experiment)-np.array(gpucb_rewards_cost_per_experiment)


x=list(range(0,T))
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
#Standard deviation of cumulative regret da sistemare
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

stducb = [(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment,axis=0)))[:i].std() for i in range(1,T+1)]
stdts = [np.cumsum(np.mean(opt-gpts_rewards_per_experiment,axis=0))[:i].std() for i in range(1,T+1)]
#exstducb=(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment,axis=0))).expanding().std(ddof=0)
#exstdts=(np.cumsum(np.mean(opt-gpts_rewards_per_experiment,axis=0))).expanding().std(ddof=0)
#GPTS,=exstdts.plot(color='b')
#GPUCB,=exstducb.plot(color='b')
GPUCB,=plt.plot(stducb,'r')
GPTS,=plt.plot(stdts,'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%%
#Cumulative reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(np.cumsum(np.mean(gpucb_rewards_per_experiment, axis=0)),'r')
GPTS,=plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)),'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%%
#istantaneous regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(x,np.mean(opt-gpucb_rewards_per_experiment,axis=0),'r')
GPTS,=plt.plot(x,np.mean(opt-gpts_rewards_per_experiment,axis=0),'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
#%%
#istantaneous reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(x,np.mean(gpucb_rewards_per_experiment,axis=0),'r')
GPTS,=plt.plot(x,np.mean(gpts_rewards_per_experiment,axis=0),'b')
plt.legend([GPUCB,GPTS],["gpucb","GPTS"])
plt.show()
# %%
