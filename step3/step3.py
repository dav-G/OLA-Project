# %%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import matplotlib.pyplot as plt
from Environment import*
from GPTS_Learner import*
from GPUCB_Learner import*
from UCB1_Learner import *
from TS_Learner import *
import pandas

# %%
n_arms_ad=100 #da cambiare bisogna mettere 100 utilizzare uno minure solo per vedere le convergenze
n_arms_pr=5

min_bid=0.0
max_bid=2.0
bids=np.linspace(min_bid,max_bid,n_arms_ad)
prices=np.linspace(10,50,n_arms_pr)
sigma_num=15
sigma_cost=0.5
T=365 #horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment=5


customers = []
customers.append(Customer('C1', -1.5, 0.5, 100, [0.95,0.82,0.53,0.28,0.14]))
customers.append(Customer('C2', -0.5, -1.5, 80,[0.8,0.78,0.63,0.48,0.3]))
customers.append(Customer('C3', -5, 0.3, 65,[0.7,0.6,0.41,0.22,0.1]))
prb=customers[0].compute_pricing(prices)

sum=0
for i in range(0,5):
  sum=sum+prb[i]**2
  
probabilities=(prb/np.sqrt(sum))
# %%

gpucb_rewards_cost_per_experiment=[]
gpts_rewards_cost_per_experiment=[]

gpucb_rewards_num_per_experiment=[]
gpts_rewards_num_per_experiment=[]

ucb1_rewards_per_experiment = []
ts_rewards_per_experiment = []
for e in range(0,n_experiment):
  beta=0
  env1=BiddingEnvironmentCost(bids,sigma_cost,customers[0])
  gpts_learner_cost=GPTS_Learner_Lo(n_arms=n_arms_ad,arms=bids)
  gpucb_learner_cost=GPUCB_LO_Learner(n_arms=n_arms_ad,arms=bids)
  for t in range (1,T+1):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward=env1.round(pulled_arm)
    gpts_learner_cost.update(pulled_arm, reward)

    #gpucb Learner
    beta=2*np.log(n_arms_ad*t**2*np.pi**2/(6*0.05))
    pulled_arm=gpucb_learner_cost.pull_arm(beta)
    reward=env1.round(pulled_arm)
    gpucb_learner_cost.update(pulled_arm, reward)

  gpts_rewards_cost_per_experiment.append(gpts_learner_cost.collected_rewards)
  gpucb_rewards_cost_per_experiment.append(gpucb_learner_cost.collected_rewards)

  env2=BiddingEnvironmentClicks(bids,sigma_num,customers[0])
  gpts_learner_clicks=GPTS_Learner(n_arms=n_arms_ad,arms=bids)
  gpucb_learner_clicks=GPUCB_UP_Learner(n_arms=n_arms_ad,arms=bids)
  for t in range (1,T+1):
    #GPTS Learner
    pulled_arm=gpts_learner_clicks.pull_arm()
    reward=env2.round(pulled_arm)
    gpts_learner_clicks.update(pulled_arm, reward)

    #gpucb Learner
    beta=2*np.log(n_arms_ad*t**2*np.pi**2/(6*0.05))
    pulled_arm=gpucb_learner_clicks.pull_arm(beta)
    reward=env2.round(pulled_arm)
    gpucb_learner_clicks.update(pulled_arm, reward)

  gpts_rewards_num_per_experiment.append(gpts_learner_clicks.collected_rewards)
  gpucb_rewards_num_per_experiment.append(gpucb_learner_clicks.collected_rewards)

  envp = PricingEnvironment(prb)
  ucb1_learner = UCB1_Learner(n_arms_pr)
  ts_learner = TS_Learner(n_arms_pr)
  for t in range(0,T):
    #UCB1
    pulled_arm = ucb1_learner.pull_arm()
    reward = envp.round(pulled_arm)
    ucb1_learner.update(pulled_arm, reward)

        #Thompson Sampling Learner
    pulled_arm = ts_learner.pull_arm()
    reward = envp.round(pulled_arm)
    ts_learner.update(pulled_arm, reward)

  ts_rewards_per_experiment.append(ts_learner.collected_rewards)
  ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)





# %%
#prendere opt dal clairovoiant
opt=700


gpts_rewards_per_experiment=best_price*np.array(gpts_rewards_num_per_experiment)-np.array(gpts_rewards_cost_per_experiment)
gpucb_rewards_per_experiment=best_price*np.array(gpucb_rewards_num_per_experiment)-np.array(gpucb_rewards_cost_per_experiment)
x=list(range(0,T))
#cost=0.6
#num=65
#gpts_rewards_per_experiment=best_price*np.array(num)-np.array(cost)
#gpucb_rewards_per_experiment=best_price*np.array(num)-np.array(gpucb_rewards_cost_per_experiment)


# %%
#Cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
GPUCB,=plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0)),'r')
GPTS,=plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0)),'b')
plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0))+np.std(gpucb_rewards_per_experiment,axis=0),'--r')
plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0))-np.std(gpucb_rewards_per_experiment,axis=0),'--r')
plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0))+np.std(gpts_rewards_per_experiment,axis=0),'--b')
plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0))-np.std(gpts_rewards_per_experiment,axis=0),'--b')

#Real confidence intervals 95% too small to see on the graph
#plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0))+0.95*np.std(gpucb_rewards_per_experiment,axis=0)/n_experiment,'--r')
#plt.plot(np.cumsum(np.mean(opt-gpucb_rewards_per_experiment, axis=0))-0.95*np.std(gpucb_rewards_per_experiment,axis=0)/n_experiment,'--r')
#plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0))+0.95*np.std(gpts_rewards_per_experiment,axis=0)/n_experiment,'--b')
#plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0))-0.95*np.std(gpts_rewards_per_experiment,axis=0)/n_experiment,'--b')

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
