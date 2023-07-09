# %%
from UserClass import *
from PricingEnvironment import *
from UCB1_Learner import *
from TS_Learner import *
from matplotlib import pyplot as plt
import numpy as np

n_arms = 5
c1 = UserClass(np.array([10, 20, 30, 40, 50]), np.array([0.95,0.82,0.53,0.28,0.14]))
print((c1.prices-8) * c1.probabilities)
prb=(c1.prices-8) * c1.probabilities
opt=max(prb)
print(opt) #the optimal reward is 11.66 which correspond to the third arm (30€)

prb=prb/np.sqrt(np.sum(prb**2))
#opt = max(c1.probabilities*(c1.prices-8))#c1.prices[np.argmax(c1.prices * c1.probabilities)]
# %%
T = 365

n_experiments = 100
ucb1_rewards_per_experiment = []
ts_rewards_per_experiment = []
#ucb1_partial_rewards_per_experiment = [[] for i in range(n_arms)]
#ts_partial_rewards_per_experiment = [[] for i in range(n_arms)]

for e in range(0, n_experiments):
    env = PricingEnvironment(prb)
    ucb1_learner = UCB1_Learner(n_arms)
    ts_learner = TS_Learner(n_arms)
    for t in range(0,T):
        #UCB1
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

        #Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)

    #for arm in range (0,n_arms-1):
     #   ts_partial_rewards_per_experiment[arm].append(ts_learner.rewards_per_arm[arm])
      #  ucb1_partial_rewards_per_experiment[arm].append(ucb1_learner.rewards_per_arm[arm])
  
# %%
#ucb1_rewards_per_experiment=np.array(np.mean(ucb1_partial_rewards_per_experiment,axis=1))*(c1.prices-8)
#ts_rewards_per_experiment=np.array(np.mean(ts_partial_rewards_per_experiment,axis=1))*(c1.prices-8)

ts_rewards_per_experiment=np.array(ts_rewards_per_experiment)*18.741590113968453
ucb1_rewards_per_experiment=np.array( ucb1_rewards_per_experiment)*18.741590113968453

# %%
#Cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
UCB1,=plt.plot(np.cumsum(np.mean(opt-ucb1_rewards_per_experiment, axis=0)),'r')
TS,=plt.plot(np.cumsum(np.mean(opt-ts_rewards_per_experiment, axis=0)),'b')
plt.plot(np.cumsum(np.mean(opt-ucb1_rewards_per_experiment, axis=0))+1.96*np.std(ucb1_rewards_per_experiment,axis=0),'--r')
plt.plot(np.cumsum(np.mean(opt-ucb1_rewards_per_experiment, axis=0))-1.96*np.std(ucb1_rewards_per_experiment,axis=0),'--r')
plt.plot(np.cumsum(np.mean(opt-ts_rewards_per_experiment, axis=0))+1.96*np.std(ts_rewards_per_experiment,axis=0),'--b')
plt.plot(np.cumsum(np.mean(opt-ts_rewards_per_experiment, axis=0))-1.96*np.std(ts_rewards_per_experiment,axis=0),'--b')

plt.legend([UCB1,TS],["UCB1","TS"])
plt.show()
# %%
#Standard deviation of cumulative regret da sistemare
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

stducb = [(np.cumsum(np.mean(opt-ucb1_rewards_per_experiment,axis=0)))[:i].std() for i in range(1,T+1)]
stdts = [np.cumsum(np.mean(opt-ts_rewards_per_experiment,axis=0))[:i].std() for i in range(1,T+1)]

UCB1,=plt.plot(stducb,'r')
TS,=plt.plot(stdts,'b')
plt.legend([UCB1,TS],["UCB1","TS"])
plt.show()
#%%
#Cumulative reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
UCB1,=plt.plot(np.cumsum(np.mean(ucb1_rewards_per_experiment, axis=0)),'r')
TS,=plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis=0)),'b')
plt.legend([UCB1,TS],["UCB1","TS"])
plt.show()
#%%
#istantaneous regret
x=list(range(0,T))
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
UCB1,=plt.plot(x,np.mean(opt-np.array(ucb1_rewards_per_experiment),axis=0),'r')
TS,=plt.plot(x,np.mean(opt-np.array(ts_rewards_per_experiment),axis=0),'b')

plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.legend([UCB1,TS],["UCB1","TS"])
plt.show()
#%%
#istantaneous reward
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
UCB1,=plt.plot(x,np.mean(np.array(ucb1_rewards_per_experiment),axis=0),'r')
TS,=plt.plot(x,np.mean(np.array(ts_rewards_per_experiment),axis=0),'b')

plt.axhline(y = opt, color = 'black', linestyle = '-')
plt.legend([UCB1,TS],["UCB1","TS"])
plt.show()
# %%

