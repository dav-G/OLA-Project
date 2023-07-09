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

def generateContext(rewards, pulled_arms, pulled_arms_features, features):
	delta = 0.9
	for feature in features:
		indexes1 = np.array( [ i if pulled_features[feature] for i,pulled_features in enumerate(pulled_arms_features) ] )
		indexes2 = np.array( [ i if not pulled_features[feature] for i,pulled_features in enumerate(pulled_arms_features) ] )
		
		rewards1, arms1, features1 = rewards[indexes1], pulled_arms[indexes1], pulled_arms_features[indexes1]
		rewards2, arms2, features2 = rewards[indexes2], pulled_arms[indexes2], pulled_arms_features[indexes2]
		
		tmp = [0] * n_arms
		for i in len(rewards1):
			tmp[arms1[i]] += rewards1[i]
		expected_reward1 = np.max(tmp)
		
		tmp = [0] * n_arms
		for i in len(rewards2):
			tmp[arms2[i]] += rewards2[i]
		expected_reward2 = np.max(tmp)
		
		tmp = [0] * n_arms
		for i in len(rewards):
			tmp[pulled_arms[i]] += rewards[i]
		expected_reward = np.max(tmp)
		
		
		prob1 = len(rewards1)/len(rewards) - np.sqrt(np.log(delta)/(2*len(rewards1)))
		prob2 = len(rewards2)/len(rewards) - np.sqrt(np.log(delta)/(2*len(rewards2)))
		
		if prob1*expected_reward1 + prob2*expected_reward2 >= expected_reward:
			new_features = features.copy().remove(feature)
			context1 = generateContext(rewards1, arms1, features1, new_features)
			context2 = generateContext(rewards2, arms2, features2, new_features)
			for i in len(context1):
				context1[i][feature] = True
			for i in len(context2):
				context2[i][feature] = False
			return context1 + context2
		
	return [{}]
	

# %%
n_arms=100 #da cambiare bisogna mettere 100 utilizzare uno minure solo per vedere le convergenze
min_bid=0.0
max_bid=2.0
bids=np.linspace(min_bid,max_bid,n_arms)
sigma_num=15
sigma_cost=0.5
T=60 #horizon ricordarsi di cambiare prima del run finale perchè deve essere 365
n_experiment=5
best_price=10 #da cambiare vedere cosa esce da step 1

customers = []
customers.append(Customer('C1', -1.5, 0.1, 100, {'student': True, 'commuter': True}))
customers.append(Customer('C2', -0.5, -1.5, 80), {'student': False, 'commuter': True})
customers.append(Customer('C3', -5, 0.3, 65), {'student': False, 'commuter': False})

# %%

gpucb_rewards_cost_per_experiment=[]
gpts_rewards_cost_per_experiment=[]

gpucb_rewards_num_per_experiment=[]
gpts_rewards_num_per_experiment=[]
for e in range(0,n_experiment):
  beta=0
  env1=BiddingEnvironmentCost(bids,sigma_cost,customers)
  gpts_learner_cost=GPTS_Learner_Lo(n_arms=n_arms,arms=bids)
  gpucb_learner_cost=GPUCB_LO_Learner(n_arms=n_arms,arms=bids)
  for t in range (1,T+1):
    #GPTS Learner
    pulled_arm=gpts_learner_cost.pull_arm()
    reward, features =env1.round(pulled_arm)
    gpts_learner_cost.update(pulled_arm, reward, features)

	if t%14 == 0:
	  contexts = generateContext(gpts_learner_cost.collected_rewards, gpts_learner_cost.pulled_arms_idx, , ['student', 'commuter'])
	  gpts.updateContexts(contexts)
	
    #gpucb Learner
    beta=2*np.log(n_arms*t**2*np.pi**2/(6*0.05))
    pulled_arm=gpucb_learner_cost.pull_arm(beta)
    reward=env1.round(pulled_arm)
    gpucb_learner_cost.update(pulled_arm, reward)

  gpts_rewards_cost_per_experiment.append(gpts_learner_cost.collected_rewards)
  gpucb_rewards_cost_per_experiment.append(gpucb_learner_cost.collected_rewards)

  env2=BiddingEnvironmentClicks(bids,sigma_num,customers)
  gpts_learner_clicks=GPTS_Learner(n_arms=n_arms,arms=bids)
  gpucb_learner_clicks=GPUCB_UP_Learner(n_arms=n_arms,arms=bids)
  for t in range (1,T+1):
    #GPTS Learner
    pulled_arm=gpts_learner_clicks.pull_arm()
    reward=env2.round(pulled_arm)
    gpts_learner_clicks.update(pulled_arm, reward)

    #gpucb Learner
    beta=2*np.log(n_arms*t**2*np.pi**2/(6*0.05))
    pulled_arm=gpucb_learner_clicks.pull_arm(beta)
    reward=env2.round(pulled_arm)
    gpucb_learner_clicks.update(pulled_arm, reward)

  gpts_rewards_num_per_experiment.append(gpts_learner_clicks.collected_rewards)
  gpucb_rewards_num_per_experiment.append(gpucb_learner_clicks.collected_rewards)


# %%
#prendere opt dal clairovoiant
opt=68


gpts_rewards_per_experiment=np.array(gpts_rewards_num_per_experiment)*(best_price-np.array(gpts_rewards_cost_per_experiment))
gpucb_rewards_per_experiment=np.array(gpucb_rewards_num_per_experiment)*(best_price-np.array(gpucb_rewards_cost_per_experiment))
x=list(range(0,T))
#cost=0.6
#num=65
#gpts_rewards_per_experiment=best_price*np.array(num)-np.array(cost)
#gpucb_rewards_per_experiment=best_price*np.array(num)-np.array(gpucb_rewards_cost_per_experiment)


# %%
#Cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
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
plt.ylabel("Standard deviation of Cumulative Regret")

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
plt.ylabel("Cumulative Regret")
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
