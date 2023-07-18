# %%
from Customer import *
from UCB1_Learner import *
from Non_Stationary_Environment import *
from SWUCB_Learner import *
from CDUCB_Learner import *
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt



c1=Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)

prices = np.array([10, 20, 30, 40, 50])
prb = np.array([
    [0.8, 0.70, 0.53, 0.28, 0.14],
    [0.4, 0.5, 0.65, 0.75, 0.8],
    [0.7, 0.6, 0.46, 0.19, 0.07]
])




margin = (prices - 8)
clicks = int(c1.num_clicks(2))
cost = c1.click_cost(2)
rewards = (margin * prb - cost) * clicks

n_arms = len(prices)



T = 365
n_phases = 3
phases_len = int(T / n_phases)
n_experiments = 30


M = 10
#eps = 0.4
h = np.log(T)
alpha = 0.1

cducb_w1_rewards_per_experiment = []
cducb_w2_rewards_per_experiment = []
cducb_w3_rewards_per_experiment = []
cducb_w4_rewards_per_experiment = []
cducb_w5_rewards_per_experiment = []
cducb_w6_rewards_per_experiment = []


for e in range(0, n_experiments):
   
    cducb_w1_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_w2_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_w3_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_w4_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_w5_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_w6_env = Non_Stationary_Environment(prb, T, n_phases)
    
    cducb_learner_w1 = CDUCB_Learner(n_arms, prices, M, 0.1, h, alpha, margin, clicks, cost)
    cducb_learner_w2 = CDUCB_Learner(n_arms, prices, M, 0.2, h, alpha, margin, clicks, cost)
    cducb_learner_w3 = CDUCB_Learner(n_arms, prices, M, 0.3, h, alpha, margin, clicks, cost)
    cducb_learner_w4 = CDUCB_Learner(n_arms, prices, M, 0.4, h, alpha, margin, clicks, cost)
    cducb_learner_w5 = CDUCB_Learner(n_arms, prices, M, 0.5, h, alpha, margin, clicks, cost)
    cducb_learner_w6 = CDUCB_Learner(n_arms, prices, M, 0.8, h, alpha, margin, clicks, cost)
    

    for t in range(0, T):
        # CDUCB M=10
        pulled_arm = cducb_learner_w1.pull_arm()
        reward = cducb_w1_env.round(pulled_arm, clicks)
        cducb_learner_w1.update(pulled_arm, reward)

        # CDUCB M=15
        pulled_arm = cducb_learner_w2.pull_arm()
        reward = cducb_w2_env.round(pulled_arm, clicks)
        cducb_learner_w2.update(pulled_arm, reward)

        # CDUCB M=20
        pulled_arm = cducb_learner_w3.pull_arm()        
        reward = cducb_w3_env.round(pulled_arm, clicks)
        cducb_learner_w3.update(pulled_arm, reward)

        # CDUCB M=25
        pulled_arm = cducb_learner_w4.pull_arm()
        reward = cducb_w4_env.round(pulled_arm, clicks)
        cducb_learner_w4.update(pulled_arm, reward)

        # CDUCB M=30
        
        pulled_arm = cducb_learner_w5.pull_arm()
        reward = cducb_w5_env.round(pulled_arm, clicks)
        cducb_learner_w5.update(pulled_arm, reward)
        
        # CDUCB M=35
        pulled_arm = cducb_learner_w6.pull_arm()
        reward = cducb_w6_env.round(pulled_arm, clicks)
        cducb_learner_w6.update(pulled_arm, reward)


    cducb_w1_rewards_per_experiment.append(cducb_learner_w1.collected_rewards)
    cducb_w2_rewards_per_experiment.append(cducb_learner_w2.collected_rewards)
    cducb_w3_rewards_per_experiment.append(cducb_learner_w3.collected_rewards)
    cducb_w4_rewards_per_experiment.append(cducb_learner_w4.collected_rewards)
    cducb_w5_rewards_per_experiment.append(cducb_learner_w5.collected_rewards)
    cducb_w6_rewards_per_experiment.append(cducb_learner_w6.collected_rewards)


cducb_w1_rewards_per_experiment = np.array(cducb_w1_rewards_per_experiment)
cducb_w2_rewards_per_experiment = np.array(cducb_w2_rewards_per_experiment)
cducb_w3_rewards_per_experiment = np.array(cducb_w3_rewards_per_experiment)
cducb_w4_rewards_per_experiment = np.array(cducb_w4_rewards_per_experiment)
cducb_w5_rewards_per_experiment = np.array(cducb_w5_rewards_per_experiment)
cducb_w6_rewards_per_experiment = np.array(cducb_w6_rewards_per_experiment)


cducb_w1_regret = np.zeros(T)
cducb_w2_regret = np.zeros(T)
cducb_w3_regret = np.zeros(T)
cducb_w4_regret = np.zeros(T)
cducb_w5_regret = np.zeros(T)
cducb_w6_regret = np.zeros(T)



cducb_w1_std = np.zeros(T)
cducb_w2_std = np.zeros(T)
cducb_w3_std= np.zeros(T)
cducb_w4_std = np.zeros(T)
cducb_w5_std = np.zeros(T)
cducb_w6_std = np.zeros(T)

opt_per_phase = rewards.max(axis=1)
optimum_per_round = np.zeros(T)

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]

    # Regret
    
    cducb_w1_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w1_rewards_per_experiment, axis=0)[t_index]
    cducb_w2_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w2_rewards_per_experiment, axis=0)[t_index]
    cducb_w3_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_w4_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w4_rewards_per_experiment, axis=0)[t_index]
    cducb_w5_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w5_rewards_per_experiment, axis=0)[t_index]
    cducb_w6_regret[t_index] = np.mean(opt_per_phase[i] - cducb_w6_rewards_per_experiment, axis=0)[t_index]


    # Standard deviation instantaneous regret
    
    cducb_w1_std[t_index] = np.std(opt_per_phase[i] - cducb_w1_rewards_per_experiment, axis=0)[t_index]
    cducb_w2_std[t_index] = np.std(opt_per_phase[i] - cducb_w2_rewards_per_experiment, axis=0)[t_index]
    cducb_w3_std[t_index] = np.std(opt_per_phase[i] - cducb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_w4_std[t_index] = np.std(opt_per_phase[i] - cducb_w4_rewards_per_experiment, axis=0)[t_index]
    cducb_w5_std[t_index] = np.std(opt_per_phase[i] - cducb_w5_rewards_per_experiment, axis=0)[t_index]
    cducb_w6_std[t_index] = np.std(opt_per_phase[i] - cducb_w6_rewards_per_experiment, axis=0)[t_index]
    


cducb_w1_label = r"$CD\ UCB1,\ \epsilon=0.1 $"
cducb_w2_label = r"$CD\ UCB1,\ \epsilon=0.2 $"
cducb_w3_label = r"$CD\ UCB1,\ \epsilon=0.3 $"
cducb_w4_label = r"$CD\ UCB1,\ \epsilon=0.4 $"
cducb_w5_label = r"$CD\ UCB1,\ \epsilon=0.5 $"
cducb_w6_label = r"$CD\ UCB1,\ \epsilon=0.8 $"


x = list(range(0,T))
stdcducb_w1 = [(np.cumsum(cducb_w1_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_w2 = [(np.cumsum(cducb_w2_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_w3 = [(np.cumsum(cducb_w3_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_w4 = [(np.cumsum(cducb_w4_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_w5 = [(np.cumsum(cducb_w5_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_w6 = [(np.cumsum(cducb_w6_regret))[:i].std() for i in range(1, T + 1)]
# %%
# Cumulative regret
line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
fill_colors = ['#FFC0CB', '#FFD700', '#00FF00', '#00FFFF', '#FFA500', '#FF00FF']
plt.figure("Cumulative regret")
plt.title("Cumulative regret")
plt.title("M=10,h=log(T),alpha=0.1", loc='right')
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(cducb_w1_regret), color=line_colors[0], label=cducb_w1_label)
plt.plot(np.cumsum(cducb_w2_regret), color=line_colors[1], label=cducb_w2_label)
plt.plot(np.cumsum(cducb_w3_regret), color=line_colors[2], label=cducb_w3_label)
plt.plot(np.cumsum(cducb_w4_regret), color=line_colors[3], label=cducb_w4_label)
plt.plot(np.cumsum(cducb_w5_regret), color=line_colors[4], label=cducb_w5_label)
plt.plot(np.cumsum(cducb_w6_regret), color=line_colors[5], label=cducb_w6_label)



plt.fill_between(x, np.cumsum(cducb_w1_regret)+stdcducb_w1, np.cumsum(cducb_w1_regret)-stdcducb_w1,
    alpha=0.25, facecolor=fill_colors[0],
    linewidth=0)

plt.fill_between(x, np.cumsum(cducb_w2_regret)+stdcducb_w2, np.cumsum(cducb_w2_regret)-stdcducb_w2,
    alpha=0.25, facecolor=fill_colors[1],
    linewidth=0)

plt.fill_between(x, np.cumsum(cducb_w3_regret)+stdcducb_w3, np.cumsum(cducb_w3_regret)-stdcducb_w3,
    alpha=0.25, facecolor=fill_colors[2],
    linewidth=0)

plt.fill_between(x, np.cumsum(cducb_w4_regret)+stdcducb_w4, np.cumsum(cducb_w4_regret)-stdcducb_w4,
    alpha=0.25, facecolor=fill_colors[3],
    linewidth=0)

plt.fill_between(x, np.cumsum(cducb_w5_regret)+stdcducb_w5, np.cumsum(cducb_w5_regret)-stdcducb_w5,
    alpha=0.25, facecolor=fill_colors[4],
    linewidth=0)

plt.fill_between(x, np.cumsum(cducb_w6_regret)+stdcducb_w6, np.cumsum(cducb_w6_regret)-stdcducb_w6,
    alpha=0.25, facecolor=fill_colors[5],
    linewidth=0)



plt.legend(loc=0)
plt.legend(loc='upper left',fontsize='large')
plt.show()

