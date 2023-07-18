
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
# customers.append(Customer('C2', -0.5, -1.5, 80,[0.8,0.78,0.63,0.48,0.3]))
# customers.append(Customer('C3', -5, 0.3, 65,[0.7,0.6,0.41,0.22,0.1]))

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
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


swucb_w1_rewards_per_experiment = []
swucb_w2_rewards_per_experiment = []
swucb_w3_rewards_per_experiment = []
swucb_w4_rewards_per_experiment = []
swucb_w5_rewards_per_experiment = []
swucb_w6_rewards_per_experiment = []


for e in range(0, n_experiments):
   
    swucb_w1_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w2_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w3_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w4_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w5_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w6_env = Non_Stationary_Environment(prb, T, n_phases)
    
    swucb_learner_w1 = SWUCB_Learner(n_arms, prices,int(sqrt(T)) ,margin, clicks, cost)
    swucb_learner_w2 = SWUCB_Learner(n_arms, prices,int(1.5*sqrt(T)), margin, clicks, cost)
    swucb_learner_w3 = SWUCB_Learner(n_arms, prices,int(2 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w4 = SWUCB_Learner(n_arms, prices,int(2.5 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w5 = SWUCB_Learner(n_arms, prices,int(3 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w6 = SWUCB_Learner(n_arms, prices,int(3.5 * sqrt(T)), margin, clicks, cost)
    

    for t in range(0, T):
        # SWUCB window size= sqrt(T)
        pulled_arm = swucb_learner_w1.pull_arm()
        reward = swucb_w1_env.round(pulled_arm, clicks)
        swucb_learner_w1.update(pulled_arm, reward)

        # SWUCB1 window size = 1.5 * sqrt(T)
        pulled_arm = swucb_learner_w2.pull_arm()
        reward = swucb_w2_env.round(pulled_arm, clicks)
        swucb_learner_w2.update(pulled_arm, reward)

        # SWUCB1 window size = 2*sqrt(T)
        pulled_arm = swucb_learner_w3.pull_arm()        
        reward = swucb_w3_env.round(pulled_arm, clicks)
        swucb_learner_w3.update(pulled_arm, reward)

        # SWUCB1 window size = 2.5 * sqrt(T)
        pulled_arm = swucb_learner_w4.pull_arm()
        reward = swucb_w4_env.round(pulled_arm, clicks)
        swucb_learner_w4.update(pulled_arm, reward)

        # SWUCB1 window size= 3*sqrt(T)
        
        pulled_arm = swucb_learner_w5.pull_arm()
        reward = swucb_w5_env.round(pulled_arm, clicks)
        swucb_learner_w5.update(pulled_arm, reward)
        
        # SWUCB1 window size= 3.5*sqrt(T)
        pulled_arm = swucb_learner_w6.pull_arm()
        reward = swucb_w6_env.round(pulled_arm, clicks)
        swucb_learner_w6.update(pulled_arm, reward)


    swucb_w1_rewards_per_experiment.append(swucb_learner_w1.collected_rewards)
    swucb_w2_rewards_per_experiment.append(swucb_learner_w2.collected_rewards)
    swucb_w3_rewards_per_experiment.append(swucb_learner_w3.collected_rewards)
    swucb_w4_rewards_per_experiment.append(swucb_learner_w4.collected_rewards)
    swucb_w5_rewards_per_experiment.append(swucb_learner_w5.collected_rewards)
    swucb_w6_rewards_per_experiment.append(swucb_learner_w6.collected_rewards)


swucb_w1_rewards_per_experiment = np.array(swucb_w1_rewards_per_experiment)
swucb_w2_rewards_per_experiment = np.array(swucb_w2_rewards_per_experiment)
swucb_w3_rewards_per_experiment = np.array(swucb_w3_rewards_per_experiment)
swucb_w4_rewards_per_experiment = np.array(swucb_w4_rewards_per_experiment)
swucb_w5_rewards_per_experiment = np.array(swucb_w5_rewards_per_experiment)
swucb_w6_rewards_per_experiment = np.array(swucb_w6_rewards_per_experiment)


swucb_w1_regret = np.zeros(T)
swucb_w2_regret = np.zeros(T)
swucb_w3_regret = np.zeros(T)
swucb_w4_regret = np.zeros(T)
swucb_w5_regret = np.zeros(T)
swucb_w6_regret = np.zeros(T)



swucb_w1_std = np.zeros(T)
swucb_w2_std = np.zeros(T)
swucb_w3_std = np.zeros(T)
swucb_w4_std = np.zeros(T)
swucb_w5_std = np.zeros(T)
swucb_w6_std = np.zeros(T)

opt_per_phase = rewards.max(axis=1)
optimum_per_round = np.zeros(T)

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]

    # Regret
    
    swucb_w1_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w3_rewards_per_experiment, axis=0)[t_index]
    swucb_w4_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w4_rewards_per_experiment, axis=0)[t_index]
    swucb_w5_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w5_rewards_per_experiment, axis=0)[t_index]
    swucb_w6_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w6_rewards_per_experiment, axis=0)[t_index]


    # Standard deviation instantaneous regret
    
    swucb_w1_std[t_index] = np.std(opt_per_phase[i] - swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_std[t_index] = np.std(opt_per_phase[i] - swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_std[t_index] = np.std(opt_per_phase[i] - swucb_w3_rewards_per_experiment, axis=0)[t_index]
    swucb_w4_std[t_index] = np.std(opt_per_phase[i] - swucb_w4_rewards_per_experiment, axis=0)[t_index]
    swucb_w5_std[t_index] = np.std(opt_per_phase[i] - swucb_w5_rewards_per_experiment, axis=0)[t_index]
    swucb_w6_std[t_index] = np.std(opt_per_phase[i] - swucb_w6_rewards_per_experiment, axis=0)[t_index]
    


swucb_w1_label = r"$SW\ UCB1,\ window\ size=\sqrt{T}$"
swucb_w2_label = r"$SW\ UCB1,\ window\ size=\frac{3}{2}\sqrt{T}$"
swucb_w3_label = r"$SW\ UCB1,\ window\ size=2\sqrt{T}$"
swucb_w4_label = r"$SW\ UCB1,\ window\ size=\frac{5}{2}\sqrt{T}$"
swucb_w5_label = r"$SW\ UCB1,\ window\ size=3\sqrt{T}$"
swucb_w6_label = r"$SW\ UCB1,\ window\ size=\frac{7}{2}\sqrt{T}$"


x = list(range(0,T))
stdswucb_w1 = [(np.cumsum(swucb_w1_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w2 = [(np.cumsum(swucb_w2_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w3 = [(np.cumsum(swucb_w3_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w4 = [(np.cumsum(swucb_w4_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w5 = [(np.cumsum(swucb_w5_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w6 = [(np.cumsum(swucb_w6_regret))[:i].std() for i in range(1, T + 1)]
# %%
# Cumulative regret

line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
fill_colors = ['#FFC0CB', '#FFD700', '#00FF00', '#00FFFF', '#FFA500', '#FF00FF']
plt.figure("Cumulative regret")
plt.title("Cumulative regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(swucb_w1_regret), color=line_colors[0], label=swucb_w1_label)
plt.plot(np.cumsum(swucb_w2_regret), color=line_colors[1], label=swucb_w2_label)
plt.plot(np.cumsum(swucb_w3_regret), color=line_colors[2], label=swucb_w3_label)
plt.plot(np.cumsum(swucb_w4_regret), color=line_colors[3], label=swucb_w4_label)
plt.plot(np.cumsum(swucb_w5_regret), color=line_colors[4], label=swucb_w5_label)
plt.plot(np.cumsum(swucb_w6_regret), color=line_colors[5], label=swucb_w6_label)



plt.fill_between(x, np.cumsum(swucb_w1_regret)+stdswucb_w1, np.cumsum(swucb_w1_regret)-stdswucb_w1,
    alpha=0.25, facecolor=fill_colors[0],
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w2_regret)+stdswucb_w2, np.cumsum(swucb_w2_regret)-stdswucb_w2,
    alpha=0.25, facecolor=fill_colors[1],
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w3_regret)+stdswucb_w3, np.cumsum(swucb_w3_regret)-stdswucb_w3,
    alpha=0.25, facecolor=fill_colors[2],
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w4_regret)+stdswucb_w4, np.cumsum(swucb_w4_regret)-stdswucb_w4,
    alpha=0.25, facecolor=fill_colors[3],
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w5_regret)+stdswucb_w5, np.cumsum(swucb_w5_regret)-stdswucb_w5,
    alpha=0.25, facecolor=fill_colors[4],
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w6_regret)+stdswucb_w6, np.cumsum(swucb_w6_regret)-stdswucb_w6,
    alpha=0.25, facecolor=fill_colors[5],
    linewidth=0)



#plt.legend(loc=0)
plt.legend(loc="upper left",fontsize="large")
plt.show()
