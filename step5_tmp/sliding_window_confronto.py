
# %%
from Customer import *
from UserClass import *
from UCB1_Learner import *
from Non_Stationary_Environment import *
from SWUCB_Learner import *
from CDUCB_Learner import *
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt



customers = []

customers.append(Customer('C1', -1.5, 0.1, 100, [0.95,0.70,0.53,0.28,0.14]))
customers.append(Customer('C2', -0.5, -1.5, 80,[0.8,0.78,0.63,0.48,0.3]))
customers.append(Customer('C3', -5, 0.3, 65,[0.7,0.6,0.41,0.22,0.1]))

c1 = UserClass(
    np.array([10, 20, 30, 40, 50]),
    np.array([
        [0.95, 0.70, 0.53, 0.28, 0.14],
        [0.80, 0.65, 0.43, 0.15, 0.50],
        [0.75, 0.64, 0.26, 0.12, 0.02]
    ])
)

prb = (c1.prices - 8) * c1.probabilities
best_gain = prb.max(axis=1)

clicks = int(customers[0].num_clicks(2))
cost = customers[0].cum_cost_clicks(2)
margin = c1.prices - 8
rewards = (margin * c1.probabilities - cost) * clicks
print(rewards)

n_arms = len(c1.prices)

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
   
    swucb_w1_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    swucb_w2_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    swucb_w3_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    swucb_w4_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    swucb_w5_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    swucb_w6_env = Non_Stationary_Environment(c1.probabilities, T, n_phases)
    
    swucb_learner_w1 = SWUCB_Learner(n_arms, c1.prices,int(sqrt(T)) ,margin, clicks, cost)
    swucb_learner_w2 = SWUCB_Learner(n_arms, c1.prices,int(1.5*sqrt(T)), margin, clicks, cost)
    swucb_learner_w3 = SWUCB_Learner(n_arms, c1.prices,int(2 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w4 = SWUCB_Learner(n_arms, c1.prices,int(2.5 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w5 = SWUCB_Learner(n_arms, c1.prices,int(3 * sqrt(T)), margin, clicks, cost)
    swucb_learner_w6 = SWUCB_Learner(n_arms, c1.prices,int(3.5 * sqrt(T)), margin, clicks, cost)
    

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
plt.figure("Cumulative regret")
plt.title("Cumulative regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(swucb_w1_regret), 'r', label=swucb_w1_label)
plt.plot(np.cumsum(swucb_w2_regret), 'g', label=swucb_w2_label)
plt.plot(np.cumsum(swucb_w3_regret), 'gold', label=swucb_w3_label)
plt.plot(np.cumsum(swucb_w4_regret), 'b', label=swucb_w4_label)
plt.plot(np.cumsum(swucb_w5_regret), 'brown', label=swucb_w5_label)
plt.plot(np.cumsum(swucb_w6_regret), 'pink', label=swucb_w6_label)



plt.fill_between(x, np.cumsum(swucb_w1_regret)+stdswucb_w1, np.cumsum(swucb_w1_regret)-stdswucb_w1,
    alpha=0.5, facecolor='#D3A18D',
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w2_regret)+stdswucb_w2, np.cumsum(swucb_w2_regret)-stdswucb_w2,
    alpha=0.5, facecolor='#8DD39D',
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w3_regret)+stdswucb_w3, np.cumsum(swucb_w3_regret)-stdswucb_w3,
    alpha=0.5, facecolor='#D3D38D',
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w4_regret)+stdswucb_w4, np.cumsum(swucb_w4_regret)-stdswucb_w4,
    alpha=0.5, facecolor='#8D99D3',
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w5_regret)+stdswucb_w5, np.cumsum(swucb_w5_regret)-stdswucb_w5,
    alpha=0.5, facecolor='#3C1632',
    linewidth=0)

plt.fill_between(x, np.cumsum(swucb_w6_regret)+stdswucb_w6, np.cumsum(swucb_w6_regret)-stdswucb_w6,
    alpha=0.5, facecolor='#A22868',
    linewidth=0)



plt.legend(loc=0)
#plt.legend(loc=0, bbox_to_anchor=(1.02, 1))
plt.show()
