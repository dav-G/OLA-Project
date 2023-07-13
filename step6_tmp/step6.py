# %%
from UserClass import *
from EXP3_Learner import *
from Non_Stationary_Environment import *
from SWUCB_Learner import *
from CDUCB_Learner import *
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

n_arms = 5
c1 = UserClass(
    np.array([10, 20, 30, 40, 50]),
    np.array([
        [0.95, 0.82, 0.53, 0.28, 0.14],
        [0.95, 0.64, 0.96, 0.08, 0.14],
        [0.05, 0.64, 0.96, 0.08, 0.64],
        [0.95, 0.64, 0.96, 0.08, 0.14],
        [0.05, 0.64, 0.96, 0.08, 0.64]
    ])
)
prb = (c1.prices - 8) * c1.probabilities
prb = prb / np.sqrt(np.sum(prb ** 2))

n_phases = 20
prb = np.tile(prb, reps=(4, 1))

T = 365

phases_len = int(T / n_phases)
n_experiments = 100
M = 50
eps = 0.15
h = 2 * np.log(T)
alpha = np.sqrt(0.5 * np.log(T) / T)
gamma = 0.8

exp3_rewards_per_experiment = []
swucb_w1_rewards_per_experiment = []
swucb_w2_rewards_per_experiment = []
swucb_w3_rewards_per_experiment = []
cducb_rewards_per_experiment = []

for e in range(0, n_experiments):
    exp3_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w1_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w2_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w3_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_env = Non_Stationary_Environment(prb, T, n_phases)

    exp3_learner = EXP3_Learner(n_arms, gamma)
    swucb_learner_w1 = SWUCB_Learner(n_arms, int(T // 2))
    swucb_learner_w2 = SWUCB_Learner(n_arms, int(sqrt(T)))
    swucb_learner_w3 = SWUCB_Learner(n_arms, int(2 * sqrt(T)))
    cducb_learner = CDUCB_Learner(n_arms, M, eps, h, alpha)

    for t in range(0, T):
        # EXP3
        pulled_arm = exp3_learner.pull_arm()
        reward = exp3_env.round(pulled_arm)
        exp3_learner.update(pulled_arm, reward)

        # UCB1 window size = T
        pulled_arm = swucb_learner_w1.pull_arm()
        reward = swucb_w1_env.round(pulled_arm)
        swucb_learner_w1.update(pulled_arm, reward)

        # UCB1 window size = sqrt(T)
        pulled_arm = swucb_learner_w2.pull_arm()        
        reward = swucb_w2_env.round(pulled_arm)
        swucb_learner_w2.update(pulled_arm, reward)

        # UCB1 window size = 2 * sqrt(T)
        pulled_arm = swucb_learner_w3.pull_arm()
        reward = swucb_w3_env.round(pulled_arm)
        swucb_learner_w3.update(pulled_arm, reward)

        # CUSUM UCB
        pulled_arm = cducb_learner.pull_arm()
        reward = cducb_env.round(pulled_arm)
        cducb_learner.update(pulled_arm, reward)

    exp3_rewards_per_experiment.append(exp3_learner.collected_rewards)
    swucb_w1_rewards_per_experiment.append(swucb_learner_w1.collected_rewards)
    swucb_w2_rewards_per_experiment.append(swucb_learner_w2.collected_rewards)
    swucb_w3_rewards_per_experiment.append(swucb_learner_w3.collected_rewards)
    cducb_rewards_per_experiment.append(cducb_learner.collected_rewards)

exp3_rewards_per_experiment = np.array(exp3_rewards_per_experiment)
swucb_w1_rewards_per_experiment = np.array(swucb_w1_rewards_per_experiment)
swucb_w2_rewards_per_experiment = np.array(swucb_w2_rewards_per_experiment)
swucb_w3_rewards_per_experiment = np.array(swucb_w3_rewards_per_experiment)
cducb_rewards_per_experiment = np.array(cducb_rewards_per_experiment)

exp3_regret = np.zeros(T)
swucb_w1_regret = np.zeros(T)
swucb_w2_regret = np.zeros(T)
swucb_w3_regret = np.zeros(T)
cducb_regret = np.zeros(T)

exp3_instantaneous_regret = np.zeros(T)
swucb_w1_instantaneous_regret = np.zeros(T)
swucb_w2_instantaneous_regret = np.zeros(T)
swucb_w3_instantaneous_regret = np.zeros(T)
cducb_instantaneous_regret = np.zeros(T)

opt_per_phase = prb.max(axis=1)
optimum_per_round = np.zeros(T)

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]

    # Regret
    exp3_regret[t_index] = np.mean(opt_per_phase[i] - exp3_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_regret[t_index] = np.mean(opt_per_phase[i] - cducb_rewards_per_experiment, axis=0)[t_index]

    # Instantaneous regret
    exp3_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(exp3_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(cducb_rewards_per_experiment, axis=0)[t_index]

exp3_label = "EXP3"
swucb_w1_label = r"$SW\ UCB1,\ window\ size=\frac{T}{2}$"
swucb_w2_label = r"$SW\ UCB1,\ window\ size=\sqrt{T}$"
swucb_w3_label = r"$SW\ UCB1,\ window\ size=2 \sqrt{T}$"
cducb_label = "CUSUM UCB1"
# %%
# Cumulative regret
plt.figure("Cumulative regret")
plt.title("Cumulative regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(exp3_regret), 'r', label=exp3_label)
plt.plot(np.cumsum(swucb_w1_regret), 'b', label=swucb_w1_label)
plt.plot(np.cumsum(swucb_w2_regret), 'g', label=swucb_w2_label)
plt.plot(np.cumsum(swucb_w3_regret), 'y', label=swucb_w3_label)
plt.plot(np.cumsum(cducb_regret), 'm', label=cducb_label)
plt.legend(loc=0)
plt.show()
# %%
# Standard deviation of cumulative regret
stdexp3 = [(np.cumsum(exp3_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w1 = [(np.cumsum(swucb_w1_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w2 = [(np.cumsum(swucb_w2_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w3 = [(np.cumsum(swucb_w3_regret))[:i].std() for i in range(1, T + 1)]
stdcducb = [(np.cumsum(cducb_regret))[:i].std() for i in range(1, T + 1)]

plt.figure("Standard deviation of cumulative regret")
plt.title("Standard deviation of cumulative regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(stdexp3, 'r', label=exp3_label)
plt.plot(stdswucb_w1, 'b', label=swucb_w1_label)
plt.plot(stdswucb_w2, 'g', label=swucb_w2_label)
plt.plot(stdswucb_w3, 'y', label=swucb_w3_label)
plt.plot(stdcducb, 'm', label=cducb_label)
plt.legend(loc=0)
plt.show()
# %%
# Cumulative reward
plt.figure("Cumulative reward")
plt.title("Cumulative reward")
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.cumsum(np.mean(exp3_rewards_per_experiment, axis=0)), 'r', label=exp3_label)
plt.plot(np.cumsum(np.mean(swucb_w1_rewards_per_experiment, axis=0)), 'b', label=swucb_w1_label)
plt.plot(np.cumsum(np.mean(swucb_w2_rewards_per_experiment, axis=0)), 'g', label=swucb_w2_label)
plt.plot(np.cumsum(np.mean(swucb_w3_rewards_per_experiment, axis=0)), 'y', label=swucb_w3_label)
plt.plot(np.cumsum(np.mean(cducb_rewards_per_experiment, axis=0)), 'm', label=cducb_label)
plt.legend(loc=0)
plt.show()
# %%
# Instantaneous regret
plt.figure("Instantaneous regret")
plt.title("Instantaneous regret")
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(exp3_instantaneous_regret), 'r', label=exp3_label)
plt.plot(np.cumsum(swucb_w1_instantaneous_regret), 'b', label=swucb_w1_label)
plt.plot(np.cumsum(swucb_w2_instantaneous_regret), 'g', label=swucb_w2_label)
plt.plot(np.cumsum(swucb_w3_instantaneous_regret), 'y', label=swucb_w3_label)
plt.plot(np.cumsum(cducb_instantaneous_regret), 'm', label=cducb_label)
plt.legend(loc=0)
plt.show()
# %%
# Instantaneous reward
plt.figure("Instantaneous reward")
plt.title("Instantaneous reward")
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(exp3_rewards_per_experiment, axis=0), 'r', label=exp3_label)
plt.plot(np.mean(swucb_w1_rewards_per_experiment, axis=0), 'b', label=swucb_w1_label)
plt.plot(np.mean(swucb_w2_rewards_per_experiment, axis=0), 'g', label=swucb_w2_label)
plt.plot(np.mean(swucb_w3_rewards_per_experiment, axis=0), 'y', label=swucb_w3_label)
plt.plot(np.mean(cducb_rewards_per_experiment, axis=0), 'm', label=cducb_label)
plt.plot(optimum_per_round, 'k--', label="Optimum")
plt.legend(loc=0)
plt.show()