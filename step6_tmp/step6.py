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
        [0.98, 0.88, 0.43, 0.25, 0.22],
        [0.86, 0.76, 0.20, 0.32, 0.08],
        [0.65, 0.68, 0.12, 0.16, 0.12],
        [0.44, 0.53, 0.28, 0.11, 0.01]
    ])
)
prb = (c1.prices - 8) * c1.probabilities
prb = prb / np.sqrt(np.sum(prb ** 2))

T = 365

n_phases = 5

n_phases_long = 100
prb_long = np.tile(prb, reps=(20, 1))

phases_len = int(T / n_phases)
phases_len_long = int(T / n_phases_long)

n_experiments = 100
M = 50
eps = 0.15
h = 2 * np.log(T)
alpha = np.sqrt(0.5 * np.log(T) / T)
gamma = 0.8

ucb1_rewards_per_experiment = []
exp3_rewards_per_experiment = []
swucb_w1_rewards_per_experiment = []
swucb_w2_rewards_per_experiment = []
swucb_w3_rewards_per_experiment = []
cducb_rewards_per_experiment = []

ucb1_long_rewards_per_experiment = []
exp3_long_rewards_per_experiment = []
swucb_w1_long_rewards_per_experiment = []
swucb_w2_long_rewards_per_experiment = []
swucb_w3_long_rewards_per_experiment = []
cducb_long_rewards_per_experiment = []

for e in range(0, n_experiments):
    ucb1_env = Non_Stationary_Environment(prb, T, n_phases)
    exp3_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w1_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w2_env = Non_Stationary_Environment(prb, T, n_phases)
    swucb_w3_env = Non_Stationary_Environment(prb, T, n_phases)
    cducb_env = Non_Stationary_Environment(prb, T, n_phases)

    ucb1_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)
    exp3_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)
    swucb_w1_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)
    swucb_w2_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)
    swucb_w3_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)
    cducb_long_env = Non_Stationary_Environment(prb_long, T, n_phases_long)

    ucb1_learner = UCB1_Learner(n_arms)
    exp3_learner = EXP3_Learner(n_arms, gamma)
    swucb_learner_w1 = SWUCB_Learner(n_arms, int(0.5 * sqrt(T)))
    swucb_learner_w2 = SWUCB_Learner(n_arms, int(sqrt(T)))
    swucb_learner_w3 = SWUCB_Learner(n_arms, int(2 * sqrt(T)))
    cducb_learner = CDUCB_Learner(n_arms, M, eps, h, alpha)

    ucb1_learner_long = UCB1_Learner(n_arms)
    exp3_learner_long = EXP3_Learner(n_arms, gamma)
    swucb_learner_w1_long = SWUCB_Learner(n_arms, int(0.5 * sqrt(T)))
    swucb_learner_w2_long = SWUCB_Learner(n_arms, int(sqrt(T)))
    swucb_learner_w3_long = SWUCB_Learner(n_arms, int(2 * sqrt(T)))
    cducb_learner_long = CDUCB_Learner(n_arms, M, eps, h, alpha)

    for t in range(0, T):
        # UCB1
        pulled_arm = ucb1_learner.pull_arm()
        reward = ucb1_env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

        # EXP3
        pulled_arm = exp3_learner.pull_arm()
        reward = exp3_env.round(pulled_arm)
        exp3_learner.update(pulled_arm, reward)

        # UCB1 window size = 0.5 * sqrt(T)
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


        # Setting with higher non-stationarity degree (more phases)
        # UCB1
        pulled_arm = ucb1_learner_long.pull_arm()
        reward = ucb1_long_env.round(pulled_arm)
        ucb1_learner_long.update(pulled_arm, reward)

        # EXP3
        pulled_arm = exp3_learner_long.pull_arm()
        reward = exp3_long_env.round(pulled_arm)
        exp3_learner_long.update(pulled_arm, reward)

        # SWUCB1 window size = 0.5 * sqrt(T)
        pulled_arm = swucb_learner_w1_long.pull_arm()
        reward = swucb_w1_long_env.round(pulled_arm)
        swucb_learner_w1_long.update(pulled_arm, reward)

        # SWUCB1 window size = sqrt(T)  
        pulled_arm = swucb_learner_w2_long.pull_arm()        
        reward = swucb_w2_long_env.round(pulled_arm)
        swucb_learner_w2_long.update(pulled_arm, reward)

        # SWUCB1 window size = 2 * sqrt(T)
        pulled_arm = swucb_learner_w3_long.pull_arm()
        reward = swucb_w3_long_env.round(pulled_arm)
        swucb_learner_w3_long.update(pulled_arm, reward)

        # CUSUM UCB
        pulled_arm = cducb_learner_long.pull_arm()
        reward = cducb_long_env.round(pulled_arm)
        cducb_learner_long.update(pulled_arm, reward)

    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    exp3_rewards_per_experiment.append(exp3_learner.collected_rewards)
    swucb_w1_rewards_per_experiment.append(swucb_learner_w1.collected_rewards)
    swucb_w2_rewards_per_experiment.append(swucb_learner_w2.collected_rewards)
    swucb_w3_rewards_per_experiment.append(swucb_learner_w3.collected_rewards)
    cducb_rewards_per_experiment.append(cducb_learner.collected_rewards)

    ucb1_long_rewards_per_experiment.append(ucb1_learner_long.collected_rewards)
    exp3_long_rewards_per_experiment.append(exp3_learner_long.collected_rewards)
    swucb_w1_long_rewards_per_experiment.append(swucb_learner_w1_long.collected_rewards)
    swucb_w2_long_rewards_per_experiment.append(swucb_learner_w2_long.collected_rewards)
    swucb_w3_long_rewards_per_experiment.append(swucb_learner_w3_long.collected_rewards)
    cducb_long_rewards_per_experiment.append(cducb_learner_long.collected_rewards)

ucb1_rewards_per_experiment = np.array(ucb1_rewards_per_experiment)
exp3_rewards_per_experiment = np.array(exp3_rewards_per_experiment)
swucb_w1_rewards_per_experiment = np.array(swucb_w1_rewards_per_experiment)
swucb_w2_rewards_per_experiment = np.array(swucb_w2_rewards_per_experiment)
swucb_w3_rewards_per_experiment = np.array(swucb_w3_rewards_per_experiment)
cducb_rewards_per_experiment = np.array(cducb_rewards_per_experiment)

ucb1_long_rewards_per_experiment = np.array(ucb1_long_rewards_per_experiment)
exp3_long_rewards_per_experiment = np.array(exp3_long_rewards_per_experiment)
swucb_w1_long_rewards_per_experiment = np.array(swucb_w1_long_rewards_per_experiment)
swucb_w2_long_rewards_per_experiment = np.array(swucb_w2_long_rewards_per_experiment)
swucb_w3_long_rewards_per_experiment = np.array(swucb_w3_long_rewards_per_experiment)
cducb_long_rewards_per_experiment = np.array(cducb_long_rewards_per_experiment)

ucb1_regret = np.zeros(T)
exp3_regret = np.zeros(T)
swucb_w1_regret = np.zeros(T)
swucb_w2_regret = np.zeros(T)
swucb_w3_regret = np.zeros(T)
cducb_regret = np.zeros(T)

ucb1_long_regret = np.zeros(T)
exp3_long_regret = np.zeros(T)
swucb_w1_long_regret = np.zeros(T)
swucb_w2_long_regret = np.zeros(T)
swucb_w3_long_regret = np.zeros(T)
cducb_long_regret = np.zeros(T)

ucb1_instantaneous_regret = np.zeros(T)
exp3_instantaneous_regret = np.zeros(T)
swucb_w1_instantaneous_regret = np.zeros(T)
swucb_w2_instantaneous_regret = np.zeros(T)
swucb_w3_instantaneous_regret = np.zeros(T)
cducb_instantaneous_regret = np.zeros(T)

ucb1_long_instantaneous_regret = np.zeros(T)
exp3_long_instantaneous_regret = np.zeros(T)
swucb_w1_long_instantaneous_regret = np.zeros(T)
swucb_w2_long_instantaneous_regret = np.zeros(T)
swucb_w3_long_instantaneous_regret = np.zeros(T)
cducb_long_instantaneous_regret = np.zeros(T)

opt_per_phase = prb.max(axis=1)
opt_per_phase_long = prb_long.max(axis=1)

optimum_per_round = np.zeros(T)
optimum_per_round_long = np.zeros(T)

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]

    # Regret
    ucb1_regret[t_index] = np.mean(opt_per_phase[i] - ucb1_rewards_per_experiment, axis=0)[t_index]
    exp3_regret[t_index] = np.mean(opt_per_phase[i] - exp3_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_regret[t_index] = np.mean(opt_per_phase[i] - swucb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_regret[t_index] = np.mean(opt_per_phase[i] - cducb_rewards_per_experiment, axis=0)[t_index]

    # Instantaneous regret
    ucb1_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(ucb1_rewards_per_experiment, axis=0)[t_index]
    exp3_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(exp3_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w1_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w2_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(swucb_w3_rewards_per_experiment, axis=0)[t_index]
    cducb_instantaneous_regret[t_index] = opt_per_phase[i] - np.mean(cducb_rewards_per_experiment, axis=0)[t_index]


for i in range(n_phases_long):
    t_index = range(i * phases_len_long, (i + 1) * phases_len_long)
    optimum_per_round_long[t_index] = opt_per_phase_long[i]

    # Regret
    ucb1_long_regret[t_index] = np.mean(opt_per_phase_long[i] - ucb1_long_rewards_per_experiment, axis=0)[t_index]
    exp3_long_regret[t_index] = np.mean(opt_per_phase_long[i] - exp3_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_long_regret[t_index] = np.mean(opt_per_phase_long[i] - swucb_w1_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_long_regret[t_index] = np.mean(opt_per_phase_long[i] - swucb_w2_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_long_regret[t_index] = np.mean(opt_per_phase_long[i] - swucb_w3_long_rewards_per_experiment, axis=0)[t_index]
    cducb_long_regret[t_index] = np.mean(opt_per_phase_long[i] - cducb_long_rewards_per_experiment, axis=0)[t_index]

    # Instantaneous regret
    ucb1_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(ucb1_long_rewards_per_experiment, axis=0)[t_index]
    exp3_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(exp3_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w1_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(swucb_w1_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w2_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(swucb_w2_long_rewards_per_experiment, axis=0)[t_index]
    swucb_w3_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(swucb_w3_long_rewards_per_experiment, axis=0)[t_index]
    cducb_long_instantaneous_regret[t_index] = opt_per_phase_long[i] - np.mean(cducb_long_rewards_per_experiment, axis=0)[t_index]

ucb1_label = "Stationary UCB1"
exp3_label = "EXP3"
swucb_w1_label = r"$SW\ UCB1,\ window\ size=\frac{1}{2}\ \sqrt{T}$"
swucb_w2_label = r"$SW\ UCB1,\ window\ size=\sqrt{T}$"
swucb_w3_label = r"$SW\ UCB1,\ window\ size=2 \sqrt{T}$"
cducb_label = "CUSUM UCB1"
# %%
# Cumulative regret
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
fig.canvas.manager.set_window_title("Cumulative regret")

axes[0].set_title('Cumulative regret')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Regret')
axes[0].plot(np.cumsum(ucb1_regret), 'r', label=ucb1_label)
axes[0].plot(np.cumsum(exp3_regret), 'brown', label=exp3_label)
axes[0].plot(np.cumsum(swucb_w1_regret), 'b', label=swucb_w1_label)
axes[0].plot(np.cumsum(swucb_w2_regret), 'g', label=swucb_w2_label)
axes[0].plot(np.cumsum(swucb_w3_regret), 'y', label=swucb_w3_label)
axes[0].plot(np.cumsum(cducb_regret), 'm', label=cducb_label)
axes[0].legend(loc=0)

axes[1].set_title('Cumulative regret (more phases)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Regret')
axes[1].plot(np.cumsum(ucb1_long_regret), 'r', label=ucb1_label)
axes[1].plot(np.cumsum(exp3_long_regret), 'brown', label=exp3_label)
axes[1].plot(np.cumsum(swucb_w1_long_regret), 'b', label=swucb_w1_label)
axes[1].plot(np.cumsum(swucb_w2_long_regret), 'g', label=swucb_w2_label)
axes[1].plot(np.cumsum(swucb_w3_long_regret), 'y', label=swucb_w3_label)
axes[1].plot(np.cumsum(cducb_long_regret), 'm', label=cducb_label)
axes[1].legend(loc=0)

plt.show()
# %%
# Standard deviation of cumulative regret
stducb = [(np.cumsum(ucb1_regret))[:i].std() for i in range(1, T + 1)]
stdexp3 = [(np.cumsum(exp3_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w1 = [(np.cumsum(swucb_w1_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w2 = [(np.cumsum(swucb_w2_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w3 = [(np.cumsum(swucb_w3_regret))[:i].std() for i in range(1, T + 1)]
stdcducb = [(np.cumsum(cducb_regret))[:i].std() for i in range(1, T + 1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
fig.canvas.manager.set_window_title("Standard deviation of cumulative regret")

axes[0].set_title('Standard deviation of cumulative regret')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Regret')
axes[0].plot(stducb, 'r', label=ucb1_label)
axes[0].plot(stdexp3, 'brown', label=exp3_label)
axes[0].plot(stdswucb_w1, 'b', label=swucb_w1_label)
axes[0].plot(stdswucb_w2, 'g', label=swucb_w2_label)
axes[0].plot(stdswucb_w3, 'y', label=swucb_w3_label)
axes[0].plot(stdcducb, 'm', label=cducb_label)
axes[0].legend(loc=0)


stducb_long = [(np.cumsum(ucb1_long_regret))[:i].std() for i in range(1, T + 1)]
stdexp3_long = [(np.cumsum(exp3_long_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w1_long = [(np.cumsum(swucb_w1_long_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w2_long = [(np.cumsum(swucb_w2_long_regret))[:i].std() for i in range(1, T + 1)]
stdswucb_w3_long = [(np.cumsum(swucb_w3_long_regret))[:i].std() for i in range(1, T + 1)]
stdcducb_long = [(np.cumsum(cducb_long_regret))[:i].std() for i in range(1, T + 1)]

axes[1].set_title('Standard deviation of cumulative regret (more phases)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Regret')
axes[1].plot(stducb_long, 'r', label=ucb1_label)
axes[1].plot(stdexp3_long, 'brown', label=exp3_label)
axes[1].plot(stdswucb_w1_long, 'b', label=swucb_w1_label)
axes[1].plot(stdswucb_w2_long, 'g', label=swucb_w2_label)
axes[1].plot(stdswucb_w3_long, 'y', label=swucb_w3_label)
axes[1].plot(stdcducb_long, 'm', label=cducb_label)
axes[1].legend(loc=0)

plt.show()
# %%
# Cumulative reward
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
fig.canvas.manager.set_window_title("Cumulative reward")

axes[0].set_title('Cumulative reward')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Reward')
axes[0].plot(np.cumsum(np.mean(ucb1_rewards_per_experiment, axis=0)), 'r', label=ucb1_label)
axes[0].plot(np.cumsum(np.mean(exp3_rewards_per_experiment, axis=0)), 'brown', label=exp3_label)
axes[0].plot(np.cumsum(np.mean(swucb_w1_rewards_per_experiment, axis=0)), 'b', label=swucb_w1_label)
axes[0].plot(np.cumsum(np.mean(swucb_w2_rewards_per_experiment, axis=0)), 'g', label=swucb_w2_label)
axes[0].plot(np.cumsum(np.mean(swucb_w3_rewards_per_experiment, axis=0)), 'y', label=swucb_w3_label)
axes[0].plot(np.cumsum(np.mean(cducb_rewards_per_experiment, axis=0)), 'm', label=cducb_label)
axes[0].legend(loc=0)

axes[1].set_title('Cumulative reward (more phases)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Reward')
axes[1].plot(np.cumsum(np.mean(ucb1_long_rewards_per_experiment, axis=0)), 'r', label=ucb1_label)
axes[1].plot(np.cumsum(np.mean(exp3_long_rewards_per_experiment, axis=0)), 'brown', label=exp3_label)
axes[1].plot(np.cumsum(np.mean(swucb_w1_long_rewards_per_experiment, axis=0)), 'b', label=swucb_w1_label)
axes[1].plot(np.cumsum(np.mean(swucb_w2_long_rewards_per_experiment, axis=0)), 'g', label=swucb_w2_label)
axes[1].plot(np.cumsum(np.mean(swucb_w3_long_rewards_per_experiment, axis=0)), 'y', label=swucb_w3_label)
axes[1].plot(np.cumsum(np.mean(cducb_long_rewards_per_experiment, axis=0)), 'm', label=cducb_label)
axes[1].legend(loc=0)

plt.show()
# %%
# Instantaneous regret
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
fig.canvas.manager.set_window_title("Instantaneous regret")

axes[0].set_title('Instantaneous regret')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Regret')
axes[0].plot(np.cumsum(ucb1_instantaneous_regret), 'r', label=ucb1_label)
axes[0].plot(np.cumsum(exp3_instantaneous_regret), 'brown', label=exp3_label)
axes[0].plot(np.cumsum(swucb_w1_instantaneous_regret), 'b', label=swucb_w1_label)
axes[0].plot(np.cumsum(swucb_w2_instantaneous_regret), 'g', label=swucb_w2_label)
axes[0].plot(np.cumsum(swucb_w3_instantaneous_regret), 'y', label=swucb_w3_label)
axes[0].plot(np.cumsum(cducb_instantaneous_regret), 'm', label=cducb_label)
axes[0].legend(loc=0)

axes[1].set_title('Instantaneous regret (more phases)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Regret')
axes[1].plot(np.cumsum(ucb1_long_instantaneous_regret), 'r', label=ucb1_label)
axes[1].plot(np.cumsum(exp3_long_instantaneous_regret), 'brown', label=exp3_label)
axes[1].plot(np.cumsum(swucb_w1_long_instantaneous_regret), 'b', label=swucb_w1_label)
axes[1].plot(np.cumsum(swucb_w2_long_instantaneous_regret), 'g', label=swucb_w2_label)
axes[1].plot(np.cumsum(swucb_w3_long_instantaneous_regret), 'y', label=swucb_w3_label)
axes[1].plot(np.cumsum(cducb_long_instantaneous_regret), 'm', label=cducb_label)
axes[1].legend(loc=0)

plt.show()
# %%
# Instantaneous reward
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
fig.canvas.manager.set_window_title("Instantaneous reward")

axes[0].set_title('Instantaneous reward')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Reward')
axes[0].plot(np.mean(ucb1_rewards_per_experiment, axis=0), 'r', label=ucb1_label)
axes[0].plot(np.mean(exp3_rewards_per_experiment, axis=0), 'brown', label=exp3_label)
axes[0].plot(np.mean(swucb_w1_rewards_per_experiment, axis=0), 'b', label=swucb_w1_label)
axes[0].plot(np.mean(swucb_w2_rewards_per_experiment, axis=0), 'g', label=swucb_w2_label)
axes[0].plot(np.mean(swucb_w3_rewards_per_experiment, axis=0), 'y', label=swucb_w3_label)
axes[0].plot(np.mean(cducb_rewards_per_experiment, axis=0), 'm', label=cducb_label)
axes[0].plot(optimum_per_round, 'k--', label="Optimum")
axes[0].legend(loc=0)

axes[1].set_title('Instantaneous reward (more phases)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('Reward')
axes[1].plot(np.mean(ucb1_long_rewards_per_experiment, axis=0), 'r', label=ucb1_label)
axes[1].plot(np.mean(exp3_long_rewards_per_experiment, axis=0), 'brown', label=exp3_label)
axes[1].plot(np.mean(swucb_w1_long_rewards_per_experiment, axis=0), 'b', label=swucb_w1_label)
axes[1].plot(np.mean(swucb_w2_long_rewards_per_experiment, axis=0), 'g', label=swucb_w2_label)
axes[1].plot(np.mean(swucb_w3_long_rewards_per_experiment, axis=0), 'y', label=swucb_w3_label)
axes[1].plot(np.mean(cducb_long_rewards_per_experiment, axis=0), 'm', label=cducb_label)
axes[1].plot(optimum_per_round_long, 'k--', label="Optimum")
axes[1].legend(loc=0)

plt.show()