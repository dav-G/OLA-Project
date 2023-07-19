from Customer import *
from Environment import Non_Stationary_Environment
from Learners import UCB1_Learner_ns as UCB1_Learner
from Learners import EXP3_Learner
from Learners import SWUCB_Learner
from Learners import CDUCB_Learner
from plotter import Plotter
import numpy as np
from math import sqrt
import json

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
prices = np.array([10, 20, 30, 40, 50])

# Get probabilities from a file .json
with open('probabilities.json', 'r') as file:
    probabilities = json.load(file)
prb = np.array(probabilities['prb_step5'][0])
prb_long = np.array(probabilities['prb_step6'][0])

T = 365
n_experiments = 10
n_arms = len(prices)

margin = (prices - 8)
clicks = int(c1.num_clicks(2))
cost = c1.click_cost(2)
rewards = (margin * prb - cost) * clicks

# UCB1, EXP3, SWUCB1, CUSUM
n_alg = 4

n_phases = 3
phases_len = int(T / n_phases)

# long: refers to setting with higher non-stationarity degree (more phases)
n_phases_long = 25
phases_len_long = int(T / n_phases_long)
prb_long = np.tile(prb_long, reps=(5, 1))

rewards_long = (margin * prb_long - cost) * clicks

opt_per_phase = rewards.max(axis=1)
opt_per_phase_long = rewards_long.max(axis=1)

optimum_per_round = np.zeros(T)
optimum_per_round_long = np.zeros(T)

# window_size
M = 10
# exploration term
eps = 0.2
# detection threshold
h = 10 * np.log(T)
# scaling
alpha = 0.1

rewards_experiment = [[] for _ in range(n_alg)]
rewards_experiment_long = [[] for _ in range(n_alg)]

for e in range(0, n_experiments):
    env = [Non_Stationary_Environment(prb, T, n_phases) for _ in range(n_alg)]
    env_long = [Non_Stationary_Environment(prb_long, T, n_phases_long) for _ in range(n_alg)]

    learner = [
        UCB1_Learner(n_arms, prices, margin, clicks, cost),
        EXP3_Learner(n_arms, prices, margin, clicks, cost),
        SWUCB_Learner(n_arms, prices, int(7/2 * sqrt(T)), margin, clicks, cost),
        CDUCB_Learner(n_arms, prices, M, eps, h, alpha, margin, clicks, cost)
    ]
    learner_long = [
        UCB1_Learner(n_arms, prices, margin, clicks, cost),
        EXP3_Learner(n_arms, prices, margin, clicks, cost),
        SWUCB_Learner(n_arms, prices, int(7/2 * sqrt(T)), margin, clicks, cost),
        CDUCB_Learner(n_arms, prices, M, eps, h, alpha, margin, clicks, cost)
    ]

    for t in range(0, T):
        for i in range(n_alg):
            pulled_arm = learner[i].pull_arm()
            reward = env[i].round(pulled_arm, clicks)
            learner[i].update(pulled_arm, reward)

            pulled_arm = learner_long[i].pull_arm()
            reward = env_long[i].round(pulled_arm, clicks)
            learner_long[i].update(pulled_arm, reward)

    [rewards_experiment[i].append(learner[i].collected_rewards) for i in range(n_alg)]
    [rewards_experiment_long[i].append(learner_long[i].collected_rewards) for i in range(n_alg)]

rewards_experiment = [np.array(rewards_experiment[i]) for i in range(n_alg)]
rewards_experiment_long = [np.array(rewards_experiment_long[i]) for i in range(n_alg)]

regret = [np.zeros(T) for _ in range(n_alg)]
regret_long = [np.zeros(T) for _ in range(n_alg)]
std_regret = [np.zeros(T) for _ in range(n_alg)]
std_regret_long = [np.zeros(T) for _ in range(n_alg)]

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]
    for alg in range(n_alg):
        # Regret
        regret[alg][t_index] = np.mean(opt_per_phase[i] - rewards_experiment[alg], axis=0)[t_index]
        # Standard deviation instantaneous regret
        std_regret[alg][t_index] = np.std(opt_per_phase[i] - rewards_experiment[alg], axis=0)[t_index]

for i in range(n_phases_long):
    t_index = range(i * phases_len_long, (i + 1) * phases_len_long)
    optimum_per_round_long[t_index] = opt_per_phase_long[i]
    for alg in range(n_alg):
        # Regret
        regret_long[alg][t_index] = np.mean(opt_per_phase_long[i] - rewards_experiment_long[alg], axis=0)[t_index]
        # Standard deviation instantaneous regret
        std_regret_long[alg][t_index] = np.std(opt_per_phase_long[i] - rewards_experiment_long[alg], axis=0)[t_index]

# Instantaneous reward
inst_reward = [np.mean(rewards_experiment[i], axis=0) for i in range(n_alg)]
inst_reward_long = [np.mean(rewards_experiment_long[i], axis=0) for i in range(n_alg)]
# Cumulative regret
cum_regret = [np.cumsum(regret[i]) for i in range(n_alg)]
cum_regret_long = [np.cumsum(regret_long[i]) for i in range(n_alg)]
# Cumulative reward
cum_reward = [np.cumsum(np.mean(rewards_experiment[i], axis=0)) for i in range(n_alg)]
cum_reward_long = [np.cumsum(np.mean(rewards_experiment_long[i], axis=0)) for i in range(n_alg)]
# Standard deviation cumulative regret
cumstd_regret = [[(np.cumsum(regret[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]
cumstd_regret_long = [[(np.cumsum(regret_long[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]
# Standard deviation instantaneous reward
std_reward = [np.std(rewards_experiment[i], axis=0) for i in range(n_alg)]
std_reward_long = [np.std(rewards_experiment_long[i], axis=0) for i in range(n_alg)]
# Standard deviation cumulative reward
cumstd_reward = [[(np.cumsum(rewards_experiment[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]
cumstd_reward_long = [[(np.cumsum(rewards_experiment_long[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]

# Plot results
dataset = np.array(
    [
            [[
                [regret[i], std_regret[i]],
                [inst_reward[i], std_reward[i]],
                [cum_regret[i], cumstd_regret[i]],
                [cum_reward[i], cumstd_reward[i]]
            ] for i in range(n_alg)],
            [[
                [regret_long[i], std_regret_long[i]],
                [inst_reward_long[i], std_reward_long[i]],
                [cum_regret_long[i], cumstd_regret_long[i]],
                [cum_reward_long[i], cumstd_reward_long[i]]
            ] for i in range(n_alg)]
            ])

titles = ["Instantaneous regret", "Instantaneous reward", "Cumulative regret", "Cumulative reward"]

ucb1_label = "Stationary UCB1"
exp3_label = "EXP3"
swucb_label = r"$SW\ UCB1,\ window\ size=\frac{7}{2}\ \sqrt{T}$"
cducb_label = "CUSUM UCB1"
labels = [ucb1_label, exp3_label, swucb_label, cducb_label]

plotter = Plotter(dataset, [optimum_per_round, optimum_per_round_long], titles, labels, T)
plotter.subplots()