from Customer import *
from Environment import Non_Stationary_Environment
from Learners import EXP3_Learner
from plotter import Plotter
import numpy as np
from math import sqrt

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
prices = np.array([10, 20, 30, 40, 50])
prb = np.array([
    [0.86, 0.7, 0.55, 0.27, 0.18],
    [0.4, 0.51, 0.66, 0.74, 0.81],
    [0.71, 0.58, 0.45, 0.2, 0.09],
    [0.35, 0.42, 0.67, 0.74, 0.8],
    [0.82, 0.7, 0.46, 0.27, 0.14]
])
prb = np.tile(prb, reps=(5, 1))

T = 365
n_experiments = 100
n_arms = len(prices)
n_phases = 3
phases_len = int(T / n_phases)

n_alg = 6

margin = (prices - 8)
clicks = int(c1.num_clicks(2))
cost = c1.click_cost(2)
rewards = (margin * prb - cost) * clicks

opt_per_phase = rewards.max(axis=1)
optimum_per_round = np.zeros(T)

rewards_experiment = [[] for i in range(n_alg)]

for e in range(0, n_experiments):
    env = [Non_Stationary_Environment(prb, T, n_phases) for _ in range(n_alg)]

    learner = [
        # EXP3 gamma = 0.01
        EXP3_Learner(n_arms, prices, 0.01, margin, clicks, cost),
        # EXP3 gamma = 0.05
        EXP3_Learner(n_arms, prices, 0.05, margin, clicks, cost),
        # EXP3 gamma = 0.1
        EXP3_Learner(n_arms, prices, 0.1, margin, clicks, cost),
        # EXP3 gamma = 0.2
        EXP3_Learner(n_arms, prices, 0.2, margin, clicks, cost),
        # EXP3 gamma = 0.3
        EXP3_Learner(n_arms, prices, 0.3, margin, clicks, cost),
        # EXP3 gamma = sqrt( log(n_arms) / n_arms)
        EXP3_Learner(n_arms, prices, np.sqrt(np.log(n_arms) /n_arms), margin, clicks, cost)
    ]

    for t in range(0, T):
        for i in range(n_alg):
            pulled_arm = learner[i].pull_arm()
            reward = env[i].round(pulled_arm, clicks)
            learner[i].update(pulled_arm, reward)
    [rewards_experiment[i].append(learner[i].collected_rewards) for i in range(n_alg)]
rewards_experiment = [np.array(rewards_experiment[i]) for i in range(n_alg)]

regret = [np.zeros(T) for _ in range(n_alg)]
std_regret = [np.zeros(T) for _ in range(n_alg)]

for i in range(n_phases):
    t_index = range(i * phases_len, (i + 1) * phases_len)
    optimum_per_round[t_index] = opt_per_phase[i]
    for alg in range(n_alg):
        # Regret
        regret[alg][t_index] = np.mean(opt_per_phase[i] - rewards_experiment[alg], axis=0)[t_index]

# Cumulative regret
cum_regret = [np.cumsum(regret[i]) for i in range(n_alg)]
# Standard deviation cumulative regret
cumstd_regret = [[(np.cumsum(regret[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]

# Plot results
dataset = np.array([[
    [cum_regret[i], cumstd_regret[i]]
] for i in range(n_alg)])

titles = ["Cumulative regret"]

exp3_1_label = r"$EXP3,\ gamma\ =\ 0.01$"
exp3_2_label = r"$EXP3,\ gamma\ =\ 0.05$"
exp3_3_label = r"$EXP3,\ gamma\ =\ 0.1$"
exp3_4_label = r"$EXP3,\ gamma\ =\ 0.2$"
exp3_5_label = r"$EXP3,\ gamma\ =\ 0.3$"
exp3_6_label = r"$EXP3,\ gamma\ =\ \sqrt{\frac{\log(5)}{5}}$"

labels = [exp3_1_label, exp3_2_label, exp3_3_label, exp3_4_label, exp3_5_label, exp3_6_label]

plotter = Plotter(dataset, optimum_per_round, titles, labels, T)
plotter.plots()