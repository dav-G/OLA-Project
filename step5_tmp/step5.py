from Customer import *
from UCB1_Learner import *
from Non_Stationary_Environment import *
from SWUCB_Learner import *
from CDUCB_Learner import *
from plotResults import Plotter
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, log

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
prices = np.array([10, 20, 30, 40, 50])
prb = np.array([
    [0.95, 0.70, 0.53, 0.28, 0.14],
    [0.05, 0.25, 0.83, 0.95, 0.45],
    [0.75, 0.64, 0.06, 0.62, 0.82]
])

T = 365
n_experiments = 50
n_arms = len(prices)
n_phases = 3
phases_len = int(T / n_phases)

# UCB1, SWUCB1, CUSUM
n_alg = 3

margin = (prices - 8)
clicks = int(c1.num_clicks(2))
cost = c1.click_cost(2)
rewards = (margin * prb - cost) * clicks

opt_per_phase = rewards.max(axis=1)
optimum_per_round = np.zeros(T)

# window_size
M = 100
# exploration term
eps = 0.1
# detection threshold
h = 2 * np.log(T)
# scaling
alpha = 0.01

rewards_experiment = [[] for i in range(n_alg)]

for e in range(0, n_experiments):
    env = [Non_Stationary_Environment(prb, T, n_phases) for i in range(n_alg)]
    
    learner = [
        UCB1_Learner(n_arms, prices, margin, clicks, cost),
        SWUCB_Learner(n_arms, prices, int(0.5 * sqrt(T)), margin, clicks, cost),
        CDUCB_Learner(n_arms, prices, M, eps, h, alpha, margin, clicks, cost)
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
        # Standard deviation instantaneous regret
        std_regret[alg][t_index] = np.std(opt_per_phase[i] - rewards_experiment[alg], axis=0)[t_index]

# Instantaneous reward
inst_reward = [np.mean(rewards_experiment[i], axis=0) for i in range(n_alg)]
# Cumulative regret
cum_regret = [np.cumsum(regret[i]) for i in range(n_alg)]
# Cumulative reward
cum_reward = [np.cumsum(np.mean(rewards_experiment[i], axis=0)) for i in range(n_alg)]
# Standard deviation cumulative regret
cumstd_regret = [[(np.cumsum(regret[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]
# Standard deviation instantaneous reward
std_reward = [np.std(rewards_experiment[i], axis=0) for i in range(n_alg)]
# Standard deviation cumulative reward
cumstd_reward = [[(np.cumsum(rewards_experiment[alg]))[:i].std() for i in range(1, T + 1)] for alg in range(n_alg)]

# Plot results
dataset = np.array([[
    [regret[i], std_regret[i]],
    [inst_reward[i], std_reward[i]],
    [cum_regret[i], cumstd_regret[i]],
    [cum_reward[i], cumstd_reward[i]]
] for i in range(n_alg)])

titles = ["Instantaneous regret", "Instantaneous reward", "Cumulative regret", "Cumulative reward"]

ucb1_label = "Stationary UCB1"
swucb_label = r"$SW\ UCB1,\ window\ size=\frac{1}{2}\ T$"
cducb_label = "CUSUM UCB1"
labels = [ucb1_label, swucb_label, cducb_label]

plotter = Plotter(dataset, optimum_per_round, titles, labels, T)
plotter.plots()