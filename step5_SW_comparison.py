from Customer import *
from Environment import Non_Stationary_Environment
from Learners import SWUCB_Learner
from plotter import Plotter
import numpy as np
from math import sqrt

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
prices = np.array([10, 20, 30, 40, 50])
prb = np.array([
    [0.8, 0.70, 0.53, 0.28, 0.14],
    [0.4, 0.5, 0.65, 0.75, 0.8],
    [0.7, 0.6, 0.46, 0.19, 0.07]
])

T = 365
n_experiments = 30
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
        # SWUCB window size= sqrt(T)
        SWUCB_Learner(n_arms, prices, int(sqrt(T)), margin, clicks, cost),
        # SWUCB1 window size = 1.5 * sqrt(T)
        SWUCB_Learner(n_arms, prices, int(1.5 * sqrt(T)), margin, clicks, cost),
        # SWUCB1 window size = 2*sqrt(T)
        SWUCB_Learner(n_arms, prices, int(2 * sqrt(T)), margin, clicks, cost),
        # SWUCB1 window size = 2.5 * sqrt(T)
        SWUCB_Learner(n_arms, prices, int(2.5 * sqrt(T)), margin, clicks, cost),
        # SWUCB1 window size= 3*sqrt(T)
        SWUCB_Learner(n_arms, prices, int(3 * sqrt(T)), margin, clicks, cost),
        # SWUCB1 window size= 3.5*sqrt(T)
        SWUCB_Learner(n_arms, prices, int(3.5 * sqrt(T)), margin, clicks, cost)
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

swucb_w1_label = r"$SW\ UCB1,\ window\ size=\sqrt{T}$"
swucb_w2_label = r"$SW\ UCB1,\ window\ size=\frac{3}{2}\sqrt{T}$"
swucb_w3_label = r"$SW\ UCB1,\ window\ size=2\sqrt{T}$"
swucb_w4_label = r"$SW\ UCB1,\ window\ size=\frac{5}{2}\sqrt{T}$"
swucb_w5_label = r"$SW\ UCB1,\ window\ size=3\sqrt{T}$"
swucb_w6_label = r"$SW\ UCB1,\ window\ size=\frac{7}{2}\sqrt{T}$"

labels = [swucb_w1_label, swucb_w2_label, swucb_w3_label, swucb_w4_label, swucb_w5_label, swucb_w6_label]

plotter = Plotter(dataset, optimum_per_round, titles, labels, T)
plotter.plots()