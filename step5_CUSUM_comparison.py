from Customer import *
from Environment import Non_Stationary_Environment
from Learners import CDUCB_Learner
from plotter import Plotter
import numpy as np

c1 = Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100)
prices = np.array([10, 20, 30, 40, 50])
prb = np.array([
    [0.86, 0.7, 0.55, 0.27, 0.18],
    [0.4, 0.51, 0.66, 0.74, 0.81],
    [0.71, 0.58, 0.45, 0.2, 0.09]
])

T = 365
n_experiments = 100
n_arms = len(prices)
n_phases = 3
phases_len = int(T / n_phases)

n_alg = 24

margin = (prices - 8)
clicks = int(c1.num_clicks(2))
cost = c1.click_cost(2)
rewards = (margin * prb - cost) * clicks

opt_per_phase = rewards.max(axis=1)
optimum_per_round = np.zeros(T)

# CUSUM UCB1 parameters
M = 10
eps = 0.1
h = 2 * np.log(T)
alpha = 0.1

rewards_experiment = [[] for _ in range(n_alg)]

for e in range(0, n_experiments):
    env = [Non_Stationary_Environment(prb, T, n_phases) for _ in range(n_alg)]

    learner = [
        # CDUCB1 with M = 10
        CDUCB_Learner(n_arms, prices, 10, eps, h, alpha, margin, clicks, cost),
        # CDUCB1 with M = 20
        CDUCB_Learner(n_arms, prices, 20, eps, h, alpha, margin, clicks, cost),
        # CDUCB1 with M = 30
        CDUCB_Learner(n_arms, prices, 30, eps, h, alpha, margin, clicks, cost),
        # CDUCB1 with M = 50
        CDUCB_Learner(n_arms, prices, 50, eps, h, alpha, margin, clicks, cost),
        # CDUCB1 with M = 80
        CDUCB_Learner(n_arms, prices, 80, eps, h, alpha, margin, clicks, cost),
        # CDUCB1 with M = 100
        CDUCB_Learner(n_arms, prices, 100, eps, h, alpha, margin, clicks, cost),

        # CDUCB1 with epsilon = 0.1
        CDUCB_Learner(n_arms, prices, M, 0.1, h, alpha, margin, clicks, cost),
        # CDUCB1 with epsilon = 0.2
        CDUCB_Learner(n_arms, prices, M, 0.2, h, alpha, margin, clicks, cost),
        # CDUCB1 with epsilon = 0.3
        CDUCB_Learner(n_arms, prices, M, 0.3, h, alpha, margin, clicks, cost),
        # CDUCB1 with epsilon = 0.4
        CDUCB_Learner(n_arms, prices, M, 0.4, h, alpha, margin, clicks, cost),
        # CDUCB1 with epsilon = 0.5
        CDUCB_Learner(n_arms, prices, M, 0.5, h, alpha, margin, clicks, cost),
        # CDUCB1 with epsilon = 0.8
        CDUCB_Learner(n_arms, prices, M, 0.8, h, alpha, margin, clicks, cost),

        # CDUCB1 with h = logT
        CDUCB_Learner(n_arms, prices, M, eps, np.log(T), alpha, margin, clicks, cost),
        # CDUCB1 with h = 2logT
        CDUCB_Learner(n_arms, prices, M, eps, 2 * np.log(T), alpha, margin, clicks, cost),
        # CDUCB1 with h = 4logT
        CDUCB_Learner(n_arms, prices, M, eps, 4 * np.log(T), alpha, margin, clicks, cost),
        # CDUCB1 with h = 5logT
        CDUCB_Learner(n_arms, prices, M, eps, 5 * np.log(T), alpha, margin, clicks, cost),
        # CDUCB1 with h = 8logT
        CDUCB_Learner(n_arms, prices, M, eps, 8 * np.log(T), alpha, margin, clicks, cost),
        # CDUCB1 with h = 10logT
        CDUCB_Learner(n_arms, prices, M, eps, 10 * np.log(T), alpha, margin, clicks, cost),

        # CDUCB1 with alpha = 0.05
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.05, margin, clicks, cost),
        # CDUCB1 with alpha = 0.1
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.1, margin, clicks, cost),
        # CDUCB1 with alpha = 0.2
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.2, margin, clicks, cost),
        # CDUCB1 with alpha = 0.3
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.3, margin, clicks, cost),
        # CDUCB1 with alpha = 0.5
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.5, margin, clicks, cost),
        # CDUCB1 with alpha = 0.7
        CDUCB_Learner(n_arms, prices, M, eps, h, 0.7, margin, clicks, cost),
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
cducb_M1_label = r"$CD\ UCB1,\ M=10 $"
cducb_M2_label = r"$CD\ UCB1,\ M=20 $"
cducb_M3_label = r"$CD\ UCB1,\ M=30 $"
cducb_M4_label = r"$CD\ UCB1,\ M=50 $"
cducb_M5_label = r"$CD\ UCB1,\ M=80 $"
cducb_M6_label = r"$CD\ UCB1,\ M=100 $"

cducb_e1_label = r"$CD\ UCB1,\ \epsilon=0.1 $"
cducb_e2_label = r"$CD\ UCB1,\ \epsilon=0.2 $"
cducb_e3_label = r"$CD\ UCB1,\ \epsilon=0.3 $"
cducb_e4_label = r"$CD\ UCB1,\ \epsilon=0.4 $"
cducb_e5_label = r"$CD\ UCB1,\ \epsilon=0.5 $"
cducb_e6_label = r"$CD\ UCB1,\ \epsilon=0.8 $"

cducb_h1_label = r"$CD\ UCB1,\ h=\log{T} $"
cducb_h2_label = r"$CD\ UCB1,\ h=2\log{T} $"
cducb_h3_label = r"$CD\ UCB1,\ h=4\log{T} $"
cducb_h4_label = r"$CD\ UCB1,\ h=5\log{T} $"
cducb_h5_label = r"$CD\ UCB1,\ h=8\log{T} $"
cducb_h6_label = r"$CD\ UCB1,\ h=10\log{T} $"

cducb_a1_label = r"$CD\ UCB1,\ \alpha=0.05 $"
cducb_a2_label = r"$CD\ UCB1,\ \alpha=0.1 $"
cducb_a3_label = r"$CD\ UCB1,\ \alpha=0.2 $"
cducb_a4_label = r"$CD\ UCB1,\ \alpha=0.3 $"
cducb_a5_label = r"$CD\ UCB1,\ \alpha=0.5 $"
cducb_a6_label = r"$CD\ UCB1,\ \alpha=0.7 $"


labels = [
    [cducb_M1_label, cducb_M2_label, cducb_M3_label, cducb_M4_label, cducb_M5_label, cducb_M6_label],
    [cducb_e1_label, cducb_e2_label, cducb_e3_label, cducb_e4_label, cducb_e5_label, cducb_e6_label],
    [cducb_h1_label, cducb_h2_label, cducb_h3_label, cducb_h4_label, cducb_h5_label, cducb_h6_label],
    [cducb_a1_label, cducb_a2_label, cducb_a3_label, cducb_a4_label, cducb_a5_label, cducb_a6_label]
]

for graph in range(4):
    dataset = np.array([[
        [cum_regret[graph*6+i], cumstd_regret[graph*6+i]]
    ] for i in range(6)])

    titles = ["Cumulative regret"]
    plotter = Plotter(dataset, optimum_per_round, titles, labels[graph], T)
    plotter.plots()