from Customer import *
from Environment import Non_Stationary_Environment
from Learners import CDUCB_Learner
from plotter import Plotter
import numpy as np

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

M = 10
# eps = 0.4
h = np.log(T)
alpha = 0.1

rewards_experiment = [[] for _ in range(n_alg)]

for e in range(0, n_experiments):
    env = [Non_Stationary_Environment(prb, T, n_phases) for _ in range(n_alg)]

    learner = [
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
        CDUCB_Learner(n_arms, prices, M, 0.8, h, alpha, margin, clicks, cost)
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

cducb_e1_label = r"$CD\ UCB1,\ \epsilon=0.1 $"
cducb_e2_label = r"$CD\ UCB1,\ \epsilon=0.2 $"
cducb_e3_label = r"$CD\ UCB1,\ \epsilon=0.3 $"
cducb_e4_label = r"$CD\ UCB1,\ \epsilon=0.4 $"
cducb_e5_label = r"$CD\ UCB1,\ \epsilon=0.5 $"
cducb_e6_label = r"$CD\ UCB1,\ \epsilon=0.8 $"
labels = [cducb_e1_label, cducb_e2_label, cducb_e3_label, cducb_e4_label, cducb_e5_label, cducb_e6_label]

plotter = Plotter(dataset, optimum_per_round, titles, labels, T)
plotter.plots()