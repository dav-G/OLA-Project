from UserClass import *
from PricingEnvironment import *
from UCB1_Learner import *
from TS_Learner import *
from matplotlib import pyplot as plt

n_arms = 5
c1 = UserClass(np.array([10, 20, 30, 40, 50]), np.array([0.8, 0.6, 0.4, 0.2, 0.1]))
print(c1.prices * c1.probabilities)
opt = max(c1.probabilities)#c1.prices[np.argmax(c1.prices * c1.probabilities)]
print(opt)

T = 365

n_experiments = 100
ucb1_rewards_per_experiment = []
ts_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = PricingEnvironment(c1.probabilities)
    ucb1_learner = UCB1_Learner(n_arms)
    ts_learner = TS_Learner(n_arms)
    for t in range(0,T):
        #UCB1
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

        #Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'g')
plt.legend(["UCB1", "TS"])
plt.show()
