import numpy as np

class PricingEnvironment():
    def __init__(self, probabilities, n_prices=5):
        self.n_prices = n_prices
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
    
