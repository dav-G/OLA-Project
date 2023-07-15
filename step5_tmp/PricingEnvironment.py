import numpy as np

class PricingEnvironment():
    def __init__(self, probabilities, n_prices=5):
        self.n_prices = n_prices
        self.probabilities = probabilities

    def round(self, pulled_arm, clicks):
        reward = np.random.binomial(clicks, self.probabilities[pulled_arm])
        return reward