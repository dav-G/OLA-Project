import numpy as np

class BiddingEnvironmentCost():
    def __init__(self,bids,sigma,customer):
        self.bids=bids
        self.means=customer.cum_cost_clicks(bids)
        self.sigmas=np.ones(len(bids))*sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm],self.sigmas[pulled_arm])
    
class BiddingEnvironmentClicks():
    def __init__(self,bids,sigma,customer):
        self.bids=bids
        self.means=customer.num_clicks(bids)
        self.sigmas=np.ones(len(bids))*sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm],self.sigmas[pulled_arm])
    
class PricingEnvironment():
    def __init__(self, probabilities, n_prices=5):
        self.n_prices = n_prices
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

class Customer:
    def __init__(self, name, clicks_a, clicks_b, clicks_c,prob):
        self.name = name
        self.clicks_a = clicks_a
        self.clicks_b = clicks_b
        self.clicks_c = clicks_c
        self.prob = prob
    
    def num_clicks(self, bid):
        return (1.0 - np.exp(self.clicks_a * bid + self.clicks_b * bid**2)) * self.clicks_c
    
    def cum_cost_clicks(self, bid):
        C=self.clicks_c/50
        return 1.5*C*np.log10(1+bid/C)
    
    def compute_pricing(self,prices):
        ret=[]
        for i in range(0,5):
            ret.append((prices[i]-8.0)*self.prob[i])
        return ret
    
