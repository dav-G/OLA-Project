import numpy as np

class Customer:
    def __init__(self, name, clicks_a, clicks_b, clicks_c, prob):
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