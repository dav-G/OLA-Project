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
    
class Customer:
    def __init__(self, name, clicks_a, clicks_b, clicks_c):
        self.name = name
        self.clicks_a = clicks_a
        self.clicks_b = clicks_b
        self.clicks_c = clicks_c
    
    def num_clicks(self, bid):
        return (1.0 - np.exp(self.clicks_a * bid + self.clicks_b * bid**2)) * self.clicks_c
    
    def cum_cost_clicks(self, bid):
        C=self.clicks_c/50
        return 1.5*C*np.log10(1+bid/C)
    

   # def num_clicks_noise(self,bid):
    #    noise_std=5.0
     #   return self.num_clicks(self.bid) +np.random.normal(0,noise_std, size=self.num_clicks(self.bid).shape)
    
   # def cum_cost_clicks_noise(self,bid):
    #    noise_std=5.0
     #   return self.cum_cost_clicks(self.bid) +np.random.normal(0,noise_std, size=self.cum_cost_clicks(self.bid).shape)
                
                