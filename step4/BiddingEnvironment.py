import numpy as np

class BiddingEnvironmentCost():
    def __init__(self,bids,sigma,customers):
        self.bids=bids
        self.sigmas=np.ones(len(bids))*sigma
		self.customers = customers

    def round(self, pulled_arm):
		i = int(np.random.rand()*3)
        return (np.random.normal(self.customers[i].cum_cost_clicks[pulled_arm],self.sigmas[pulled_arm]), self.customers.features)
    
class BiddingEnvironmentClicks():
    def __init__(self,bids,sigma,customers):
        self.bids=bids
        self.sigmas=np.ones(len(bids))*sigma
		self.customers = customers

    def round(self, pulled_arm):
		i = int(np.random.rand()*3)
        return (np.random.normal(self.customers[i].num_clicks[pulled_arm],self.sigmas[pulled_arm]), self.customers.features)
    
class Customer:
    def __init__(self, name, clicks_a, clicks_b, clicks_c, features):
        self.name = name
		self.features = features
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
                
                