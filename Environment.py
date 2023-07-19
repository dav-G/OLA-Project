import numpy as np
from math import floor

class PricingEnvironment():
	def __init__(self, prices, customers, prob):
		self.customers = customers
		self.prices = prices
		self.prob=prob

	def round(self, pulled_arm,clicks):
		i = int(np.random.rand() * len(self.customers))
		
		return (np.random.binomial(clicks, self.prob[pulled_arm]),
				self.customers[i].features)
	
	
	
class BiddingEnvironment():
	def __init__(self, best_price, bids, sigma, customers):
		self.customers = customers
		self.bids = bids
		
		self.sigmas = np.ones(len(bids)) * sigma
		
		self.best_price = best_price
		self.item_cost = 8

	def round(self, pulled_arm):
		i = int(np.random.rand() * len(self.customers))
		
		clicks = np.random.normal(self.customers[i].num_clicks(self.bids[pulled_arm]), self.sigmas[pulled_arm])
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_arm]), self.sigmas[pulled_arm]/20)
		
		conv_prob = self.customers[i].conversion_probability(self.best_price)
		
		return (clicks * (conv_prob * (self.best_price - self.item_cost) - click_cost),
				self.customers[i].features)

				
				
class Environment():
	def __init__(self, prices, bids, sigma, customers):
		self.customers = customers
		self.bids = bids
		self.prices = prices
		
		self.sigmas = np.ones(len(bids)) * sigma
		
		self.item_cost = 8

	def round(self, pulled_price, pulled_bid):
		i = int(np.random.rand() * len(self.customers))
		
		clicks = max(0,np.random.normal(self.customers[i].num_clicks(self.bids[pulled_bid]), self.sigmas[pulled_bid]))
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_bid]), self.sigmas[pulled_bid]/10)
		
		sold = np.random.binomial(clicks, self.customers[i].conversion_probability(self.prices[pulled_price]))
		
		return (sold,clicks,click_cost,
				self.customers[i].features)
				
				
class ContextEnvironment():
	def __init__(self, prices, bids, sigma, customers):
		self.customers = customers
		self.bids = bids
		self.prices = prices
		
		self.sigmas = np.ones(len(bids)) * sigma
		
		self.item_cost = 8

	def getFeatures(self):
		self.i = int(np.random.rand() * len(self.customers))
		return self.customers[self.i].features
				
	def getReward(self, pulled_price, pulled_bid):
		i = self.i
		
		clicks = max(0,np.random.normal(self.customers[i].num_clicks(self.bids[pulled_bid]), self.sigmas[pulled_bid]))
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_bid]), self.sigmas[pulled_bid]/10)
		
		sold = np.random.binomial(clicks, self.customers[i].conversion_probability(self.prices[pulled_price]))
		
		return (sold, clicks, click_cost,
				self.customers[i].features)

class Non_Stationary_Environment():
    def __init__(self, probabilities, horizon, n_phases):
        self.probabilities = probabilities
        self.t = 0
        self.horizon = horizon
        self.n_phases = n_phases
        self.phases_size = horizon / self.n_phases

    def round(self, pulled_arm, clicks):    
        current_phase = min(floor(self.t / self.phases_size), self.n_phases-1)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(clicks, p)
        self.t += 1        
        return reward
