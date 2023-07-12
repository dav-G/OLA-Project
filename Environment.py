import numpy as np

class PricingEnvironment():
	def __init__(self, prices, customers):
		self.customers = customers
		self.prices = prices

	def round(self, pulled_arm):
		i = int(np.random.rand() * len(self.customers))

		return (np.random.binomial(1, self.customers[i].conversion_probability(self.prices[pulled_arm])),
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
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_arm]), self.sigmas[pulled_arm]/10)
		
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
		
		clicks = np.random.normal(self.customers[i].num_clicks(self.bids[pulled_bid]), self.sigmas[pulled_bid])
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_bid]), self.sigmas[pulled_bid])
		
		sold = (pulled_price - self.item_cost) * np.random.binomial(clicks, self.customers[i].conversion_probability(self.prices[pulled_price]))
		
		return (sold - clicks * click_cost,
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
		
		clicks = np.random.normal(self.customers[i].num_clicks(self.bids[pulled_bid]), self.sigmas[pulled_bid])
		click_cost = np.random.normal(self.customers[i].click_cost(self.bids[pulled_bid]), self.sigmas[pulled_bid])
		
		sold = (self.prices[pulled_price] - self.item_cost) * np.random.binomial(max(clicks,0), self.customers[i].conversion_probability(self.prices[pulled_price]))
		
		return sold - clicks * click_cost