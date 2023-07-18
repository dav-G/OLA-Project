import numpy as np

class Customer:

	def __init__(self, name, pricing_a, pricing_b, pricing_c, pricing_d, bidding_a, bidding_b, bidding_c, features = {}):
		self.name = name
		self.features = features

		self.pricing_a = pricing_a
		self.pricing_b = pricing_b
		self.pricing_c = pricing_c
		self.pricing_d = pricing_d

		self.bidding_a = bidding_a
		self.bidding_b = bidding_b
		self.bidding_c = bidding_c

	def num_clicks(self, bid):
		return (1.0 - np.exp(self.bidding_a * bid + self.bidding_b * bid**2)) * self.bidding_c

	def click_cost(self, bid):
		C = self.bidding_c/50
		return 1.5 * C * np.log10(1 + bid/C)

	def conversion_probability(self, price):
		return self.pricing_a + self.pricing_b/(1 + pow(price/self.pricing_c, self.pricing_d))