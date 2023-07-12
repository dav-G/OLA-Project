import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C
		
from Customer import Customer
		
def sell_margin(price):
    return price - 8
	
# Return[0]: Array of best prices (for every customer)
# Return[1]: Array of best bids (for every customer)
# Return[2]: Array of rewards (for every customer)
def getOptimal(f_print = False):
	customers = []
	customers.append(Customer('C1', -0.0081, 0.97, 32, 3.8, -1.5, 0.1, 100))
	customers.append(Customer('C2', -0.079, 0.89, 46, 3.4, -0.5, -1.5, 80))
	customers.append(Customer('C3', -0.082, 0.79, 35, 3.3, -5, 0.3, 65))

	bids = np.linspace(0.0,2.0,100)
	prices = np.linspace(10.0,50.0,5)

	if f_print:
		x = np.atleast_2d(bids).T
		plt.figure(0)

		for C in customers:
			plt.plot(x, C.num_clicks(x), label=r'$' + C.name + '$')

		plt.xlabel('$bid$')
		plt.ylabel('$clicks(bid)$')
		plt.legend(loc='lower right')
		plt.show

	reward = 0
	best_prices = []
	best_bids = []
	rewards = []

	for C in customers:
		best_price = prices[ np.argmax( [C.conversion_probability(price) * sell_margin(price) for price in prices] ) ]
		best_prices.append(best_price)
		
		best_bid = bids[ np.argmax( [C.num_clicks(bid) * (C.conversion_probability(best_price) * sell_margin(best_price) - C.click_cost(bid)) for bid in bids] ) ]
		best_bids.append(best_bid)
		
		reward = C.num_clicks(best_bid) * (C.conversion_probability(best_price) * sell_margin(best_price) - C.click_cost(best_bid))
		rewards.append(reward)
		
		if f_print:
			print('CUSTOMER {}'.format(C.name))
			print('   Best price: {}'.format(best_price))
			print('   Best bid: {}'.format(best_bid))
		
	if f_print:
		print('Total reward: {}'.format(sum(rewards)))
		
	return (best_prices, best_bids, rewards)
		
if __name__ == "__main__":
	getOptimal(f_print = True)