from matplotlib import pyplot as plt
import numpy as np

class Plotter():
	"""
		dataset : 	matrix of tuples
					e.g.:
					UCB1 	[ [ucb_inst_regret, std_ucb_inst_regret],..., [ucb_inst_reward, std_ucb_inst_reward]]
					TS		[ [ts_inst_regret, std_ts_inst_regret],..., [ts_inst_reward, std_ts_inst_reward]]
	"""

	def __init__(self, dataset, optimal, titles, labels, T):
		self.dataset = dataset
		self.optimal = optimal
		self.titles = titles
		self.labels = labels
		self.T = T
		self.x = list(range(0, T))

	def plots(self):
		for graph in range(self.dataset.shape[1]):
			plt.figure(self.titles[graph])
			plt.title(self.titles[graph])
			plt.xlabel('t')
			plt.ylabel(self.titles[graph].split()[-1].capitalize())	
			
			for curve in range(self.dataset.shape[0]):
				plt.plot(
					self.x,
					self.dataset[curve][graph][0],
					label=self.labels[curve]
				)
				plt.fill_between(
					self.x,
					self.dataset[curve][graph][0] + self.dataset[curve][graph][1],
					self.dataset[curve][graph][0] - self.dataset[curve][graph][1],
					alpha=0.3
				)
			if (self.titles[graph].split()[0].lower() == "instantaneous"):
				if (self.titles[graph].split()[-1].lower() == "regret"):
					plt.plot(
						self.x,
						self.T * [0],
						'k--',
						label="Optimum"
					)
				else:
					plt.plot(
						self.x,
						self.optimal,
						'k--',
						label="Optimum"
					)
			plt.legend(loc=0)
			plt.show()

	def subplots(self):
		"""
			to do
		"""
		pass