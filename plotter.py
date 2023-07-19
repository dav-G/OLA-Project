from matplotlib import pyplot as plt

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
					alpha=0.15
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
			plt.legend(loc=2)
			plt.show()

	def subplots(self):
		for graph in range(self.dataset.shape[2]):
			fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), layout="constrained")
			fig.canvas.manager.set_window_title(self.titles[graph])

			axes[0].set_title(self.titles[graph])
			axes[0].set_xlabel('t')
			axes[0].set_ylabel(self.titles[graph].split()[-1].capitalize())
			for curve in range(self.dataset.shape[1]):
				axes[0].plot(
					self.x,
					self.dataset[0][curve][graph][0],
					label=self.labels[curve]
				)
				axes[0].fill_between(
					self.x,
					self.dataset[0][curve][graph][0] + self.dataset[0][curve][graph][1],
					self.dataset[0][curve][graph][0] - self.dataset[0][curve][graph][1],
					alpha=0.15
				)
			if (self.titles[graph].split()[0].lower() == "instantaneous"):
				if (self.titles[graph].split()[-1].lower() == "regret"):
					axes[0].plot(
						self.x,
						self.T * [0],
						'k--',
						label="Optimum"
					)
				else:
					axes[0].plot(
						self.x,
						self.optimal[0],
						'k--',
						label="Optimum"
					)
			axes[0].legend(loc=0)


			axes[1].set_title(self.titles[graph] + " (more phases)")
			axes[1].set_xlabel('t')
			axes[1].set_ylabel(self.titles[graph].split()[-1].capitalize())

			for curve in range(self.dataset.shape[1]):
				axes[1].plot(
					self.x,
					self.dataset[1][curve][graph][0],
					label=self.labels[curve]
				)
				axes[1].fill_between(
					self.x,
					self.dataset[1][curve][graph][0] + self.dataset[1][curve][graph][1],
					self.dataset[1][curve][graph][0] - self.dataset[1][curve][graph][1],
					alpha=0.15
				)
			if (self.titles[graph].split()[0].lower() == "instantaneous"):
				if (self.titles[graph].split()[-1].lower() == "regret"):
					axes[1].plot(
						self.x,
						self.T * [0],
						'k--',
						label="Optimum"
					)
				else:
					axes[1].plot(
						self.x,
						self.optimal[1],
						'k--',
						label="Optimum"
					)
			axes[1].legend(loc=0)
			
			plt.show()