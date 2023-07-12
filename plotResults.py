from matplotlib import pyplot as plt
import numpy as np

def plot(optimal, T, datasets, labels):

	x = list(range(0,T))
	
	# ISTANTANEOUS REGRET
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Istantaneous regret")
		plots.append(plt.plot(x, np.mean(optimal - np.array(datasets[i]),axis=0))[0])

	plt.axhline(y = 0, color = 'black', linestyle = '-')
	plt.legend(plots, labels)
	plt.show()
	
	# ISTANTANEOUS REWARD
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Istantaneous reward")
		plots.append(plt.plot(x, np.mean(np.array(datasets[i]),axis=0))[0])

	plt.axhline(y = optimal, color = 'black', linestyle = '-')
	plt.legend(plots, labels)
	plt.show()

	# CUMULATIVE REGRET
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Cumulative regret")
		plots.append(plt.plot(np.cumsum(np.mean(optimal - datasets[i], axis=0)))[0])
		# plt.plot(np.cumsum(np.mean(optimal-datasets[i], axis=0)) + 1.96 * np.std(datasets[i],axis=0),'--')
		# plt.plot(np.cumsum(np.mean(optimal-datasets[i], axis=0)) - 1.96 * np.std(datasets[i],axis=0),'--')

	plt.legend(plots, labels)
	plt.show()
	
	# CUMULATIVE REWARD
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Cumulative reward")
		plots.append(plt.plot(np.cumsum(np.mean(datasets[i], axis=0)))[0])
	
	plt.legend(plots, labels)
	plt.show()
	
	# STANDARD DEVIATION OF CUMULATIVE REGRET
	# DA SISTEMARE
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Standard deviation of cumulative regret")

		std = [(np.cumsum(np.mean(optimal - datasets[i],axis=0)))[:i].std() for i in range(1,T+1)]
		plots.append(plt.plot(std)[0])
		
	plt.legend(plots, labels)
	plt.show()
	
	
	



