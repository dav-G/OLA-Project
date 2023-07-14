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
	plt.fill_between(x,np.mean(optimal - np.array(datasets[0]),axis=0)+np.std(optimal-np.array(datasets[0]),axis=0), np.mean(optimal - np.array(datasets[0]),axis=0)-np.std(optimal-np.array(datasets[0]),axis=0),alpha=0.3)
	plt.fill_between(x,np.mean(optimal - np.array(datasets[1]),axis=0)+np.std(optimal-np.array(datasets[1]),axis=0), np.mean(optimal - np.array(datasets[1]),axis=0)-np.std(optimal-np.array(datasets[1]),axis=0),alpha=0.3)
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

	plt.fill_between(x,np.mean( np.array(datasets[0]),axis=0)+np.std(np.array(datasets[0]),axis=0), np.mean( np.array(datasets[0]),axis=0)-np.std(np.array(datasets[0]),axis=0),alpha=0.5)
	plt.fill_between(x,np.mean( np.array(datasets[1]),axis=0)+np.std(np.array(datasets[1]),axis=0), np.mean( np.array(datasets[1]),axis=0)-np.std(np.array(datasets[1]),axis=0),alpha=0.5)
	plt.axhline(y = optimal, color = 'black', linestyle = '-')
	plt.legend(plots, labels)
	plt.show()

	# CUMULATIVE REGRET
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Cumulative regret")
		regret=optimal - datasets[i]
		cum_regret=np.cumsum(regret,axis=1)
		mean=np.mean(cum_regret,axis=0)
		std=np.std(cum_regret,axis=0)
		plots.append(plt.plot(mean)[0])
		plt.fill_between(x,mean+std, mean-std,alpha=0.5)

	plt.legend(plots, labels)
	plt.show()
	
	# CUMULATIVE REWARD
	plots = []
	for i in range(len(datasets)):
		plt.figure(0)
		plt.xlabel("t")
		plt.ylabel("Cumulative reward")
		cum_reward=np.cumsum(datasets[i],axis=1)
		mean=np.mean(cum_reward,axis=0)
		std=np.std(cum_reward,axis=0)
		plots.append(plt.plot(mean)[0])
		plt.fill_between(x,mean+std, mean-std,alpha=0.5)
	plt.legend(plots, labels)
	plt.show()
	

	
	



