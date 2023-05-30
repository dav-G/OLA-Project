from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPUCB_LO_Learner(Learner):
    def __init__(self,n_arms,arms,beta=110.):
        super().__init__(n_arms)
        self.arms=arms
        self.means=np.zeros(self.n_arms)
        self.sigmas=np.ones(self.n_arms)*10
        self.pulled_arms=[]
        alpha=10.0
        self.beta=beta
        kernel=C(1.0,(1e-3,1e3))*RBF(1.0,(1e-3,1e3))
        self.gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self,arm_idx, reward):
        super().update_observations(arm_idx,reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x=np.atleast_2d(self.pulled_arms).T
        y=self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas=np.maximum(self.sigmas,1e-2)

    def update(self,pulled_arm,reward):
        self.t+=1
        self.update_observations(pulled_arm,reward)
        self.update_model()

    def pull_arm(self):
        
        return np.argmin(self.means - self.sigmas * np.sqrt(self.beta))
    
class GPUCB_UP_Learner(Learner):
    def __init__(self,n_arms,arms,beta=110.):
        super().__init__(n_arms)
        self.arms=arms
        self.means=np.zeros(self.n_arms)
        self.sigmas=np.ones(self.n_arms)*10
        self.pulled_arms=[]
        alpha=10.0
        self.beta=beta
        kernel=C(1.0,(1e-3,1e3))*RBF(1.0,(1e-3,1e3))
        self.gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self,arm_idx, reward):
        super().update_observations(arm_idx,reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x=np.atleast_2d(self.pulled_arms).T
        y=self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas=np.maximum(self.sigmas,1e-2)

    def update(self,pulled_arm,reward):
        self.t+=1
        self.update_observations(pulled_arm,reward)
        self.update_model()

    def pull_arm(self):
        
        return np.argmax(self.means + self.sigmas * np.sqrt(self.beta))
    