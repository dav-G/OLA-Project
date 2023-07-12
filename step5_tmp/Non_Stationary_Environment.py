from PricingEnvironment import *
import numpy as np
from math import floor

class Non_Stationary_Environment(PricingEnvironment):
    def __init__(self, probabilities, horizon, n_phases):
        super().__init__(probabilities)
        self.t = 0
        self.horizon = horizon
        self.n_phases = n_phases
        self.phases_size = horizon / self.n_phases

    def round(self, pulled_arm):       
        current_phase = min(floor(self.t / self.phases_size), self.n_phases-1)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        self.t += 1        
        return reward