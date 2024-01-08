'''
    An implementation of PsoOptimizer class for discrete optimization problems.
'''

from optimizers import PsoOptimizer
from optimizers_core import OptimizerCore
import numpy as np


class DiscretePsoOptimizer(PsoOptimizer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._features_steps = kwargs.get('features_steps')
    
    def define_optimizer(self):
        
        self._optimizer = OptimizerCore(
            n_particles=self._n_particles,
            n_iterations=self._n_iterations,
            n_dimensions=self._n_dimensions,
            options=self._options,
            bounds=self._bounds,
            features_steps=self._features_steps ,
            objective_function=self._objective_function,
            n_processes=20
        )

    def run_optimizer(self):
        
        self.define_optimizer()
        
        least_cost, best_position = self._optimizer.optimize()
                
        self.report_results(least_cost, best_position)
                
        return least_cost, best_position