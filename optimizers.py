'''
    This is class which wraps pyswarms library, which is the professional python library to optimize problems with PSO algorithm.
    This class provides functionalities to define optimization problem, optimize, plot results, and store results of interest.
'''

import pyswarms as ps
import numpy as np
from datetime import datetime
from utils.utils import Utils
import os
import json
from pyswarms.utils.plotters import plot_cost_history, plot_contour
from pyswarms.utils.plotters.formatters import Mesher
from matplotlib import pyplot as plt
from IPython.display import Image


class PsoOptimizer:
    
    def __init__(self, **kwargs):
        
        self._optimization_id = kwargs.get('id', str(datetime.now().timestamp()).split('.')[0])
        self._problem_name = kwargs.get('problem_name', 'unknown_problem')
        self._saving_parent_directory = kwargs.get('saving_directory', Utils.get_path('outputs/optimization/'))
        self._saving_directory = os.path.join(self._saving_parent_directory, self._problem_name, self._optimization_id)
        self._store_results = kwargs.get('store_results', True)
        self._plot_cost_history = kwargs.get('plot_cost_history', True)
        self._animate_positions = kwargs.get('animate_positions', False)
        
        self._options:dict = kwargs.get(
            'optimization_options',
            {'c1': 0.5, 'c2': 0.3, 'w': 0.9}    
        )
        
        self._n_particles:int = kwargs.get('n_particles', 100)
        self._n_dimensions:int = kwargs.get('n_dimensions')
        self._bounds = kwargs.get('bounds', (np.zeros(self._n_dimensions), np.ones(self._n_dimensions)))
        self._n_iterations:int = kwargs.get('n_iterations', 100)
        self._objective_function:function = kwargs.get('objective_function')
                
    def set_objective_function(self, obj_f):
        
        self._objective_function = obj_f
        
    def define_optimizer(self):
        
        self._optimizer = ps.single.GlobalBestPSO(
            n_particles=self._n_particles,
            dimensions=self._n_dimensions,
            options=self._options,
            bounds=self._bounds
        )
        
    def store_results(self, cost, position):
        
        Utils.makedirs(self._saving_directory)
        
        with open(os.path.join(self._saving_directory, "best_result.json"), 'w') as f:
            json.dump(
                {
                    'best_position': list(position),
                    'best_cost': cost
                },
                f
            )
        
    def plot_cost_history(self):
        
        plot_cost_history(cost_history=self._optimizer.cost_history)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(
            os.path.join(self._saving_directory, "cost_history.jpg"),
            dpi=300
        )

    def animate(self):
        
        if self._n_dimensions == 2:
            m = Mesher(func=self._objective_function)
            
            animation = plot_contour(
                pos_history=self._optimizer.pos_history,
                mesher=m,
                mark=(0,0)
            )
            
            animation.save(
                os.path.join(self._saving_directory, 'positions_gif.gif'), 
                writer='imagemagick',
                fps=10
            )
            
            Image(url=os.path.join(self._saving_directory, 'positions_gif.gif'))
        
        else:
            
            raise Exception('2D positions evolution animation is only available for problems with 2 dimensions.')
            exit()
            
    def report_results(self, least_cost, best_position):
        
        if self._store_results:
            self.store_results(least_cost, best_position)
        
        if self._plot_cost_history:
            self.plot_cost_history()
            
        if self._animate_positions:
            self.animate()
            
    def run_optimizer(self):
        
        self.define_optimizer()

        least_cost, best_position = self._optimizer.optimize(
            self._objective_function,
            iters=self._n_iterations,
            n_processes=20
        )
        
        self.report_results(least_cost, best_position)
                
        return least_cost, best_position
        
        