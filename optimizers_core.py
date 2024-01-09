'''
    To implement custom PSO optimization evolution, we need to overwrite Pyswarms library core.
'''

import pyswarms.backend as P
from pyswarms.backend.topology import Star
import numpy as np
import multiprocessing as mp
from pyswarms.backend.operators import compute_pbest, compute_objective_function


class OptimizerCore:
    
    def __init__(self, **kwargs) -> None:
        
        self._n_particles:int = kwargs.get('n_particles', 100)
        self._n_dimensions:int = kwargs.get('n_dimensions')
        self._bounds = kwargs.get('bounds', (np.zeros(self._n_dimensions), np.ones(self._n_dimensions)))
        self._n_iterations:int = kwargs.get('n_iterations', 100)
        self._objective_function:function = kwargs.get('objective_function')
        self._options:dict = kwargs.get(
            'optimization_options',
            {'c1': 0.5, 'c2': 0.3, 'w': 0.9}    
        )
        self._features_steps = kwargs.get('features_steps')
        
        self._topology = Star()
        self._swarm = P.create_swarm(
            n_particles=self._n_particles,
            dimensions=self._n_dimensions,
            options=self._options,
            bounds=self._bounds
        ) 
        self._n_processes = kwargs.get('n_processes', None)
        
        self.cost_history = []
        self.pos_history = []
        self.best_pos_history = []
    
    def custom_round_function(self, value, step):
        
        int_part = value // step
        remainder_part = 1 if (value % step > step/2) else 0
        return (int_part + remainder_part) * step 
                
    def optimize(self):
        
        pool = None if self._n_processes is None else mp.Pool(self._n_processes)
        
        for i in range(self._n_iterations):
            
            # Part 1: Update personal best
            self._swarm.current_cost = compute_objective_function(self._swarm, self._objective_function, pool) # Compute current cost
            # self._swarm.pbest_cost = self._objective_function(self._swarm.pbest_pos)  # Compute personal best pos
            
            if i == 0:
                self._swarm.pbest_cost = self._swarm.current_cost
                
            self._swarm.pbest_pos, self._swarm.pbest_cost = P.compute_pbest(self._swarm) # Update and store

            # Part 2: Update global best
            # Note that gbest computation is dependent on your topology
            if np.min(self._swarm.pbest_cost) < self._swarm.best_cost:
                self._swarm.best_pos, self._swarm.best_cost = self._topology.compute_gbest(self._swarm)

            # Let's print our output
            if i%20==0 or i==self._n_iterations-1:
                print('Iteration: {} | self._swarm.best_cost: {:.4f}'.format(i+1, self._swarm.best_cost))

            # Part 3: Update position and velocity matrices
            # Note that position and velocity updates are dependent on your topology
            self._swarm.velocity = self._topology.compute_velocity(self._swarm)
            self._swarm.position = self._topology.compute_position(self._swarm)
            
            # Part 4: Map positions to discrete space with custom round funciton and features steps
            for dim in range(self._n_dimensions):
                self._swarm.position[:, dim] = np.array(list(map(self.custom_round_function, self._swarm.position[:, dim], np.ones((self._n_particles)) * self._features_steps[dim])))            
            
            self.cost_history.append(self._swarm.best_cost)
            self.pos_history.append(self._swarm.position)
            self.best_pos_history.append(self._swarm.best_pos)
            
        # Close Pool of Processes
        if self._n_processes is not None:
            pool.close()
        
        return self._swarm.best_cost, self._swarm.best_pos