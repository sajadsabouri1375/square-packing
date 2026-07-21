'''
    This test scripts provides test cases to test PSO optimizer library wrapper for square packing problem.
'''

import unittest
import json
import os
import numpy as np
from utils.utils import Utils
from discrete_optimizers import DiscretePsoOptimizer
from square_packing import SquarePacking

        
def square_packing_function(args, **_kwargs):
    n_squares = int(args.shape[1]/3)
    n_particles = args.shape[0]
    
    overlap_penalty_power = 16

    results = []
    for solution_index in range(n_particles):
        square_packing_obj = SquarePacking(
            squares_properties=Utils.chunk_into_n(list(args[solution_index, :]), n_squares)
        )
        
        obj_f = square_packing_obj.calculate_bounding_area() * ( (square_packing_obj.calculate_sum_overlapped_area_matrix() + 1) ** overlap_penalty_power)
        
        results.append(obj_f)

    return np.array(results)


def normalize_best_result_positions(best_result_path, features_steps):
    with open(best_result_path, 'r') as f:
        optimization_steps = json.load(f)

    for step in optimization_steps:
        normalized_position = SquarePacking.normalize_position(step['best_position'])
        step['best_position'] = SquarePacking.snap_position(normalized_position, features_steps)

    with open(best_result_path, 'w') as f:
        json.dump(optimization_steps, f)


class TestOptimizer(unittest.TestCase):

    def setUp(cls):

        lower_bounds = np.array([0, 0, - np.pi / 2])
        upper_bounds = np.array([30, 30, np.pi / 2])
        features_steps = np.array([0.01, 0.01, np.pi/4])
        cls._n_squares = 5
        
        cls._optimizer = DiscretePsoOptimizer(
            problem_name=f'square_packing_discrete_n_{cls._n_squares}',
            saving_directory=Utils.get_path('outputs/optimization/'),
            store_results=True,
            plot_cost_history=True,
            objective_function=square_packing_function,
            n_dimensions=3*cls._n_squares,
            animate_positions=False,
            bounds=(np.tile(lower_bounds, cls._n_squares), np.tile(upper_bounds, cls._n_squares)),
            n_particles=2000,
            n_iterations=200,
            features_steps=np.tile(features_steps, cls._n_squares)
        )
        
    def test_square_packing_discrete(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        self._optimizer.animate_square_pack()

        best_result_path = os.path.join(self._optimizer._saving_directory, 'best_result.json')
        normalize_best_result_positions(best_result_path, self._optimizer._features_steps)
        
        self.assertLess(least_cost, self._n_squares * self._n_squares)
        
if __name__ == '__main__':
    unittest.main()