'''
    This test scripts provides test cases to test PSO optimizer library wrapper for square packing problem.
'''

import unittest
import numpy as np
from utils.utils import Utils
from discrete_optimizers import DiscretePsoOptimizer
from square_packing import SquarePacking

        
def square_packing_function(args, **_kwargs):
    n_squares = int(args.shape[1]/3)
    n_particles = args.shape[0]
    
    overlap_penalty_power = 8

    results = []
    for solution_index in range(n_particles):
        square_packing_obj = SquarePacking(
            squares_properties=Utils.chunk_into_n(list(args[solution_index, :]), n_squares)
        )
        
        obj_f = square_packing_obj.calculate_bounding_area() * ( (square_packing_obj.calculate_sum_overlapped_area_matrix() + 1.05) ** overlap_penalty_power)
        
        results.append(obj_f)

    return np.array(results)


class TestOptimizer(unittest.TestCase):

    def setUp(cls):

        lower_bounds = np.array([0, 0, - np.pi / 2])
        upper_bounds = np.array([20, 20, np.pi / 2])
        features_steps = np.array([0.01, 0.01, np.pi/4])
        cls._n_squares = 15
        
        cls._optimizer = DiscretePsoOptimizer(
            problem_name=f'square_packing_discrete_n_{cls._n_squares}',
            saving_directory=Utils.get_path('outputs/optimization/'),
            store_results=True,
            plot_cost_history=True,
            objective_function=square_packing_function,
            n_dimensions=3*cls._n_squares,
            animate_positions=False,
            bounds=(np.tile(lower_bounds, cls._n_squares), np.tile(upper_bounds, cls._n_squares)),
            n_particles=3000,
            n_iterations=400,
            features_steps=np.tile(features_steps, cls._n_squares)
        )
        
    def test_trim_optimization_steps_removes_plateau_and_consecutive_duplicates(self):
        optimization_steps = [
            {'iteration': 0, 'best_cost': 100.0, 'best_position': [0]},
            {'iteration': 1, 'best_cost': 100.0, 'best_position': [1]},
            {'iteration': 2, 'best_cost': 50.0, 'best_position': [2]},
            {'iteration': 3, 'best_cost': 50.0, 'best_position': [3]},
            {'iteration': 4, 'best_cost': 20.0, 'best_position': [4]},
            {'iteration': 5, 'best_cost': 20.0, 'best_position': [5]},
            {'iteration': 6, 'best_cost': 20.0, 'best_position': [6]},
            {'iteration': 7, 'best_cost': 20.0, 'best_position': [7]},
        ]

        trimmed_steps = DiscretePsoOptimizer.trim_optimization_steps(optimization_steps)

        self.assertEqual(
            [step['iteration'] for step in trimmed_steps],
            [0, 2, 4]
        )
        self.assertEqual(
            [step['best_cost'] for step in trimmed_steps],
            [100.0, 50.0, 20.0]
        )

    def test_square_packing_discrete(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        self._optimizer.animate_square_pack()
        
        self.assertLess(least_cost, self._n_squares * self._n_squares)
        
if __name__ == '__main__':
    unittest.main()