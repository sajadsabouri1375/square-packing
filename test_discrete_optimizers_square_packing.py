'''
    This test scripts provides test cases to test PSO optimizer library wrapper for square packing problem.
'''

import unittest
import numpy as np
from utils.utils import Utils
from discrete_optimizers import DiscretePsoOptimizer
from square_packing import SquarePacking

        
def square_packing_function(args):
    n_squares = int(args.shape[1]/3)
    n_particles = args.shape[0]
    
    results = []
    for solution_index in range(n_particles):
        square_packing_obj = SquarePacking(
            squares_properties=Utils.chunk_into_n(list(args[solution_index, :]), n_squares)
        )
        
        obj_f = square_packing_obj.calculate_bounding_area() + 6 * square_packing_obj.calculate_sum_overlapped_area_matrix()
        
        results.append(obj_f)

    return np.array(results)


class TestOptimizer(unittest.TestCase):

    def setUp(cls):

        lower_bounds = np.array([0, 0, - np.pi / 2])
        upper_bounds = np.array([10, 10, np.pi / 2])
        features_steps = np.array([0.001, 0.001, np.pi/8])
        cls._n_squares = 29
        
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
            n_iterations=1000,
            features_steps=np.tile(features_steps, cls._n_squares)
        )
        
    def test_square_packing_discrete(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        self._optimizer.animate_square_pack()
        
        self.assertLess(least_cost, self._n_squares * self._n_squares)
        
if __name__ == '__main__':
    unittest.main()