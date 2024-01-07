'''
    This test scripts provides test cases to test PSO optimizer library wrapper.
'''

import unittest
import numpy as np
from utils.utils import Utils
from optimizers import PsoOptimizer


def sphere_function(args):
    return np.square(args).sum(axis=1)


class TestOptimizer(unittest.TestCase):

    def setUp(cls):
                
        cls._optimizer = PsoOptimizer(
            problem_name='sphere',
            saving_directory=Utils.get_path('outputs/optimization/'),
            store_results=True,
            objective_function=sphere_function,
            n_dimensions=2,
            animate_positions=True
        )

    def test_sphere(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        
        self.assertAlmostEqual(least_cost, 0.0, 3)
        
        
if __name__ == '__main__':
    unittest.main()