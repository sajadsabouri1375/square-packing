'''
    This test scripts provides test cases to test discrete PSO optimizer library wrapper for a simple sphere objective function.
'''

import unittest
import numpy as np
from utils.utils import Utils
from discrete_optimizers import DiscretePsoOptimizer


def sphere_function(args):
    return np.square(args).sum(axis=1)


class TestOptimizer(unittest.TestCase):

    def setUp(cls):
                
        cls._optimizer = DiscretePsoOptimizer(
            problem_name='sphere_discrete',
            saving_directory=Utils.get_path('outputs/optimization/'),
            store_results=True,
            objective_function=sphere_function,
            n_dimensions=2,
            animate_positions=True,
            features_steps=np.array([0.25, 0.5])
        )

    def test_sphere_discrete(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        
        self.assertAlmostEqual(least_cost, 0.0, 3)
        
        
if __name__ == '__main__':
    unittest.main()