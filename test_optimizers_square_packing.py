'''
    This test scripts provides test cases to test PSO optimizer library wrapper for square packing problem.
'''

import unittest
import numpy as np
from utils.utils import Utils
from optimizers import PsoOptimizer
from square_packing import SquarePacking
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def chunk_into_n(lst, n):
  size = math.ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )
  

def square_packing_function(args):
    n_squares = int(args.shape[1]/3)
    n_particles = args.shape[0]
    
    results = []
    for solution_index in range(n_particles):
        square_packing_obj = SquarePacking(
            squares_properties=chunk_into_n(list(args[solution_index, :]), n_squares)
        )
        
        obj_f = square_packing_obj.calculate_bounding_area() + 5 * square_packing_obj.calculate_sum_overlapped_area_matrix()
        
        results.append(obj_f)

    return np.array(results)


def plot_packed_squares(squares_properties):
    
    plt.close('all')
    
    n_squares = int(len(squares_properties)/3)

    square_packing_obj = SquarePacking(
        squares_properties=chunk_into_n(list(squares_properties), n_squares)
    )
    bounding_area = square_packing_obj.calculate_bounding_area()
    overlapped_area = square_packing_obj.calculate_sum_overlapped_area_matrix()
    
    plt.figure(figsize=(25.6, 14.4))
    
    # Plot bounding area
    x, y = square_packing_obj.get_bounding_area_coordinates()
    plt.plot(
        x,
        y,
        color='gray'
    )
    
    # Plot squares
    for square in square_packing_obj.get_squares():
        coordinates = square.get_corners_coordinates()
        rect = Rectangle(
            tuple(coordinates[0, :]),
            1, 
            1, 
            angle=np.rad2deg(square.get_rotation()),
            color='crimson',
            alpha=0.4,
            edgecolor='none'
        )
        plt.gca().add_patch(rect)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Bounding Area: {round(bounding_area, 5)}  |  Overlapped Area: {round(overlapped_area, 5)}')
    plt.show(block=True)
    

class TestOptimizer(unittest.TestCase):

    def setUp(cls):

        lower_bounds = np.array([0, 0, -1 * np.pi/2])
        upper_bounds = np.array([30, 30, np.pi/2])
        n_squares = 10
        
        cls._optimizer = PsoOptimizer(
            problem_name=f'square_packing_n_{n_squares}',
            saving_directory=Utils.get_path('outputs/optimization/'),
            store_results=True,
            plot_cost_history=True,
            objective_function=square_packing_function,
            n_dimensions=3*n_squares,
            animate_positions=False,
            bounds=(np.tile(lower_bounds, n_squares), np.tile(upper_bounds, n_squares)),
            n_particles=3000,
            n_iterations=500
        )

    def test_square_packing(self):
        
        least_cost, best_position = self._optimizer.run_optimizer()
        plot_packed_squares(best_position)
        self.assertLess(least_cost, 10 * 10)
        
if __name__ == '__main__':
    unittest.main()