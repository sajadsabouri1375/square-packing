'''
    An implementation of PsoOptimizer class for discrete optimization problems.
'''

from optimizers import PsoOptimizer
from optimizers_core import OptimizerCore
import numpy as np
import imageio
from matplotlib import pyplot as plt
import matplotlib
from square_packing import SquarePacking
from utils.utils import Utils
from matplotlib.patches import Rectangle
import os
matplotlib.rcParams['font.family'] = ['Ubuntu', 'serif', 'ubuntu']


class DiscretePsoOptimizer(PsoOptimizer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._features_steps = kwargs.get('features_steps')
        self._animate_per_iteration = kwargs.get('animate_per_iteration', 8)
        
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
        
    def get_cmap(self):
        return plt.cm.get_cmap('tab20')

    def plot_packed_squares(self, axes, squares_properties, n_squares, cmap):
        
        to_be_removed_elements = []

        square_packing_obj = SquarePacking(
            squares_properties=Utils.chunk_into_n(list(squares_properties), n_squares)
        )
        bounding_area = square_packing_obj.calculate_bounding_area()
        overlapped_area = square_packing_obj.calculate_sum_overlapped_area_matrix()
        square_packing_obj.transform_squares()
        
        # Plot bounding area
        x, y = square_packing_obj.get_bounding_area_coordinates()
        element = axes.plot(
            x,
            y,
            color='gray',
            linewidth=10,
            alpha=0.5
        )
        to_be_removed_elements.append(element[0])
        
        # Plot squares
        for i, square in enumerate(square_packing_obj.get_squares()):
            coordinates = square.get_corners_coordinates()
            rect = Rectangle(
                tuple(coordinates[0, :]),
                1, 
                1, 
                angle=np.rad2deg(square.get_rotation()),
                alpha=0.9,
                facecolor=cmap(i)
            )
            element = axes.add_patch(rect)
            to_be_removed_elements.append(element)
            
        return to_be_removed_elements, bounding_area, overlapped_area

    def animate_square_pack(self):
    
        figure = plt.figure(figsize=(25.6, 14.4))
        axes = figure.gca()
        frames_names = []
        
        Utils.makedirs(os.path.join(self._saving_directory, 'gif'))
        for i, best_position in enumerate(self._optimizer.best_pos_history): 
            
            n_squares = int(len(best_position)/3)
            cmap = self.get_cmap()

            if i % self._animate_per_iteration == 0 or i == self._n_iterations - 1:
                to_be_removed_elements, bounding_area, overlapped_area = self.plot_packed_squares(axes, best_position, n_squares, cmap)
                
                axes.set_aspect('equal', adjustable='box')
                plt.title(f'N Squares = {n_squares} - Iteration {i}\nBounding Area: {round(bounding_area, 2)}\nOverlapped Area: {round(overlapped_area, 2)}', fontsize=25)
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                plt.xlim([-1, self._bounds[1][0] + 2])
                plt.ylim([-1, self._bounds[1][1] + 2])
                # axes.relim()
                # axes.autoscale_view()
                plt.xlabel('$\\alpha$', fontsize=30)
                plt.ylabel('$\\alpha$', fontsize=30)
                
                figure.canvas.draw()
                figure.canvas.flush_events()
                figure.savefig(os.path.join(self._saving_directory, 'gif', f'frame_{i}.jpg'), dpi=120)
                frames_names.append(f'frame_{i}.jpg')
                
                for element in to_be_removed_elements:
                    element.remove()
                
        images = []
        for frame in frames_names:
            images.append(imageio.imread(os.path.join(self._saving_directory, 'gif', frame)))
            
        imageio.mimsave(os.path.join(self._saving_directory, 'gif', 'animation.gif'), images, fps=15)
            
    def run_optimizer(self):
        
        self.define_optimizer()
        
        least_cost, best_position = self._optimizer.optimize()
                
        self.report_results(least_cost, best_position)
                
        return least_cost, best_position