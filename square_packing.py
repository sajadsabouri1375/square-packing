'''
    This class includes all required methods and properties to calculate bounding square area,
    volume of overlaps, and other required metrics for a given state with n squares.
'''

from square import Square
import numpy as np
from shapely import Polygon


class SquarePacking:
    
    def __init__(self, **kwargs):
        
        self.set_squares_properties(kwargs.get('squares_properties'))
        
        self._bounding_square_area = None
        self._overlapped_area_matrix = None
    
    def set_squares_properties(self, new_squares_properties):
        
        self._squares_properties = new_squares_properties
        
        self._squares = [
            Square(
                bottom_left_corner_coordinates=np.array([square_property[0:2]]),
                rotation=square_property[2],
                size=1
            ) 
            for square_property in self._squares_properties
        ]
        
        self._squares_polygons = [
            Polygon(square.get_coordinates_tuple())
            for square in self._squares
        ]
        
    def calculate_bounding_area(self):
        min_x_s = min([square.get_min_x() for square in self._squares])
        max_x_s = max([square.get_max_x() for square in self._squares])
        min_y_s = min([square.get_min_y() for square in self._squares])
        max_y_s = max([square.get_max_y() for square in self._squares])
        
        x_delta = max_x_s - min_x_s
        y_delta = max_y_s - min_y_s
        
        self._bounding_square_area = np.square(max([x_delta, y_delta]))
        return self._bounding_square_area
    
    def get_bounding_area(self):
        return self._bounding_square_area
    
    def calculate_overlapped_area_matrix(self):
        n_squares = len(self._squares)
        self._overlapped_area_matrix = np.zeros((n_squares, n_squares))
        
        for i in range(len(self._squares)):
            
            for j in range(i+1, len(self._squares)):
                
                overlapped_area = self._squares_polygons[i].intersection(self._squares_polygons[j]).area
                self._overlapped_area_matrix[i, j] = overlapped_area
                self._overlapped_area_matrix[j, i] = overlapped_area
    
        return self._overlapped_area_matrix
    
    def get_overlapped_area_matrix(self):
        return self._overlapped_area_matrix
    
    def calculate_sum_overlapped_area_matrix(self):
        return self._overlapped_area_matrix.sum()