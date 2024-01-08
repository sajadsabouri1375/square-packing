'''
    Square class includes all required methods and properties to generate a square polygon and define its behaviors.
    Each square would be defined with its bottom left corner coordinates, and rotation magnitude in radians. 
'''

import numpy as np


class Square:
    
    def __init__(self, **kwargs):
        
        self._bottom_left_corner:np.array = kwargs.get('bottom_left_corner_coordinates')
        self._rotation:float = kwargs.get('rotation')
        self._size = kwargs.get('size', 1)
        self._diameter = np.sqrt(2 * np.square(self._size))
        self.calculate_corners_coordinates()
    
    def get_coordinates_tuple(self):
        return tuple(map(tuple, self._coordinates))
    
    def get_rotation(self):
        return self._rotation
    
    def set_rotation(self, new_rotation):
        self._rotation = new_rotation
        self.calculate_corners_coordinates()
        
    def calculate_corners_coordinates(self):
        
        self._bottom_right_corner = self._bottom_left_corner + self._size * np.array([np.cos(self._rotation), np.sin(self._rotation)]).reshape(self._bottom_left_corner.shape)
        self._top_right_corner = self._bottom_left_corner + self._diameter * np.array([np.cos(self._rotation + np.pi/4), np.sin(self._rotation + np.pi/4)]).reshape(self._bottom_left_corner.shape)
        self._top_left_corner = self._bottom_left_corner + self._size * np.array([-1 * np.sin(self._rotation), np.cos(self._rotation)]).reshape(self._bottom_left_corner.shape)
        
        self._coordinates = np.array(
            [
                self._bottom_left_corner,
                self._bottom_right_corner,
                self._top_right_corner,
                self._top_left_corner
            ]
        ).reshape(-1, 2)
        
    def get_corners_coordinates(self):

        return self._coordinates
    
    def get_min_x(self):
        
        return self._coordinates[:, 0].min()
    
    def get_max_x(self):
        
        return self._coordinates[:, 0].max()
    
    def get_min_y(self):
        
        return self._coordinates[:, 1].min()
    
    def get_max_y(self):
        
        return self._coordinates[:, 1].max()