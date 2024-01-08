'''
    This test scripts provides test cases to test Square class.
'''

import unittest
import numpy as np
from square import Square


class TestSquare(unittest.TestCase):

    def setUp(cls):
                
        cls._square = Square(
            bottom_left_corner_coordinates=np.array([0,0]),
            rotation=np.pi/4,
            size=1
        )

    def test_square_corners_pi_4(self):
        
        self._square.set_rotation(np.pi/4)
        coordinates = self._square.get_corners_coordinates()
        
        self.assertAlmostEqual(coordinates[0,0], 0, 3)
        self.assertAlmostEqual(coordinates[0,1], 0, 3)
        self.assertAlmostEqual(coordinates[1,0], 0.707, 3)
        self.assertAlmostEqual(coordinates[1,1], 0.707, 3)
        self.assertAlmostEqual(coordinates[2,0], 0, 3)
        self.assertAlmostEqual(coordinates[2,1], 1.414, 3)
        self.assertAlmostEqual(coordinates[3,0], -0.707, 3)
        self.assertAlmostEqual(coordinates[3,1], 0.707, 3)
        
        
    def test_square_corners_pi_2(self):
        
        self._square.set_rotation(np.pi/2)
        coordinates = self._square.get_corners_coordinates()
        
        self.assertAlmostEqual(coordinates[0,0], 0, 3)
        self.assertAlmostEqual(coordinates[0,1], 0, 3)
        self.assertAlmostEqual(coordinates[1,0], 0, 3)
        self.assertAlmostEqual(coordinates[1,1], 1, 3)
        self.assertAlmostEqual(coordinates[2,0], -1, 3)
        self.assertAlmostEqual(coordinates[2,1], 1, 3)
        self.assertAlmostEqual(coordinates[3,0], -1, 3)
        self.assertAlmostEqual(coordinates[3,1], 0, 3)
        
    def test_square_corners_minus_pi_4(self):
        
        self._square.set_rotation(-1 * np.pi/4)
        coordinates = self._square.get_corners_coordinates()
        
        self.assertAlmostEqual(coordinates[0,0], 0, 3)
        self.assertAlmostEqual(coordinates[0,1], 0, 3)
        self.assertAlmostEqual(coordinates[1,0], 0.707, 3)
        self.assertAlmostEqual(coordinates[1,1], -0.707, 3)
        self.assertAlmostEqual(coordinates[2,0], 1.414, 3)
        self.assertAlmostEqual(coordinates[2,1], 0, 3)
        self.assertAlmostEqual(coordinates[3,0], 0.707, 3)
        self.assertAlmostEqual(coordinates[3,1], 0.707, 3)
        
    def test_square_corners_minus_pi_2(self):
        
        self._square.set_rotation(-1 * np.pi/2)
        coordinates = self._square.get_corners_coordinates()
        
        self.assertAlmostEqual(coordinates[0,0], 0, 3)
        self.assertAlmostEqual(coordinates[0,1], 0, 3)
        self.assertAlmostEqual(coordinates[1,0], 0, 3)
        self.assertAlmostEqual(coordinates[1,1], -1, 3)
        self.assertAlmostEqual(coordinates[2,0], 1, 3)
        self.assertAlmostEqual(coordinates[2,1], -1, 3)
        self.assertAlmostEqual(coordinates[3,0], 1, 3)
        self.assertAlmostEqual(coordinates[3,1], 0, 3)
        
if __name__ == '__main__':
    unittest.main()