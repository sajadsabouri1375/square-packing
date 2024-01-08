'''
    This test scripts provides test cases to test SquarePacking class.
'''

import unittest
import numpy as np
from square_packing import SquarePacking


class TestSquare(unittest.TestCase):

    def setUp(cls):
                
        cls._square_pack = SquarePacking(
            squares_properties=[
                [0, 0, 0],
                [1, 0, np.pi/2]
            ]
        )

    def test_square(self):
        
        bounding_area = self._square_pack.calculate_bounding_area()
        
        self.assertAlmostEqual(bounding_area, 1, 3)
   
    def test_overlapped_area_matrix(self):
        
        overlapped_area_matrix = self._square_pack.calculate_overlapped_area_matrix()
        
        self.assertAlmostEqual(overlapped_area_matrix[0,0], 0, 3)
        self.assertAlmostEqual(overlapped_area_matrix[0,1], 1.0, 3)
        self.assertAlmostEqual(overlapped_area_matrix[1,0], 1.0, 3)
        self.assertAlmostEqual(overlapped_area_matrix[1,1], 0, 3)

    
if __name__ == '__main__':
    unittest.main()