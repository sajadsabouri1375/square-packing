'''
    Utils is a static class which contains handy small functions which are required in the majority of classes.
'''

from abc import abstractmethod
import os
import math

class Utils:
    
    @abstractmethod
    def get_path(path:str):
        '''
            Input: A directory within an os (linux/windows format)
            Output: The directory in the format of the machine operating system on which the program is running.
        '''
        
        # If path is defined in linux format
        address = os.path.join(*path.split('/'))

        # If windows address
        address = os.path.join(*address.split('\\'))

        root_project_dir = os.path.dirname(os.path.dirname(__file__))

        return os.path.join(root_project_dir, address)    
    
    @staticmethod
    def makedirs(path:str):
        
        os.makedirs(path, exist_ok=True)
        
    @staticmethod
    def chunk_into_n(lst, n):
        size = math.ceil(len(lst) / n)
        return list(
            map(lambda x: lst[x * size:x * size + size],
            list(range(n)))
        )