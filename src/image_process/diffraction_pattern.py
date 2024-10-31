import numpy as np
from typing import Tuple

class eDiffractionPattern:
    def __init__(self, data: np.ndarray, center: Tuple[int,int], mask: np.ndarray):
        self.data = data
        self.center = tuple(tuple(map(int, center)))
        self._mask = mask
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask_data):
        self._mask = mask_data