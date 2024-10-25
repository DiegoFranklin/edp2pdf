import numpy as np
from typing import Tuple

class eDiffractionPattern:
    def __init__(self, data: np.ndarray, center: Tuple[int,int]):
        self.data = data
        self.center = tuple(tuple(map(int, center)))