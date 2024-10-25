from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from scipy import signal, ndimage


class PreProcess(ABC):

    @abstractmethod
    def pre_process(self, data: np.array) -> np.array:
        return data

class AllPositive(PreProcess):
    def pre_process(self, data: np.array) -> np.array:
        return data - np.min(data)
    
class MedianFilter(PreProcess):
    def __init__(self, kernel_size=3):
        self._kernel_size = kernel_size
    
    def pre_process(self, data: np.array) -> np.array:
        return signal.medfilt2d(data, self._kernel_size)
    
class GaussianBlur(ABC):
    def __init__(self, sigma=3):
        self._sigma = sigma
    
    def pre_process(self, data: np.array) -> np.array:
        return ndimage.gaussian_filter(data, self._sigma)


class Resizer(PreProcess):
    def __init__(self, scaling_factor: float):
        self.scaling_factor = scaling_factor

    def pre_process(self, data: np.array) -> np.array:
        if self.scaling_factor > 1:
            order = 3 
        elif self.scaling_factor < 1:
            order = 1  
        else:
            return data 
        
        resized_data = ndimage.zoom(data, self.scaling_factor, order=order)
        return resized_data
    

class PreProcessPipe:
    def __init__(self, pre_processors: Optional[List[PreProcess]]=None):
        self._pre_processors = pre_processors or []
    
    def add_pre_processor(self, pre_processor: PreProcess):
        self._pre_processors.append(pre_processor)

    def pre_process_pipe(self, data: np.array) -> np.array:
        for pre_processor in self._pre_processors:
            data = pre_processor.pre_process(data)
        return data
    


