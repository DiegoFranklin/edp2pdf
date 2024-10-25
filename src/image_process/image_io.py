import hyperspy.api as hs
import numpy as np

class LoadImage:
    _cache = {}

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._hs_data = None
    
    def _load_data(self) -> None:
        if self._file_path in self._cache:
            self._hs_data = self._cache[self._file_path] 
        else:
            try:
                self._hs_data = hs.load(self._file_path)
                LoadImage._cache[self._file_path] = self._hs_data 
            except Exception as e:
                raise ValueError(f'Error loading image: {e}')

    @property
    def data(self):
        self._load_data() 
        try:
            return self._hs_data.data
        except AttributeError:
            raise ValueError(f'The loaded object does not have a "data" attribute')
    

class WriteImage:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        
        if len(data.shape) != 2:
            raise TypeError("Data must be a two dimensional numpy array.")

        self._data = data
        self._signal2d = None
    
    def _initialize_signal(self) -> None:
        if self._signal2d is None:
            try:
                self._signal2d = hs.signals.Signal2D(self._data)
            except Exception as e:
                raise ValueError(f"Error initializing Signal2D object: {e}")
    
    def save(self, file_path: str) -> None:
        self._initialize_signal()
        
        try:
            self._signal2d.save(file_path)
        except Exception as e:
            raise ValueError(f"Error saving image to {file_path}: {e}")


