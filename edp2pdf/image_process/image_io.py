import hyperspy.api as hs
import numpy as np


class LoadImage:
    """
    A class for loading image data from a file using HyperSpy.

    Attributes:
        _cache (dict): A class-level cache to store loaded data for reuse.
        _file_path (str): The path to the image file.
        _hs_data: The loaded HyperSpy data object.
    """

    _cache = {}

    def __init__(self, file_path: str):
        """
        Initializes the LoadImage object.

        Args:
            file_path (str): The path to the image file.

        Raises:
            TypeError: If `file_path` is not a string.
        """
        if not isinstance(file_path, str):
            raise TypeError("Input `file_path` must be a string.")

        self._file_path = file_path
        self._hs_data = None

    def _load_data(self) -> None:
        """
        Loads the image data from the file using HyperSpy.

        Raises:
            ValueError: If the file cannot be loaded or is invalid.
        """
        if self._file_path in self._cache:
            self._hs_data = self._cache[self._file_path]
        else:
            try:
                self._hs_data = hs.load(self._file_path)
                LoadImage._cache[self._file_path] = self._hs_data
            except Exception as e:
                raise ValueError(f"Error loading image: {e}")

    @property
    def data(self) -> np.ndarray:
        """
        Gets the image data as a numpy array.

        Returns:
            np.ndarray: The image data as a 2D numpy array.

        Raises:
            ValueError: If the loaded object does not have a "data" attribute.
        """
        self._load_data()
        try:
            return self._hs_data.data
        except AttributeError:
            raise ValueError('The loaded object does not have a "data" attribute.')


class WriteImage:
    """
    A class for saving image data to a file using HyperSpy.

    Attributes:
        _data (np.ndarray): The image data to save.
        _signal2d: The HyperSpy Signal2D object.
    """

    def __init__(self, data: np.ndarray):
        """
        Initializes the WriteImage object.

        Args:
            data (np.ndarray): The image data to save as a 2D numpy array.

        Raises:
            TypeError: If `data` is not a numpy array or is not 2D.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise TypeError("Input `data` must be a 2D numpy array.")

        self._data = data
        self._signal2d = None

    def _initialize_signal(self) -> None:
        """
        Initializes the HyperSpy Signal2D object from the image data.

        Raises:
            ValueError: If the Signal2D object cannot be initialized.
        """
        if self._signal2d is None:
            try:
                self._signal2d = hs.signals.Signal2D(self._data)
            except Exception as e:
                raise ValueError(f"Error initializing Signal2D object: {e}")

    def save(self, file_path: str) -> None:
        """
        Saves the image data to the specified file path.

        Args:
            file_path (str): The path where the image will be saved.

        Raises:
            TypeError: If `file_path` is not a string.
            ValueError: If the image cannot be saved.
        """
        if not isinstance(file_path, str):
            raise TypeError("Input `file_path` must be a string.")

        self._initialize_signal()

        try:
            self._signal2d.save(file_path)
        except Exception as e:
            raise ValueError(f"Error saving image to {file_path}: {e}")