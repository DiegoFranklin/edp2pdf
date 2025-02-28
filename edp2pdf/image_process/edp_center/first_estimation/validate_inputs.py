import numpy as np

def validate_inputs(data: np.ndarray, mask: np.ndarray) -> None:
    """Validates the input data and mask."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input `data` must be a numpy array.")
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input `mask` must be a numpy array.")
    if data.shape != mask.shape:
        raise ValueError("Shapes of `data` and `mask` must match.")