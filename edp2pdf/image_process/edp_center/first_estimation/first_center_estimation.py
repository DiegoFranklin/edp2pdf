import numpy as np
from typing import Tuple, Union
from edp2pdf.image_process.edp_center.first_estimation.centroid import get_centroid
from edp2pdf.image_process.edp_center.first_estimation.autocorrelation import autocorrelation
from edp2pdf.image_process.edp_center.first_estimation.validate_inputs import validate_inputs

def first_center_estimation(data: np.ndarray,
                                   mask: np.ndarray = None,
                                   method: str = None) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Estimates the center of a diffraction pattern using the specified method.

    Args:
        data (np.ndarray): array representing the diffraction pattern
        mask (np.ndarray): the mask representing valid data points in data (1 for valid, 0 for invalid)
        method (str): the method to use to estimate the center

    Returns:
        Tuple[Union[int, float], Union[int, float]]: the estimated center of the diffraction pattern
    
    Raises:
        TypeError: if `data` or `mask` are not numpy arrays..
        ValueError: if shapes of `data` or `mask` do not match
        ValueError: if `method` is not a valid method
    """
    method_map = {'centroid': lambda: get_centroid(data),
                  'autocorrelation': lambda: autocorrelation(data, mask)}
    valid_methods = list(method_map.keys())


    if method is not None and method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Supported methods are {valid_methods}.")

    if method is None:
        method = 'centroid'
    if mask is None:
        mask = np.ones(data.shape)

    validate_inputs(data, mask)
    
    return method_map[method]()