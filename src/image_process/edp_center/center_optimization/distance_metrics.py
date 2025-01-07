from scipy.spatial import distance
from typing import Callable
import numpy as np
from sklearn import metrics


def manhattan(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the Manhattan distance between two arrays, applying a symmetric mask.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array.
        mask (np.ndarray): A binary mask to apply symmetrically to both arrays.

    Returns:
        float: The normalized Manhattan distance between the masked arrays.

    Raises:
        ValueError: If the input arrays or mask have incompatible shapes.
    """
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("Input arrays and mask must have the same shape.")
    
    sym_mask = mask * np.flip(mask)
    return np.sum(np.abs(a * sym_mask - b * sym_mask)) / np.sum(sym_mask)


def mean_squared_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the mean squared error between two arrays, applying a symmetric mask.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array.
        mask (np.ndarray): A binary mask to apply symmetrically to both arrays.

    Returns:
        float: The normalized mean squared error between the masked arrays.

    Raises:
        ValueError: If the input arrays or mask have incompatible shapes.
    """
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("Input arrays and mask must have the same shape.")
    
    sym_mask = mask * np.flip(mask)
    return np.sum((a - b) ** 2 * sym_mask) / np.sum(sym_mask)


def cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the cosine distance between two arrays, applying a symmetric mask.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array.
        mask (np.ndarray): A binary mask to apply symmetrically to both arrays.

    Returns:
        float: The normalized cosine distance between the masked arrays.

    Raises:
        ValueError: If the input arrays or mask have incompatible shapes.
    """
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("Input arrays and mask must have the same shape.")
    
    sym_mask = mask * np.flip(mask)
    return distance.cosine((a * sym_mask).flatten(), (b * sym_mask).flatten()) / np.sum(sym_mask)


def jaccard_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the Jaccard distance between two arrays, applying a symmetric mask.

    Args:
        a (np.ndarray): The first input array.
        b (np.ndarray): The second input array.
        mask (np.ndarray): A binary mask to apply symmetrically to both arrays.

    Returns:
        float: The normalized Jaccard distance between the masked arrays.

    Raises:
        ValueError: If the input arrays or mask have incompatible shapes.
    """
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("Input arrays and mask must have the same shape.")
    
    a = 255 * (a / np.max(a))
    b = 255 * (b / np.max(b))
    a, b = a.astype('uint8'), b.astype('uint8')

    sym_mask = mask * np.flip(mask)
    return 1 - metrics.jaccard_score((a * sym_mask).flatten(), (b * sym_mask).flatten(), average='macro')


metrics_map = {
    'mean_squared_error': mean_squared_error,
    'manhattan': manhattan,
    'cosine': cosine,
    'jaccard_distance': jaccard_distance
}


def masked_metric_factory(metric_name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Factory function to retrieve a masked metric function by name.

    Args:
        metric_name (str): The name of the metric to retrieve. Supported metrics are:
                           'mean_squared_error', 'manhattan', 'cosine', 'jaccard_distance'.

    Returns:
        Callable[[np.ndarray, np.ndarray, np.ndarray], float]: The corresponding metric function.

    Raises:
        ValueError: If the provided metric name is not supported.
    """
    if metric_name not in metrics_map:
        raise ValueError(f'Unknown metric: {metric_name}. Supported metrics are: {list(metrics_map.keys())}')
    return metrics_map[metric_name]