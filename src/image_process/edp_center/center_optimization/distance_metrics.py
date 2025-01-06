from scipy.spatial import distance
from typing import Callable
import numpy as np
from sklearn import metrics

def manhattan(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sym_mask = mask * np.flip(mask)
    return np.sum(np.abs(a * sym_mask - b * sym_mask)) / np.sum(sym_mask)

def mean_squared_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sym_mask = mask * np.flip(mask)
    return np.sum((a - b) ** 2 * sym_mask) / np.sum(sym_mask)

def cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sym_mask = mask * np.flip(mask)
    return distance.cosine((a * sym_mask).flatten(), (b * sym_mask).flatten()) / np.sum(sym_mask)

def jaccard_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
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

def masked_metric_factory(metric_name: str, ) -> Callable[[np.ndarray, np.ndarray], float]:
    if metric_name not in metrics_map:
        raise ValueError(f'Unknown metric: {metric_name}')
    return metrics_map[metric_name]

