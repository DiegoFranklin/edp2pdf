from scipy.spatial import distance
from typing import Callable
import numpy as np
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim

def euclidean(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return np.sum(np.sqrt((a - b) ** 2) * mask) / np.sum(mask)

def manhattan(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sym_mask = mask * np.flip(mask)


    return np.sum(np.abs(a * sym_mask - b * sym_mask)) / np.sum(mask)

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return ssim(a, b)

def mean_squared_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return np.sum((a - b) ** 2 * mask) / np.sum(mask)

def masked_dot_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return 1 - np.sum(np.multiply(a , b) * mask) / np.sum(mask)

def cosine(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return distance.cosine((a * mask).flatten(), (b * mask).flatten())

def jaccard_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = 255 * (a / np.max(a))
    b = 255 * (b / np.max(b))

    a, b = a.astype('uint8'), b.astype('uint8') 

    mask = mask.astype('uint8')
    return 1 - metrics.jaccard_score((a * mask).flatten(), (b * mask).flatten(), average='macro')



metrics_map = {
    'euclidean': euclidean,
    'mean_squared_error': mean_squared_error,
    'manhattan': manhattan,
    'cosine': cosine,
    'masked_dot_distance': masked_dot_distance,
    'jaccard_distance': jaccard_distance,
    'ssim': ssim
}

def masked_metric_factory(metric_name: str, ) -> Callable[[np.ndarray, np.ndarray], float]:
    if metric_name not in metrics_map:
        raise ValueError(f'Unknown metric: {metric_name}')
    return metrics_map[metric_name]

