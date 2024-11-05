import numpy as np

def expand_hole(precursor_mask: np.array, theta: np.array, angular_range_expansion) -> np.array:
    num_beams = round(angular_range_expansion*theta.shape[0]/360)

    expanded_hole_mask = precursor_mask.copy()
    for i in range(-num_beams, num_beams):
        rolled_mask = np.roll(precursor_mask, shift=i)
        expanded_hole_mask = np.logical_and(expanded_hole_mask, rolled_mask)

    return expanded_hole_mask.copy()

def compute_cyclic_shift( precursor_mask: np.array, theta: np.array, cyclic_shift) -> np.array:
    shift = np.argmin(np.abs((cyclic_shift % 360) - theta))
    return np.logical_and(precursor_mask, np.roll(precursor_mask, shift))