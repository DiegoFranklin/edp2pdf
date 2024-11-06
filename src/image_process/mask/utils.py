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

def find_ones_group_limits(arr):
    one_indexes = np.where(arr)[0]
    
    if arr[0] == 1 and arr[-1] == 1:
        for i, valid_index in enumerate(one_indexes):
            diff = one_indexes[i + 1] - valid_index

            if diff > 1:
                start = one_indexes[i + 1]
                end = valid_index
                
                return start, end
            
    start = one_indexes[0]
    end = one_indexes[-1]
    
    return start, end

def get_valid_theta_range(angular_mask, theta_space):
    start, end = find_ones_group_limits(angular_mask)

    return theta_space[start], theta_space[end]