import numpy as np
from typing import Tuple


def expand_hole(precursor_mask: np.ndarray, theta: np.ndarray, angular_range_expansion: float) -> np.ndarray:
    """
    Expands the hole in the precursor mask by applying a logical AND operation with rolled versions of the mask.

    Args:
        precursor_mask (np.ndarray): The input binary mask as a numpy array.
        theta (np.ndarray): The angular space corresponding to the mask.
        angular_range_expansion (float): The angular range (in degrees) to expand the hole.

    Returns:
        np.ndarray: The expanded hole mask as a numpy array.

    Raises:
        TypeError: If `precursor_mask` or `theta` are not numpy arrays.
        ValueError: If `precursor_mask` and `theta` have incompatible shapes or if `angular_range_expansion` is invalid.
    """
    if not isinstance(precursor_mask, np.ndarray) or not isinstance(theta, np.ndarray):
        raise TypeError("Inputs `precursor_mask` and `theta` must be numpy arrays.")
    if precursor_mask.shape != theta.shape:
        raise ValueError("Shapes of `precursor_mask` and `theta` must match.")
    if angular_range_expansion <= 0 or angular_range_expansion > 360:
        raise ValueError("`angular_range_expansion` must be a positive value less than or equal to 360.")

    num_beams = round(angular_range_expansion * theta.shape[0] / 360)

    expanded_hole_mask = precursor_mask.copy()
    for i in range(-num_beams, num_beams):
        rolled_mask = np.roll(precursor_mask, shift=i)
        expanded_hole_mask = np.logical_and(expanded_hole_mask, rolled_mask)

    return expanded_hole_mask.copy()


def compute_cyclic_shift(precursor_mask: np.ndarray, theta: np.ndarray, cyclic_shift: float) -> np.ndarray:
    """
    Computes a cyclic shift of the precursor mask and applies a logical AND operation with the original mask.

    Args:
        precursor_mask (np.ndarray): The input binary mask as a numpy array.
        theta (np.ndarray): The angular space corresponding to the mask.
        cyclic_shift (float): The cyclic shift angle (in degrees).

    Returns:
        np.ndarray: The resulting mask after applying the cyclic shift and logical AND operation.

    Raises:
        TypeError: If `precursor_mask` or `theta` are not numpy arrays.
        ValueError: If `precursor_mask` and `theta` have incompatible shapes.
    """
    if not isinstance(precursor_mask, np.ndarray) or not isinstance(theta, np.ndarray):
        raise TypeError("Inputs `precursor_mask` and `theta` must be numpy arrays.")
    if precursor_mask.shape != theta.shape:
        raise ValueError("Shapes of `precursor_mask` and `theta` must match.")

    shift = np.argmin(np.abs((cyclic_shift % 360) - theta))
    return np.logical_and(precursor_mask, np.roll(precursor_mask, shift))


def find_ones_group_limits(arr: np.ndarray) -> Tuple[int, int]:
    """
    Finds the start and end indices of the largest continuous group of ones in a binary array.

    Args:
        arr (np.ndarray): The input binary array.

    Returns:
        Tuple[int, int]: The start and end indices of the largest continuous group of ones.

    Raises:
        TypeError: If `arr` is not a numpy array.
        ValueError: If `arr` is not a 1D array or does not contain any ones.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input `arr` must be a numpy array.")
    if len(arr.shape) != 1:
        raise ValueError("Input `arr` must be a 1D array.")
    if np.sum(arr) == 0:
        raise ValueError("Input `arr` must contain at least one '1'.")

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


def get_valid_theta_range(angular_mask: np.ndarray, theta_space: np.ndarray) -> Tuple[float, float]:
    """
    Computes the valid theta range based on the angular mask.

    Args:
        angular_mask (np.ndarray): The input binary mask as a numpy array.
        theta_space (np.ndarray): The angular space corresponding to the mask.

    Returns:
        Tuple[float, float]: The start and end angles of the valid theta range.

    Raises:
        TypeError: If `angular_mask` or `theta_space` are not numpy arrays.
        ValueError: If `angular_mask` and `theta_space` have incompatible shapes or if the mask is invalid.
    """
    if not isinstance(angular_mask, np.ndarray) or not isinstance(theta_space, np.ndarray):
        raise TypeError("Inputs `angular_mask` and `theta_space` must be numpy arrays.")
    if angular_mask.shape != theta_space.shape:
        raise ValueError("Shapes of `angular_mask` and `theta_space` must match.")
    if np.sum(angular_mask) == 0:
        raise ValueError("Input `angular_mask` must contain at least one '1'.")

    start, end = find_ones_group_limits(angular_mask)
    return theta_space[start], theta_space[end]