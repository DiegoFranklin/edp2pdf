import numpy as np
from scipy import signal


def cosine_distance(pk: np.ndarray, qk: np.ndarray) -> float:
    """
    Compute the cosine distance between two vectors.

    The cosine distance is defined as 1 - cos(angle), where angle is the angle between
    the two vectors. The cosine of the angle is computed as the dot product of the two
    vectors divided by their magnitudes.

    Args:
        pk (np.ndarray): The first input vector.
        qk (np.ndarray): The second input vector.

    Returns:
        float: The cosine distance between the two vectors.

    Raises:
        TypeError: If `pk` or `qk` are not numpy arrays.
        ValueError: If `pk` or `qk` are empty or have different shapes.
    """
    if not isinstance(pk, np.ndarray) or not isinstance(qk, np.ndarray):
        raise TypeError("Inputs `pk` and `qk` must be numpy arrays.")
    if pk.size == 0 or qk.size == 0:
        raise ValueError("Inputs `pk` and `qk` must not be empty.")
    if pk.shape != qk.shape:
        raise ValueError("Inputs `pk` and `qk` must have the same shape.")

    return 1 - np.dot(pk, qk) / (np.linalg.norm(pk) * np.linalg.norm(qk))


def shrink_signal(sig: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compresses a signal by shrinking it by a factor of alpha while maintaining the same length.

    Args:
        sig (np.ndarray): The input signal to be shrunk.
        alpha (float): The compression factor.

    Returns:
        np.ndarray: The compressed signal.

    Raises:
        TypeError: If `sig` is not a numpy array or `alpha` is not a float.
        ValueError: If `sig` is empty or `alpha` is not positive.
    """
    if not isinstance(sig, np.ndarray):
        raise TypeError("Input `sig` must be a numpy array.")
    if sig.size == 0:
        raise ValueError("Input `sig` must not be empty.")
    if not isinstance(alpha, (int, float)):
        raise TypeError("Input `alpha` must be a float.")
    if alpha <= 0:
        raise ValueError("Input `alpha` must be positive.")

    original_indices = np.arange(len(sig))
    new_indices = np.linspace(0, (len(sig) - 1) * alpha, len(sig))
    compressed_signal = np.interp(new_indices, original_indices, sig)

    return compressed_signal


def taper_and_filter(sig: np.ndarray) -> np.ndarray:
    """
    Apply a Tukey window and a Savitzky-Golay filter to a signal.

    Args:
        sig (np.ndarray): The input signal.

    Returns:
        np.ndarray: The filtered signal.

    Raises:
        TypeError: If `sig` is not a numpy array.
        ValueError: If `sig` is empty.
    """
    if not isinstance(sig, np.ndarray):
        raise TypeError("Input `sig` must be a numpy array.")
    if sig.size == 0:
        raise ValueError("Input `sig` must not be empty.")

    sig = np.asarray(sig)
    tukey_window = signal.windows.tukey(sig.shape[0], alpha=1)

    return tukey_window * (signal.savgol_filter(sig, 5, 3))