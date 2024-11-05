import numpy as np
from scipy import signal

def cosine_distance(pk, qk):
    """
    Compute the cosine distance between two vectors.

    The cosine distance is defined as 1 - cos(angle) where angle is the angle between
    the two vectors. The cosine of the angle is computed as the dot product of the two
    vectors divided by their magnitudes.

    Parameters
    ----------
    pk, qk : array_like
        Input vectors.

    Returns
    -------
    float
        The cosine distance between the two vectors.
    """
    return 1 - np.dot(pk, qk) / (np.linalg.norm(pk) * np.linalg.norm(qk))

def shrink_signal(sig, alpha):
    
    """
    Compresses a signal by shrinking it by a factor of alpha while
    maintaining the same length.

    Parameters
    ----------
    sig : array_like
        Input signal to be shrunk.
    alpha : float
        Compression factor.

    Returns
    -------
    array_like
        The compressed signal.
    """
    original_indices = np.arange(len(sig))
    
    new_indices = np.linspace(0, (len(sig) - 1) * alpha, len(sig))
    
    compressed_signal = np.interp(new_indices, original_indices, sig)
    
    return compressed_signal

def taper_and_filter(sig):
    """
    Apply a Tukey window and a Savitzky-Golay filter to a signal.

    Parameters
    ----------
    sig : array_like
        Input signal.

    Returns
    -------
    array_like
        The filtered signal.
    """

    sig = np.asarray(sig)
    tukey_window = signal.windows.tukey(sig.shape[0], alpha=1)

    return tukey_window * (signal.savgol_filter(sig, 5, 3))