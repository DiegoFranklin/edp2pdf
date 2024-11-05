import numpy as np
from scipy import signal

def cosine_distance(pk, qk):
    return 1 - np.dot(pk, qk) / (np.linalg.norm(pk) * np.linalg.norm(qk))

def shrink_signal(sig, alpha):
    
    original_indices = np.arange(len(sig))
    
    new_indices = np.linspace(0, (len(sig) - 1) * alpha, len(sig))
    
    compressed_signal = np.interp(new_indices, original_indices, sig)
    
    return compressed_signal

def taper_and_filter(sig):
    sig = np.asarray(sig)
    tukey_window = signal.windows.tukey(sig.shape[0], alpha=1)

    return tukey_window * (signal.savgol_filter(sig, 5, 3))