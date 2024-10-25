import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Callable

def optimize_center(penalty_func: Callable[[np.ndarray], float], data_shape: Tuple[int, int], 
                    initial_guess: Tuple[float, float] = None, options=None) -> Tuple[float, float]:
    
    if initial_guess is None:
        initial_guess = np.asarray(data_shape) / 2.0

    if options is None:
        options = {
            'xtol': 1e-8,
            'disp': False   
        }


    result = minimize(
        penalty_func,
        initial_guess,
        method='trust-constr',
        options=options
    )


    if result.success:
        return tuple(result.x)
    else:
        raise ValueError('Optimization did not converge. Message: {}'.format(result.message))