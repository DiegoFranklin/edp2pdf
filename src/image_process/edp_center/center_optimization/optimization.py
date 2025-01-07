import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Callable, Optional, Dict


def optimize_center(
    penalty_func: Callable[[np.ndarray], float],
    data_shape: Tuple[int, int],
    initial_guess: Optional[Tuple[float, float]] = None,
    options: Optional[Dict] = None
) -> Tuple[float, float]:
    """
    Optimize the center of a given penalty function using the 'trust-constr' method.

    Args:
        penalty_func (Callable[[np.ndarray], float]): A function that computes the penalty for a given point.
        data_shape (Tuple[int, int]): The shape of the data, used to determine the default initial guess.
        initial_guess (Optional[Tuple[float, float]]): The initial guess for the center. If not provided,
                                                      it defaults to the center of the data shape.
        options (Optional[Dict]): Additional options to pass to the `minimize` function. Defaults to
                                  {'xtol': 1e-8, 'disp': False}.

    Returns:
        Tuple[float, float]: The optimized center coordinates.

    Raises:
        ValueError: If `data_shape` is invalid, `initial_guess` is out of bounds, or the optimization fails to converge.
        TypeError: If `penalty_func` is not callable or `data_shape` is not a tuple of integers.
    """
    # Validate input types
    if not callable(penalty_func):
        raise TypeError("`penalty_func` must be a callable function.")
    if not isinstance(data_shape, tuple) or len(data_shape) != 2 or not all(isinstance(d, int) for d in data_shape):
        raise TypeError("`data_shape` must be a tuple of two integers.")

    # Validate data shape
    if data_shape[0] <= 0 or data_shape[1] <= 0:
        raise ValueError("`data_shape` must have positive dimensions.")

    # Set default initial guess if not provided
    if initial_guess is None:
        initial_guess = np.asarray(data_shape) / 2.0
    else:
        if not isinstance(initial_guess, (tuple, np.ndarray)) or len(initial_guess) != 2:
            raise TypeError("`initial_guess` must be a tuple or array of two floats.")
        if initial_guess[0] < 0 or initial_guess[1] < 0 or initial_guess[0] > data_shape[0] or initial_guess[1] > data_shape[1]:
            raise ValueError("`initial_guess` must be within the bounds of `data_shape`.")

    # Set default options if not provided
    if options is None:
        options = {
            'xtol': 1e-8,
            'disp': False
        }
    elif not isinstance(options, dict):
        raise TypeError("`options` must be a dictionary.")

    # Perform optimization
    result = minimize(
        penalty_func,
        initial_guess,
        method='trust-constr',
        options=options
    )

    # Check if optimization was successful
    if result.success:
        return tuple(result.x)
    else:
        raise ValueError(f'Optimization did not converge. Message: {result.message}')