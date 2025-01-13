# %%
from import_src import import_src
import_src()

from edp2pdf.image_process.image_io import LoadImage, WriteImage
import edp2pdf.image_process.pre_process as prep
import edp2pdf.image_process.mask.mask_getters as maskget
from edp2pdf.image_process.edp_center.first_estimation.centroid import get_centroid
from edp2pdf.image_process.edp_center.center_optimization.opt_funcs import Distance
from edp2pdf.image_process.edp_center.center_optimization.optimization import optimize_center
from edp2pdf.image_process.edp_center.first_estimation.autocorrelation import AutoCorrelation
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern
from edp2pdf.image_process.ellipse.elp_params import EllipseParams
from edp2pdf.image_process.ellipse.elp_correction import correct_ellipse
from edp2pdf.image_process.polar.polar_representation import PolarRepresentation
from edp2pdf.image_process.polar.rotational_integration import RotationalIntegration


import os
import numpy as np
from typing import Iterable, List, Union, Callable, TypeVar, Tuple
from collections.abc import Iterable as IterableABC


# %%
from typing import TypeVar, Callable, Iterable, List, Union, Any
from collections.abc import Iterable as IterableABC

T = TypeVar("T")
U = TypeVar("U")

def batch_run(batch_input: Iterable[T], operate_single: Callable[..., U], *args, **kwargs) -> List[U]:
    """
    Applies the given function to each item in the batch, supporting multiple arguments.

    Args:
        batch_input: An iterable of inputs to process.
        operate_single: A function that processes a single input of type `T` and accepts additional arguments.
        *args: Additional positional arguments to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        A list of processed inputs.

    Raises:
        TypeError: If any element in the iterable is not of the expected type.
    """
    # Check that all elements in the iterable are of the expected type
    first_element_type = type(next(iter(batch_input)))
    if not all(isinstance(x, first_element_type) for x in batch_input):
        raise TypeError(f"All elements in the iterable must be of type {first_element_type.__name__}.")
    return [operate_single(x, *args, **kwargs) for x in batch_input]

def make_batch_version(operate_single: Callable[..., U]) -> Callable[[Union[Iterable[T], T], ...], Union[List[U], U]]:
    """
    Creates a batch-compatible version of a function that can take multiple arguments.

    Args:
        operate_single: A function that processes a single input of type `T` and accepts additional arguments.

    Returns:
        A function that can process either a single input or an iterable of inputs, and accepts additional arguments.
    """
    def batch_version(input: Union[Iterable[T], T], *args, **kwargs) -> Union[List[U], U]:
        """
        Processes either a single input or an iterable of inputs, supporting multiple arguments.

        Args:
            input: A single input or an iterable of inputs.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            The processed input or a list of processed inputs.

        Raises:
            TypeError: If the input is not of the expected type.
            ValueError: If the iterable is empty.
        """
        if not isinstance(input, IterableABC) or isinstance(input, (str, bytes)):
            # Single item
            return operate_single(input, *args, **kwargs)
        elif isinstance(input, IterableABC):
            # Batch processing
            if not input:
                raise ValueError("The input iterable must not be empty.")
            return batch_run(input, operate_single, *args, **kwargs)
        else:
            # Unsupported type
            raise TypeError(
                f"Input must be of type {T} or an iterable of {T}, "
                f"but got {type(input).__name__}."
            )
    return batch_version

# %%
def load_single(path: str) -> np.ndarray:
    return LoadImage(path).data

load = make_batch_version(load_single)

# %%
def preprocess_single(data):
    median_filter = prep.MedianFilter(kernel_size=5)
    all_positive = prep.AllPositive()

    pre_processors = [median_filter, all_positive]

    pre_pipe = prep.PreProcessPipe(pre_processors=pre_processors)
    data = pre_pipe.pre_process_pipe(data)

    return data

preprocess = make_batch_version(preprocess_single)

# %%
def get_mask_single(data):
    mean_mask = maskget.MeanTreshMask(.1)

    mask = maskget.superpose_masks(data, [mean_mask])

    return mask

get_mask = make_batch_version(get_mask_single)

# %%
from typing import Tuple
from functools import partial

def autocorr(data, mask):
    ac = AutoCorrelation(data, mask)
    center, _ = ac.compute()
    return center

def first_guess(data, mask, method: str = "centroid"):
    """
    Computes the first guess for the center using the specified method.

    Args:
        data: Input data (e.g., a numpy array).
        mask: Mask for the input data.
        method: Method to use for computing the first guess. Supported values are "autocorrelation" and "centroid".

    Returns:
        The computed first guess for the center.

    Raises:
        ValueError: If the method is not supported.
    """
    methods = {
        "autocorrelation": lambda: autocorr(data, mask),
        "centroid": lambda: get_centroid(data),
    }

    if method not in methods:
        raise ValueError(f"Unknown first guess method: {method}")

    return methods[method]()

def find_center_single(data_mask: Tuple, distance_metric: str = "manhattan", first_guess_method: str = "centroid"):
    """
    Finds the center of the input data using the specified distance metric and first guess method.

    Args:
        data_mask: A tuple containing the input data and mask.
        distance_metric: The distance metric to use for optimization. Defaults to "manhattan".
        first_guess_method: The method to use for computing the first guess. Defaults to "centroid".

    Returns:
        The computed center.
    """
    data, mask = data_mask

    # Compute the first guess
    initial_guess = first_guess(data, mask, method=first_guess_method)

    # Create the penalty function
    penalty_func = Distance(data, mask, distance_metric=distance_metric).get_penalty_func()

    # Optimize the center
    center = optimize_center(penalty_func, data.shape, initial_guess=initial_guess)

    return center

# Configuration
first_guess_method = "centroid"
distance_metric = "manhattan"

# Create the batch version of find_center_single
find_center = make_batch_version(
    partial(find_center_single, distance_metric=distance_metric, first_guess_method=first_guess_method)
)

# %%
def edp_instantiate_single(edp_data: Tuple[np.ndarray, np.ndarray, Tuple]) -> eDiffractionPattern:
    data = edp_data[0]
    center = edp_data[1]
    mask = edp_data[2]
    return eDiffractionPattern(data=data, center=center, mask=mask)

edp_instantiate = make_batch_version(edp_instantiate_single)

# %%
def measure_elp_params_single(calibration_edp: eDiffractionPattern):
    elp_params = EllipseParams(edp=calibration_edp)
    params = elp_params.measure_ellipse_params()

    return params

measure_elp_params = make_batch_version(measure_elp_params_single)


# %%
# Define the single-item correction function
def correct_ellipse_single(input_data: Tuple[eDiffractionPattern, dict]) -> np.ndarray:
    edp, ellipse_params = input_data  # Unpack the tuple
    return correct_ellipse(edp, ellipse_params)  # Call the correct_ellipse function

# Create the batched version
elp_correction = make_batch_version(correct_ellipse_single)

# %%
def polar_representation_instantiate_single(edp: eDiffractionPattern) -> PolarRepresentation:
    polar_representation = PolarRepresentation(edp=edp)
    return polar_representation

polar_representation_instantiate = make_batch_version(polar_representation_instantiate_single)

# %%
def rotational_integrate_single(input: Tuple[PolarRepresentation, str, bool]) -> np.ndarray:
    polar_representation, method, use_mask = input
    rotational_integration = RotationalIntegration(polar_representation)
    return rotational_integration.get_rotational_integration(start_angle=0, end_angle=359, method=method, use_mask=use_mask)

rotational_integrate = make_batch_version(rotational_integrate_single)

# %%
samples_path = "../raw_data/samples/"
samples_file_names = os.listdir(samples_path)
samples_paths = [samples_path + x for x in samples_file_names]

calibration_samples_path = "../raw_data/gold/"
calibration_file_names = os.listdir(calibration_samples_path)
calibration_paths = [calibration_samples_path + x for x in calibration_file_names]

# %%
all_paths = samples_paths + calibration_paths

print("Loading data...")
loaded_data = load(all_paths)

print("Preprocessing data...")
preprocessed_data = preprocess(loaded_data)

print("Getting masks...")
masks = get_mask(preprocessed_data)

print("Finding centers...")
centers = find_center([(d, m) for d, m in zip(preprocessed_data, masks)])

print("Instantiating edps...")
all_edps = edp_instantiate([(d, c, m) for d, c, m in zip(preprocessed_data, centers, masks)])

print("Measuring elliptical params...")
elp_params = measure_elp_params(all_edps[-len(calibration_paths):])
mean_params = elp_params[0]

print("Instantiating edps...")
all_raw_edps = edp_instantiate([(d, c, m) for d, c, m in zip(loaded_data, centers, masks)])

print("Correcting ellipses...")
corrected_ellipses = elp_correction([(edp, mean_params) for edp in all_raw_edps])

print("Preprocessing data...")
preprocessed_corrected_data = preprocess(corrected_ellipses)

print("Getting masks...")
masks_corrected = get_mask(preprocessed_corrected_data)

print("Finding centers...")
centers_corrected = find_center([(d, m) for d, m in zip(preprocessed_corrected_data, masks_corrected)])

print("Instantiating edps...")
all_corrected_edps = edp_instantiate([(d, c, m) for d, c, m in zip(corrected_ellipses, centers_corrected, masks_corrected)])

print("Instantiating polar representations...")
all_polar_representations = polar_representation_instantiate(all_corrected_edps)

print("Rotational integration...")
all_rotational_integrations = rotational_integrate([(plrep, 'median', True) for plrep in all_polar_representations])

print("Rotational integration...")
all_rotational_integrations = rotational_integrate([(plrep, 'mean', True) for plrep in all_polar_representations])




