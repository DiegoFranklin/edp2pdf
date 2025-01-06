from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_representation import PolarRepresentation
from src.image_process.polar.rotational_integration import RotationalIntegration
from src.image_process.mask.utils import expand_hole, compute_cyclic_shift

from src.signal_process.utils import cosine_distance, shrink_signal, taper_and_filter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from typing import Tuple, List
from scipy import signal

class EllipseParams:
    def __init__(self, edp: eDiffractionPattern) -> None:
        """
        Initializes an EllipseParams object from an eDiffractionPattern.

        Args:
            edp: An eDiffractionPattern object.
        """
        self._polar_representation = PolarRepresentation(edp, radial_range=(0.06, 0.6))
        self._semi_angle_range = 10

    
    def _construct_angular_mask(self) -> np.ndarray:
        """
        Construct an angular mask for the diffraction pattern based on the
        original angular mask and the specified semi-angle range.

        The mask is expanded by the specified semi-angle range and then intersected with its
        90 degrees shifted version to produce the final mask.

        Args:
            semi_angle_range: The semi-angle range to expand the mask by.

        Returns:
            A 1-dimensional numpy array representing the final angular mask
        """
    
        angular_mask = expand_hole(self._polar_representation.angular_mask, 
                                   self._polar_representation.theta, 
                                   self._semi_angle_range)

        return compute_cyclic_shift(angular_mask, self._polar_representation.theta, -90)

    def _extract_orthogonal_curves(
        self, angle: float, rotational_average: RotationalIntegration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts two orthogonal curves from the rotational average of the
        diffraction pattern. The first curve is centered at the specified
        angle and the second curve is centered at the same angle plus 90
        degrees. The curves are extracted by computing the rotational
        average over the specified semi-angle range.

        Parameters
        ----------
        angle : float
            The angle at which to extract the curves.
        rotational_average : RotationalIntegration
            The rotational average of the diffraction pattern.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two arrays, each representing one of the extracted curves.
        """
        pk_angle = angle
        qk_angle = pk_angle + 90

        pk_angles = (pk_angle - self._semi_angle_range, pk_angle + self._semi_angle_range)
        qk_angles = (qk_angle - self._semi_angle_range, qk_angle + self._semi_angle_range)

        pk = rotational_average.get_rotational_integration(*pk_angles)
        qk = rotational_average.get_rotational_integration(*qk_angles)

        return pk, qk
        
    @staticmethod
    def _elastic_divergence(pk: np.ndarray, qk: np.ndarray) -> float:

        """
        Compute the elastic divergence between two rotational averages.

        This method calculates the divergence by finding the optimal scaling factor 
        that minimizes the cosine distance between the input array `pk` 
        and the scaled version of `qk`. The scaling factor is determined using 
        scalar minimization over a specified range.

        Parameters:
        pk (np.ndarray): The first rotational average array.
        qk (np.ndarray): The second rotational average array to be scaled.

        Returns:
        float: The scaling factor that minimizes the cosine distance.
        """
        min_func = lambda alpha: cosine_distance(pk, shrink_signal(qk, alpha))

        result = minimize_scalar(min_func, method='Brent', bracket=(0.8, 1.2))

        return result.x
    
    def _build_probe_angles(self, num_points: int = 10) -> np.ndarray:
 
        """
        Build probe angles from the valid theta space.

        This method constructs an angular mask and extracts valid theta
        values based on the mask. It then selects probe angles at regular
        intervals specified by the number of points.

        Args:
            num_points (int): The number of points to sample from the valid
                                theta space.

        Returns:
            np.ndarray: An array of probe angles sampled from the valid theta
                        space at the specified interval.
        """
        complete_angular_mask: np.ndarray = self._construct_angular_mask()
        valid_theta_space: np.ndarray = self._polar_representation.theta[np.where(complete_angular_mask)]

        probe_angles = valid_theta_space[::num_points] 
        
        return probe_angles
    
    def _azimuthal_scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform an azimuthal scan to obtain valid theta values and their corresponding
        divergences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the valid theta values and
                their corresponding divergences.
        """
        probe_angles: np.ndarray = self._build_probe_angles()
        divergences: List[float] = []

        rotational_average: RotationalIntegration = RotationalIntegration(self._polar_representation)

        for angle in probe_angles:
            pk, qk = self._extract_orthogonal_curves(angle, rotational_average)

            pp_pk: np.ndarray = taper_and_filter(pk)
            pp_qk: np.ndarray = taper_and_filter(qk)

            divergence: float = EllipseParams._elastic_divergence(pp_pk, pp_qk)
            divergences.append(divergence)

        return probe_angles, np.asarray(divergences) - 1
    
    @staticmethod
    def _cos(thetas: np.ndarray, amplitude: float, phase: float) -> np.ndarray:

        return amplitude * np.cos(2 * (np.pi / 180 * thetas - (np.pi / 180) * phase))
    
    def measure_ellipse_params(self) -> Tuple[float, float]:
        
        """
        Measure the parameters of an ellipse from a diffraction pattern.

        This method performs an azimuthal scan to obtain valid theta values 
        and their corresponding divergences. It then fits a cosine function 
        to these divergences to extract the ellipse parameters, specifically 
        the axis ratio and orientation.

        Returns:
            Tuple[float, float]: A dictionary containing:
                - 'axis_ratio': The ratio between the semi-major and semi-minor axes.
                - 'orientation': The orientation angle of the ellipse in degrees.
        """
        valid_theta_space, divergences = self._azimuthal_scan()

        
        bounds = ((0, -np.inf), (np.inf, np.inf))

        params, _ = curve_fit(EllipseParams._cos, valid_theta_space, divergences, bounds=bounds)

        amplitude, phase = 1 + params[0], params[1] % 180
    

        env_mean = 1 + np.mean(np.abs(signal.hilbert(divergences)))

        params = {'axis_ratio': env_mean,
                  'orientation': phase}
        
        return params

