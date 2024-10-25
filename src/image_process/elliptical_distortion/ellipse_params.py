from ..elliptical_distortion.angular_measure import AngularMeasure
from ..diffraction_pattern import eDiffractionPattern
from ..polar.polar_representation import PolarRepresentation

import numpy as np
from scipy import signal

class EllipseParams:
    def __init__(self, edp: eDiffractionPattern):
        self._polar_representation = PolarRepresentation(edp)
        self._polar_representation.get_polar_representation(0.1, 0.6)

    def get_ellipse_correction_params(self):
        angular_measure = AngularMeasure(polar_representation=self._polar_representation)
        theta, opt_sf = angular_measure.measure()

        env = np.abs(signal.hilbert(opt_sf - np.mean(opt_sf)))
        vanila_cos = opt_sf/env

        # Calculate the mean and standard deviation of the 'env' array
        mean_env = np.mean(env)
        std_env = np.std(env)

        # Select values within one standard deviation of the mean
        filtered_env = env[(env > mean_env - std_env) & (env < mean_env + std_env)]

        # Compute the average amplitude from the filtered values
        amplitude = np.mean(filtered_env)


        # to radians
        angles = np.pi*np.asarry(theta)/2

        # Compute the negative cosine of twice the angles
        probe_cosine_wave = -np.cos(2 * angles)

        # Apply the Hilbert transform to both the probe and the reference signal
        hilbert_probe = signal.hilbert(probe_cosine_wave)
        hilbert_reference = signal.hilbert(vanila_cos)

        # Calculate the instantaneous phase angles from the Hilbert transform
        phase_probe = np.angle(hilbert_probe)
        phase_reference = np.angle(hilbert_reference)

        # Compute the phase difference between the probe and reference signals
        phase_diff = phase_probe - phase_reference

        # Calculate the mean and standard deviation of the phase difference
        mean_diff = np.mean(phase_diff)
        std_diff = np.std(phase_diff)

        # Select phase differences within one standard deviation of the mean
        filtered_diff = phase_diff[(phase_diff > mean_diff - std_diff) & (phase_diff < mean_diff + std_diff)]

        # Compute the average phase from the filtered differences
        phase = np.mean(filtered_diff)

        orientation = phase*90/np.pi + 90
        sc_fac = 1+amplitude/2

        return orientation, sc_fac
