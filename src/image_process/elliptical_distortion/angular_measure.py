from ..polar.polar_representation import PolarRepresentation
from ..polar.rotational_average import RotationalAverage
from ..mask.polar_mask import PolarLineMask
from .elastic_closeness import ElasticCloseness


class AngularMeasure:
    def __init__(self, polar_representation: PolarRepresentation):
        self._polar_representation = polar_representation
        self._polar_representation.get_polar_representation(0.1, 0.6)

        self._angular_range = 20
        self._polar_line_mask = PolarLineMask(self._polar_representation).get_polar_line_mask(
                                                                                    angular_range_expansion=self._angular_range/2,
                                                                                    cyclic_shift=90
                                                                                    )
    def measure(self):
        rotational_average = RotationalAverage(polar_representation=self._polar_representation)

        theta_list = []
        optimal_shrink_factors = []
        for i, theta in enumerate(self._polar_representation.theta):
            direct_theta = theta
            ort_theta = theta + 90

            if self._polar_line_mask[i]:
                direc_profile = rotational_average.get_rotational_average(direct_theta-self._angular_range/2,
                                                                          direct_theta+self._angular_range/2)  
                ort_profile = rotational_average.get_rotational_average(ort_theta-self._angular_range/2,
                                                                        ort_theta+self._angular_range/2)
                
                elastic_closeness = ElasticCloseness(direc_profile, ort_profile)
                optimal_shrink_factor = elastic_closeness.get_optimal_shrink_factor()

                theta_list.append(theta)
                optimal_shrink_factors.append(optimal_shrink_factor)
        return theta_list, optimal_shrink_factors

        