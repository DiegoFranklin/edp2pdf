from ..diffraction_pattern import eDiffractionPattern
from ..polar.rotational_average import RotationalAverage
from ..polar.polar_representation import PolarRepresentation
import matplotlib.pyplot as plt

class CenterProber:
    def __init__(self, edp: eDiffractionPattern):
        self._polar_rep = PolarRepresentation(edp, 0.05, 0.6)
        self._rotational_average = RotationalAverage(self._polar_rep)

    def probe(self, angle, angle_range):
        x = self._polar_rep.radius
        angles = [angle, angle+90]

        for a in angles:
            plt.figure(figsize=(10, 3))
            
            plt.plot(x,
                     self._rotational_average.get_rotational_average(a - angle_range / 2, a + angle_range / 2),
                     label=f'{a - angle_range / 2}° - {a + angle_range / 2}°')
            
            plt.plot(x,
                     self._rotational_average.get_rotational_average(a + 180 - angle_range / 2, a + 180 + angle_range / 2),
                     label=f'{a + 180 - angle_range / 2}° - {a + 180 + angle_range / 2}°')
            
            plt.title(f'Rotational Average for Angles {a}° and {a + 180}°')
            plt.xlabel('Radial Distance')
            plt.ylabel('Rotational Average')
            plt.yscale('log')
            plt.grid()
            plt.legend()
            plt.show()
