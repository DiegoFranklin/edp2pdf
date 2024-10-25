import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy import signal

def shrink_signal(old_x, sig, shrink_factor):
    newx = np.linspace(old_x[0], old_x[-1] * shrink_factor, len(old_x))
    return np.interp(newx, old_x, sig)

class ElasticCloseness:
    def __init__(self, curve1, curve2):
        self.x = np.arange(0, curve1.shape[0], 1)
        tukey_window = signal.windows.tukey(curve1.shape[0], alpha=0.5)

        decaying1 = Polynomial.fit(self.x, curve1, 4, domain=[self.x[0], self.x[-1]]).linspace(n=self.x.shape[0])[1]
        decaying2 = Polynomial.fit(self.x, curve2, 4, domain=[self.x[0], self.x[-1]]).linspace(n=self.x.shape[0])[1]

        self.curve1 = tukey_window*(curve1 - decaying1)
        self.curve2 = tukey_window*(curve2 - decaying2)

    def measure(self, shrink_factor):
        shrinked_curve1 = shrink_signal(self.x, self.curve1, shrink_factor)

        closeness = np.dot(shrinked_curve1, self.curve2)
        return closeness
    
    def get_optimal_shrink_factor(self, min: float = 0.9, max: float = 1.1):

        shrink_factors = np.linspace(min, max, 1000)
        
        closeness_list = [self.measure(sf) for sf in shrink_factors]

        return shrink_factors[np.argmax(closeness_list)]