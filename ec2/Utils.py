from math import sqrt
import numpy as np
import math

'''
A class to collect statistics about the evolution of the dimensions during optimisation.
For now we keep:
- the sum of moves per dimension (i.e. how far we are from the initial value for each dimension)
- the sum of absolute moves per dimension
- the sum of absolute moves of the param vector
'''
class Statistics:
    '''
    When initialising, we provide the parameter vector (a numpy array),
    because we need its shape.
    '''
    def __init__(self, params, status=None):
        if status is None:
            self.dim = np.zeros(params.shape, dtype=np.float32)
            self.dim_abs = np.zeros(params.shape, dtype=np.float32)
            self.tot_abs = 0
        else:
            self.dim = np.array(status['dim'], dtype=np.float32)
            self.dim_abs = np.array(status['dim_abs'], dtype=np.float32)
            self.tot_abs = status['tot_abs']

    def step(self, diff):
        self.dim += diff
        self.dim_abs += np.abs(diff)
        self.tot_abs += np.linalg.norm(diff)

    def pressure(self):
        return self.dim / self.dim_abs

    def relevance(self):
        pres = np.abs(self.pressure())
        return pres / pres.sum()

    def get_status(self):
        return {
            'dim': self.dim.tolist(),
            'dim_abs': self.dim_abs.tolist(),
            'tot_abs': self.tot_abs
        }

'''
A class to estimate statistics of a gaussian distribution
Can be used for example to estimate the timeout of the game playing
'''
class Gauss:
    def __init__(self):
        self.n = 0
        self.s1 = 0
        self.s2 = 0

    def add(self, v):
        self.n += 1
        self.s1 += v
        self.s2 += v * v

    def mean(self):
        return self.s1 / self.n

    def std(self):
        return sqrt((self.s2 - self.s1 * self.s1 / self.n) / self.n)

'''
Game duration is modelled as an exponential distribution, where lambda is the frequence
of games terminated in unit of time, i.e. 1/lambda is game duration (and standard deviation
at the same time)

Then we are interested in the quantile for some percent of games beeing terminated normally
(i.e. not through a timeout)
'''
class Expo(Gauss):
    def quantile(self, p):
        return -self.mean() * math.log(1 - p)

if __name__ == '__main__':
    ex = Expo()
    for t in (452, 393, 379, 406, 442, 373):
        ex.add(t)
    print('95%:', ex.quantile(0.95))
    print('98%:', ex.quantile(0.98))

# vim: tabstop=4 shiftwidth=4 expandtab
