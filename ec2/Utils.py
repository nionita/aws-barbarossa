import numpy as np

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

# vim: tabstop=4 shiftwidth=4 expandtab
