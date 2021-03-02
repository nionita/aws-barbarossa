import numpy as np
from skopt import Optimizer
from skopt.utils import expected_minimum
from skopt.space import Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

'''
Bayes optimization with Skopt
We have to optimize a stochastic function of n integer parameters,
but we can only get noisy results from measurements
'''
class BayesOptimizer:
    def __init__(self, config, f, save=None, status=None):
        self.config = config
        self.pnames = config.pnames
        self.msteps = config.msteps
        # self.n_jobs = config.n_jobs
        self.save = save
        self.base = config.pinits
        # All dimensions are integers in our case
        # Here we will consider the scale as the +/- range from the initial value
        dimensions = []
        x0 = []
        assert type(config.pscale) == list or type(config.pmin) == list and type(config.pmax) == list
        transform_x = 'normalize' if 'X' in config.normalize else 'identity'
        if type(config.pscale) == list:
            for pi, si in zip(config.pinits, config.pscale):
                x0.append(pi)
                start = pi - si
                end   = pi + si
                dimensions.append(Integer(start, end, transform=transform_x))
        else:
            for pi, si, ei in zip(config.pinits, config.pmin, config.pmax):
                x0.append(pi)
                dimensions.append(Integer(si, ei, transform=transform_x))
        self.is_gp = False
        if config.regressor == 'GP':
            self.is_gp = True
            # GPR with Matern isotropic kernel, white noise and a constant kernel for mean estimatation
            # Matern nu as config parameter (1.5 = once differentiable functions)
            # An anisotropic kernel could be slightly better, but we have much more hyperparameters,
            # which may make the fit worse (for a given number of samples)
            # When we normalize X, we need other limits for the kernel parameters
            if 'X' in config.normalize:
                kernel = 1.0 * Matern(nu=config.nu, length_scale=0.1, length_scale_bounds=(1e-2, 1e1)) \
                         + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e+0)) \
                         + ConstantKernel()
            else:
                kernel = 1.0 * Matern(nu=config.nu, length_scale=1000, length_scale_bounds=(1e0, 1e4)) \
                         + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e+1)) \
                         + ConstantKernel()
            # Y normalization
            # GP assumes mean 0, or otherwise normalize, which seems not to work well,
            # as the sample mean should be taken only for random points
            # We could calculate the mean ourselves from the initial random points
            normalize_y = 'Y' in config.normalize
            # We put alpha=0 because we count in the kernel for the noise
            # n_restarts_optimizer is important to find a good fit! (but it costs time)
            rgr = GaussianProcessRegressor(kernel=kernel,
                    alpha=0.0, normalize_y=normalize_y, n_restarts_optimizer=config.ropoints)
        else:
            rgr = config.regressor
        # When we start to use the regressor, we should have enough random points
        # for a good space exploration
        if config.isteps == 0:
            n_initial_points = max(self.msteps // 10, len(dimensions) + 1)
        else:
            n_initial_points = config.isteps
        self.optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=rgr,
                acq_func=config.acq_func,
                acq_optimizer='sampling',   # without this I got this error:
                                            # grad = self.kernel_.gradient_x(X[0], self.X_train_)
                                            # AttributeError: 'Sum' object has no attribute 'gradient_x'
                n_initial_points=n_initial_points,
                model_queue_size=2)
        self.done = False
        if status is None:
            # When we start, we know that the initial point is the reference, mark it with 0
            self.theta = x0
            self.best = 0
            self.xi = [ x0 ]
            self.yi = [ 0 ]
            self.optimizer.tell(x0, 0)
        else:
            self.theta = status['theta']
            self.best = status['best']
            self.xi = status['xi']
            self.yi = status['yi']
            # Here: because we maximize (while the bayes optimizer minimizes), negate!
            self.optimizer.tell(self.xi, list(map(lambda y: -y, self.yi)))
        # Because we added the reference result, we have now one measurement more than steps
        self.step = len(self.xi) - 1
        self.func = f

    '''
    Serialize (JSON) the changing parts of our optimizer.
    We save only what is needed to restore the state.
    For example we do not save the step, as this can be restored from the length of the xi.
    We also do not save the bayesian optimizer itself, but reconstruct it
    from the xi and yi that we save.
    '''
    def get_status(self):
        return {
            'theta': self.theta,
            'best': self.best,
            'xi': self.xi,
            'yi': self.yi
        }

    '''
    Maximize by gaussian process
    The callback is called after every step
    It is called with the BayesOptimizer object and it must return True
    if the optimization has to be stopped immediately
    '''
    def optimize(self, callback):
        done = False
        last = None
        while not (self.done or done):
            res = self.step_ask_tell()
            # print('Type res:', type(res))
            if res:
                last = res
            done = callback(self)
        # Instead of returning the current best, we return the one that the model considers best
        if last is None or not last.models:
            print('No models to calculate the expected maximum')
            return self.theta
        else:
            print('Best taken from expected maximum')
            xm, ym = expected_minimum(last)
            print('Best expected x:', xm)
            # Because we maximize: negate!
            print('Best expected y:', -ym)
            return xm

    def step_ask_tell(self):
        if self.step >= self.msteps:
            self.done = True
            return None
        print('Step:', self.step)
        x = self.optimizer.ask()
        print('Params:', x)
        y = self.func(self.config, x, self.base)
        print('Result:', y)
        # The x returned by ask is of type [np.int32], so we have to cast it
        self.xi.append(list(map(int, x)))
        self.yi.append(y)
        # Here: because we maximize, negate!
        res = self.optimizer.tell(x, -y)
        last = self.yi[-1]
        if last > self.best:
            self.theta = self.xi[-1]
            self.best = last
        print('Theta / value:', self.theta, '/', self.best)
        if self.is_gp and len(self.optimizer.models) > 0:
            rgr = self.optimizer.models[-1]
            if hasattr(rgr, 'kernel_'):
                print('Kernel:', rgr.kernel_)
            if hasattr(rgr, 'log_marginal_likelihood_value_'):
                print('LML value:', rgr.log_marginal_likelihood_value_)
        self.step += 1
        return res

    def report(self, vec, title=None, file='report.txt'):
        if title is None:
            title = 'Current best:'
        if file is None:
            print(title)
            for n, v in zip(self.pnames, list(vec)):
                print(n, '=', v)
        else:
            with open(file, 'w', encoding='utf-8') as repf:
                print(title, file=repf)
                for n, v in zip(self.pnames, list(vec)):
                    print(n, '=', v, file=repf)

# vim: tabstop=4 shiftwidth=4 expandtab
