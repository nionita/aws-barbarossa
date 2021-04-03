import math
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
        # Dimensions will be automatically normalized by skopt
        # transform_x = 'normalize' if 'X' in config.normalize else 'identity'
        if type(config.pscale) == list:
            for pi, si in zip(config.pinits, config.pscale):
                x0.append(pi)
                start = pi - si
                end   = pi + si
                # dimensions.append(Integer(start, end, transform=transform_x))
                dimensions.append(Integer(start, end))
        else:
            for pi, si, ei in zip(config.pinits, config.pmin, config.pmax):
                x0.append(pi)
                # dimensions.append(Integer(si, ei, transform=transform_x))
                dimensions.append(Integer(si, ei))
        self.is_gp = False
        if config.regressor == 'GP':
            self.is_gp = True
            # Y normalization: GP assumes mean 0, or otherwise normalize (which seems not to work well,
            # as the sample mean should be taken only for random points - we could calculate the mean
            # ourselves from the initial random points - not done for now)
            # Yet is normalization the way to go...
            normalize_y = 'Y' in config.normalize

            # The noise level is fix in our case, as we play a fixed number of games per step
            # (exception: initial / reference point, with noise = 0 - we ignore this)
            # It depends of number of games per step and the loss function (elo/elowish)
            # The formula below is an ELO error approximation for the confidence interval of 95%,
            # which lies by about 2 sigma - we can compute sigma of the error
            if config.fix_noise:
                noise_level_bounds = 'fixed'
                sigma = 250. / math.sqrt(config.games)
                if not config.elo:
                    sigma = sigma * math.log(10) / 400.
                noise_level = sigma * sigma
            else:
                if config.elo and not normalize_y:
                    noise_level_bounds = (1e-2, 1e4)
                    noise_level = 10
                else:
                    noise_level_bounds = (1e-6, 1e0)
                    noise_level = 0.01
            # Length scales: because skopt normalizes the dimensions automatically, it is unclear
            # how to proceed here, as the kernel does not know about that normalization...
            length_scale_bounds = (1e-3, 1e4)
            length_scale = 1
            # Isotropic or anisotropic kernel
            if not config.isotropic:
                length_scale = length_scale * np.ones(len(dimensions))
            # Matern kernel with configurable parameter nu (1.5 = once differentiable functions),
            # white noise kernel and a constant kernel for mean estimatation
            kernel = \
                  ConstantKernel() \
                + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds) \
                + 1.0 * Matern(nu=config.nu, length_scale=length_scale, \
                               length_scale_bounds=length_scale_bounds)
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
        self.show_kernel()
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
        self.show_kernel()
        x = self.optimizer.ask()
        print('Candidate:', x)
        y = self.func(self.config, x, self.base)
        print('Candidate / result:', x, '/', y)
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
        self.step += 1
        return res

    def show_kernel(self):
        if self.is_gp and len(self.optimizer.models) > 0:
            rgr = self.optimizer.models[-1]
            if hasattr(rgr, 'kernel_'):
                print('Kernel:', rgr.kernel_)
            if hasattr(rgr, 'log_marginal_likelihood_value_'):
                print('LML value:', rgr.log_marginal_likelihood_value_)

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
