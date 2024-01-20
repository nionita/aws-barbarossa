import warnings
import math
import random
import numpy as np
from skopt import Optimizer
from skopt.utils import expected_minimum
from skopt.space import Integer
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def make_optimizer(config, f, save, status):
    return BayesOptimizer(config, f, save=save, status=status)

'''
Bayes optimization with Skopt
We have to optimize a stochastic function of n integer parameters,
but we can only get noisy results from measurements
'''
class BayesOptimizer:
    def __init__(self, config, f, save=None, status=None):
        self.check_config(config)
        self.config = config
        self.save = save
        local_config = {}
        # self.n_jobs = self.config.n_jobs
        if self.config.old_type:
            self.pnames  = self.config.pnames
            self.msteps  = self.config.msteps
            self.base    = self.config.pinits
            self.in_real = self.config.in_real
            # All dimensions are integers in our case, but we can optimize in real (more noise)
            # Here we will consider the scale as the +/- range from the initial value
            dimensions = []
            x0 = []
            # Dimensions will be automatically normalized by skopt
            if type(self.config.pscale) == list:
                for pi, si in zip(self.config.pinits, self.config.pscale):
                    x0.append(pi)
                    start = pi - si
                    end   = pi + si
                    if self.in_real:
                        dimensions.append(Real(start, end))
                    else:
                        dimensions.append(Integer(start, end))
            else:
                for pi, si, ei in zip(self.config.pinits, self.config.pmin, self.config.pmax):
                    x0.append(pi)
                    if self.in_real:
                        dimensions.append(Real(si, ei))
                    else:
                        dimensions.append(Integer(si, ei))
            regressor   = self.config.regressor
            normalize_y = self.config.normalize
            fix_noise   = self.config.fix_noise
            games       = self.config.games
            elo         = self.config.elo
            isotropic   = self.config.isotropic
            nu          = self.config.nu
            ropoints    = self.config.ropoints
            acq_func    = self.config.acq_func
            simul       = self.config.simul
            texel       = self.config.texel
        else:
            self.pnames  = list(map(lambda p: p.name, self.config.optimization.params))
            self.msteps  = self.config.optimization.msteps
            self.base    = list(map(lambda p: p.ini, self.config.optimization.params))
            self.in_real = self.config.optimization.in_real
            # All dimensions are integers in our case, but we can optimize in real (more noise)
            # Here we will consider the scale as the +/- range from the initial value
            dimensions = []
            x0 = []
            # Dimensions will be automatically normalized by skopt
            for param in self.config.optimization.params:
                x0.append(param.ini)
                if self.in_real:
                    dimensions.append(Real(param.min, param.max))
                else:
                    dimensions.append(Integer(param.min, param.max))
            regressor   = self.config.method.params.regressor
            normalize_y = self.config.method.params.normalize
            fix_noise   = self.config.method.params.fix_noise
            games       = self.config.eval.params.games
            elo         = self.config.eval.params.elo
            isotropic   = self.config.method.params.isotropic
            nu          = self.config.method.params.nu
            ropoints    = self.config.method.params.ropoints
            isteps      = self.config.method.params.isteps
            acq_func    = self.config.method.params.acq_func
            simul       = self.config.eval.type == 'simul'
            texel       = self.config.eval.type == 'texel'

        self.is_gp = False
        if regressor == 'GP':
            self.is_gp = True
            # Y normalization: GP assumes mean 0, or otherwise normalize (which seems not to work well,
            # as the sample mean should be taken only for random points - we could calculate the mean
            # ourselves from the initial random points - not done for now)
            # Yet is normalization the way to go...

            # The noise level is fix in our case, as we play a fixed number of games per step
            # (exception: initial / reference point, with noise = 0 - we ignore this)
            # It depends of number of games per step and the loss function (elo/elowish)
            # The formula below is an ELO error approximation for the confidence interval of 95%,
            # which lies by about 2 sigma - we can compute sigma of the error
            assert not (fix_noise and normalize_y), "Fixed noise does't work with normalize"
            if fix_noise:
                noise_level_bounds = 'fixed'
                sigma = 250. / math.sqrt(games)
                if not elo:
                    sigma = sigma * math.log(10) / 400.
                noise_level = sigma * sigma
            else:
                if elo and not normalize_y:
                    noise_level_bounds = (1e-2, 1e5)
                    noise_level = 10
                else:
                    noise_level_bounds = (1e-16, 1e1)
                    noise_level = 0.01
            # Length scales: because skopt normalizes the dimensions automatically, it is unclear
            # how to proceed here, as the kernel does not know about that normalization...
            length_scale_bounds = (1e-4, 1e5)
            length_scale = 1
            # Isotropic or anisotropic kernel
            if not isotropic:
                length_scale = length_scale * np.ones(len(dimensions))
            # Matern kernel with configurable parameter nu (1.5 = once differentiable functions),
            # white noise kernel and a constant kernel for mean estimatation
            # When we normalize y, we do not need the constant kernel part
            if normalize_y:
                kernel = \
                      WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds) \
                    + 1.0 * Matern(nu=nu, length_scale=length_scale, \
                                   length_scale_bounds=length_scale_bounds)
            else:
                kernel = \
                      ConstantKernel() \
                    + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds) \
                    + 1.0 * Matern(nu=nu, length_scale=length_scale, \
                                   length_scale_bounds=length_scale_bounds)
            # We put alpha=0 because we count in the kernel for the noise
            # n_restarts_optimizer is important to find a good fit! (but it costs time)
            rgr = GaussianProcessRegressor(kernel=kernel,
                    alpha=0.0, normalize_y=normalize_y, n_restarts_optimizer=ropoints)
        else:
            rgr = regressor
        # Uniq: maybe not a good idea? Noise is wrong estimated - is it?
        # It should be a parameter
        self.uniq = True
        # When we start to use the regressor, we should have enough random points
        # for a good space exploration
        if isteps == 0:
            n_initial_points = max(self.msteps // 10, len(dimensions) + 1)
        else:
            n_initial_points = isteps
        self.optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=rgr,
                acq_func=acq_func,
                acq_optimizer='sampling',   # without this I got this error:
                                            # grad = self.kernel_.gradient_x(X[0], self.X_train_)
                                            # AttributeError: 'Sum' object has no attribute 'gradient_x'
                n_initial_points=n_initial_points,
                initial_point_generator='lhs', # latin hypercube, just a try
                n_jobs=-1,  # use all cores in the estimator
                model_queue_size=2)
        self.done = False
        if status is None:
            # When we start, we know that the initial point is the reference, mark it with 0
            # Unless for texel or simulation!
            if texel or simul:
                self.theta = None
                self.best = None
                self.xi = []
                self.yi = []
            else:
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
        self.step = len(self.xi)
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
            res = self.step_ask_tell(last)
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
            return self.get_best(last)

    def step_ask_tell(self, last):
        if self.step >= self.msteps:
            self.done = True
            return None
        print('Step:', self.step)
        self.show_kernel()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            x = self.optimizer.ask()
        if self.uniq:
            x = self.uniq_candidate(x, last)
        print('Candidate:', x)
        y = self.func(x, self.base)
        print('Candidate / result:', x, '/', y)
        # The x returned by ask is of type [np.int32] if not in_real, so we have to cast it
        if self.in_real:
            self.xi.append(list(map(float, x)))
        else:
            self.xi.append(list(map(round, x)))
        self.yi.append(y)
        # Here: because we maximize, negate!
        res = self.optimizer.tell(x, -y)
        last_y = self.yi[-1]
        if self.best is None or last_y > self.best:
            self.theta = self.xi[-1]
            self.best = last_y
        print('Theta / value:', self.theta, '/', self.best)
        self.step += 1
        return res

    # Propose best candidates: this is a hard decision, as long the model is inaccurate
    # This one is used when we finish the number of experiment steps
    # We want to take a candidate which is pretty good pretty sure:
    # - we predict the mean and standard deviation (m, s) of all visited points
    # - do the same for the best possible candidate from the model, if not already visited
    # - order all candidates reverse by m - 3 * s
    # - return the best n of them
    # We can do this only if the regressor is a gaussian process!
    def get_best(self, last):
        xm, ym = expected_minimum(last)
        print('Best expected candidate:', xm)
        # Because we maximize: negate!
        print('Best expected value:', -ym)
        if self.is_gp and len(self.optimizer.models) > 0:
            print('Best taken by MS score')
            rgr = self.optimizer.models[-1]
            if not self.in_real:
                xm = list(map(round, xm))
            # candidates = list(set(map(tuple, self.xi + [xm])))
            candidates = list(set(map(tuple, [self.theta, xm])))
            y_mean, y_std = rgr.predict(candidates, return_std=True)
            mso = []
            for c, m, s in zip(candidates, y_mean, y_std):
                # Because we maximize: negate!
                # mso.append((-m - 3 * s, c, -m, s))
                mso.append((-m - s, c, -m, s))
            mso = sorted(mso, reverse=True)
            print('Best estimated candidates:')
            i = 0
            for m3s, c, m, s in mso:
                print(c, ':\tscore:', m3s, '\tmean:', m, '\tstd:', s)
                i = i + 1
                if i >= 3:
                    break
            return mso[0][1]
        else:
            print('Best taken from expected maximum')
            return xm

    # Check and propose uniq candidates
    # When ask gives us an already evaluated point, we don't wand to reevaluate it,
    # and instead take another one, which was not already evaluated
    def uniq_candidate(self, candidate, last):
        if candidate not in self.xi:
            return candidate
        print('Uniq: ask repeated:', candidate, ' -> try another one')
        # Only when we have a valid last result!
        if last is not None:
            # Now try with the best so far by the model:
            xm, _ = expected_minimum(last)
            # Is this necessary?
            if not self.in_real:
                xm = list(map(round, xm))
            if xm not in self.xi:
                return xm
            print('Uniq: already visited:', xm, ' -> try another one')
        # Now try combinations of 2 already visited points
        # They will be in the domain (even after round), because the domain is convex
        visited = random.sample(self.xi, k=len(self.xi))
        for i in range(len(visited)):
            for j in range(i+1, len(visited)):
                x = list(map(lambda pair: (pair[0] + pair[1]) / 2, zip(visited[i], visited[j])))
                if not self.in_real:
                    x = list(map(round, x))
                if x not in self.xi:
                    return x
                print('Uniq: already visited:', x, ' -> try another one')
        # All is already there: return the initial candidate
        return candidate

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

    '''
    We check here all entries on the config that we need, if the config is new style
    The old style checks all required values or delivers defaults for optional parameters
    '''
    def check_config(self, config):
        if config.old_type:
            assert config.check('pscale', vtype=list) \
                or config.check('pmin', vtype=list) and config.check('pmax', vtype=list)
        else:
            assert config.check('method.params.regressor', vtype=str, required=True)
            assert config.check('method.params.normalize', vtype=bool, required=True)
            assert config.check('method.params.fix_noise', vtype=bool, required=True)
            assert config.check('method.params.isotropic', vtype=bool, required=True)
            assert config.check('method.params.nu', vtype=float, required=True)
            assert config.check('method.params.ropoints', vtype=int, required=True)
            assert config.check('method.params.isteps', vtype=int, required=True)
            assert config.check('optimization.params', vtype=list, required=True)
            for param in config.optimization.params:
                assert type(param.name) == str
                assert type(param.ini) == int
                assert type(param.min) == int
                assert type(param.max) == int
            # self.n_jobs = config.n_jobs
            assert config.check('optimization.msteps', vtype=int, required=True)
            assert config.check('optimization.in_real', vtype=bool, required=True)
            assert config.check('eval.type', vtype=str, required=True)
            if config.old_type and config.simul == 0.0 or not config.old_type and config.eval.type == 'selfplay':
                assert config.check('eval.params.games', vtype=int, required=True)
                assert config.check('eval.params.elo', vtype=bool, required=True)

# vim: tabstop=4 shiftwidth=4 expandtab
