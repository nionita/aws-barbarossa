import math
import random
import numpy as np
#from skopt import Optimizer
#from skopt.utils import expected_minimum
#from skopt.space import Integer
#from skopt.space import Real
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# We need sqrt(2)
sqrt2 = math.sqrt(2)

'''
Greedy Bayes optimization with sklearn - with self expanding limits
We have to optimize a stochastic function of n integer parameters,
but we can only get noisy results from measurements
'''
class BayesDirectGP:
    def __init__(self, config, f, save=None, status=None):
        self.check_config(config)
        self.config = config
        self.save = save
        local_config = {}
        self.pnames  = list(map(lambda p: p.name, self.config.optimization.params))
        self.msteps  = self.config.optimization.msteps
        self.base    = list(map(lambda p: p.ini, self.config.optimization.params))
        self.std     = np.array(list(map(lambda p: p.std, self.config.optimization.params)))
        # All dimensions are integers in our case, but we can optimize in real (more noise)
        self.in_real = self.config.optimization.in_real
        regressor   = self.config.method.params.regressor
        #self.normalize_y = self.config.method.params.normalize
        fix_noise   = self.config.method.params.fix_noise
        games       = self.config.eval.params.games
        elo         = self.config.eval.params.elo
        self.nu          = self.config.method.params.nu
        self.alpha       = self.config.method.params.alpha
        self.ropoints    = self.config.method.params.ropoints
        self.mopoints    = self.config.method.params.mopoints
        self.isteps      = self.config.method.params.isteps
        simul       = self.config.eval.type == 'simul'
        texel       = self.config.eval.type == 'texel'

        # Y normalization: GP assumes mean 0

        # The noise level is fix in our case, as we play a fixed number of games per step
        # (exception: initial / reference point, with noise = 0 - we ignore this)
        # It depends of number of games per step and the loss function (elo/elowish)
        # The formula below is an ELO error approximation for the confidence interval of 95%,
        # which lies by about 2 sigma - we can compute sigma of the error
        if fix_noise:
            self.noise_level_bounds = 'fixed'
            sigma = 250. / math.sqrt(games)
            if not elo:
                sigma = sigma * math.log(10) / 400.
            self.noise_level = sigma * sigma
        else:
            if elo:
                self.noise_level_bounds = (1e-2, 1e4)
                self.noise_level = 10
            else:
                self.noise_level_bounds = (1e-6, 1e0)
                self.noise_level = 0.01
        self.length_scale_bounds = (1e-3, 1e4)
        # We always use an anisotropic kernel
        self.length_scale = np.ones(len(self.base))
        # When we start to use the regressor, we should have enough random points
        # for a good space exploration
        if self.isteps == 0:
            self.isteps = len(self.base) + 1
        self.done = False
        if status is None:
            # When we start, we know that the initial point is the reference, mark it with 0
            # Unless for texel or simulation!
            if simul or texel:
                self.theta = self.base
                self.best = None
                self.xi = []
                self.yi = []
                self.center = np.array(self.theta, dtype=np.float)
            else:
                self.theta = self.base
                self.best = 0
                self.xi = [ self.base ]
                self.yi = [ 0 ]
                self.center = np.array(self.theta, dtype=np.float)
        else:
            self.theta = status['theta']
            self.best = status['best']
            self.xi = status['xi']
            self.yi = status['yi']
            self.center = np.array(status['center'], dtype=np.float)
        # Because we added the reference result, we have now one measurement more than steps
        self.step = len(self.xi)
        self.func = f
        self.lml = None

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
            'yi': self.yi,
            'center': list(map(float, list(self.center)))
        }

    # Find the models best by sampling and predicting
    # The proposed best will eventually update the center, which is used to generate new samples
    def model_best(self, max_iters=100, max_unchanged=5, optimistic=True):
        # We have the data so far and want to find the best from the current model
        # We optimize by sampling many points and take the maximum from the GP estimates
        best_x = None
        best_y = None
        best_s = None
        unchanged = None
        X = np.array(self.xi, dtype=np.float)
        y = np.array(self.yi, dtype=np.float)
        # We must have 0 mean for GP, so subtract the mean
        y_mean = np.mean(y)
        y = y - y_mean
        # Matern kernel with configurable parameter nu (1.5 = once differentiable functions),
        # white noise kernel and a constant kernel for mean estimatation
        kernel = \
              ConstantKernel() \
            + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds) \
            + 1.0 * Matern(nu=self.nu, length_scale=self.length_scale, \
                           length_scale_bounds=self.length_scale_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False,
                n_restarts_optimizer=self.ropoints).fit(X, y)
        self.lml = gp.log_marginal_likelihood_value_
        self.kernel = gp.kernel_
        for _ in range(max_iters):
            alea = np.random.randn(self.mopoints, len(self.base))
            alea = self.std * alea + self.center
            y_pred, y_std = gp.predict(alea, return_std=True)
            if optimistic:
                y_pred_opt = y_pred + y_std
            else:
                y_pred_opt = y_pred
            i_max = np.argmax(y_pred_opt)
            if best_y is None or y_pred_opt[i_max] > best_y:
                best_y = y_pred_opt[i_max]
                best_x = alea[i_max]
                best_s = y_std[i_max]
                unchanged = 0
            else:
                unchanged += 1
                if unchanged > max_unchanged:
                    break
        # Add back the mean
        return (self.make_list(best_x), best_y + y_mean, best_s)

    # Adjust the center depending on prediction & confidence vs. reality
    # We move towards the candidate with a step proportional to the probability
    # P(x < r|x ~ N(p, s)), where r is reality, p is prediction and s is stddev of prediction
    def adjust_center(self, x_cand, y_pred, y_std, y_real):
        alpha = self.alpha * 0.5 * (1 + math.erf((y_real - y_pred) / y_std / sqrt2))
        print('Adjust alpha:', alpha)
        self.center += alpha * (np.array(x_cand, dtype=np.float) - self.center)

    def random_x(self):
        # We mus generate random value to bootstrap the optimizer
        alea = np.random.randn(1, len(self.base))
        alea = self.std * alea + self.center
        return self.make_list(alea[0])

    def make_list(self, x):
        if self.in_real:
            return list(map(float, list(x)))
        return list(map(round, list(x)))

    '''
    Maximize by gaussian process
    The callback is called after every step
    It is called with the optimizer object and it must return True
    if the optimization has to be stopped immediately
    '''
    def optimize(self, callback):
        done = False
        while not (self.done or done):
            self.opt_step()
            done = callback(self)
        self.show_kernel()
        # Instead of returning the current best, we return the one that the model considers best
        if self.lml is None:
            print('No enough steps for the expected maximum')
            return self.theta
        else:
            return self.get_best()

    def opt_step(self):
        if self.step >= self.msteps:
            self.done = True
            return None
        print('Step:', self.step)
        self.show_kernel()
        if self.step >= self.isteps:
            x, y_pred, y_std = self.model_best(optimistic=False)
        else:
            x = self.random_x()
            y_pred = None
        # x = self.uniq_candidate(x, last)
        x = list(x)
        print('Candidate:', x)
        if y_pred is not None:
            print('Predicted y:', y_pred, 'with std', y_std)
        y = self.func(x, self.base)
        print('Candidate / result:', x, '/', y)
        if y_pred is not None:
            self.adjust_center(x, y_pred, y_std, y)
        self.xi.append(x)
        self.yi.append(y)
        last_y = self.yi[-1]
        if self.best is None or last_y > self.best:
            self.theta = self.xi[-1]
            self.best = last_y
        print('Theta / value:', self.theta, '/', self.best)
        self.step += 1

    # Propose best candidates: this is a hard decision, as long the model is inaccurate
    # This one is used when we finish the number of experiment steps
    # We want to take a candidate which is pretty good pretty sure:
    # - we predict the mean and standard deviation (m, s) of all visited points
    # - do the same for the best possible candidate from the model, if not already visited
    # - order all candidates reverse by m - 3 * s
    # - return the best n of them
    def get_best(self):
        print('Best taken from expected maximum')
        xm, ym, ys = self.model_best(optimistic=False)
        print('Best expected candidate:', xm)
        print('Best expected value:', ym, 'with std', ys)
        print('Model center:', list(self.center))
        return xm

    # Check and propose uniq candidates
    # When ask gives us an already evaluated point, we don't wand to reevaluate it,
    # and instead take another one, which was not already evaluated
    def uniq_candidate(self, candidate, last):
        if candidate not in self.xi:
            return candidate
        print('Uniq: ask repeated:', candidate, ' -> try another one')
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
        if self.lml is not None:
            print('Kernel:', self.kernel)
            print('LML value:', self.lml)
            print('Center:', self.center)

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
    We check here all entries of the config that we need, if the config is new style
    '''
    def check_config(self, config):
        if config.old_type:
            raise "BayesDirectGP needs new style config (yaml)"
        else:
            # assert config.check('method.params.normalize', vtype=bool, required=True)
            assert config.check('method.params.fix_noise', vtype=bool, required=True)
            assert config.check('method.params.nu', vtype=float, required=True)
            assert config.check('method.params.alpha', vtype=float, required=True)
            assert config.check('method.params.ropoints', vtype=int, required=True)
            assert config.check('method.params.mopoints', vtype=int, required=True)
            assert config.check('method.params.isteps', vtype=int, required=True)
            assert config.check('optimization.params', vtype=list, required=True)
            for param in config.optimization.params:
                assert type(param.name) == str
                assert type(param.ini) == int
                assert type(param.std) == float or type(param.std) == int
            assert config.check('optimization.msteps', vtype=int, required=True)
            assert config.check('optimization.in_real', vtype=bool, required=True)
            assert config.check('eval.type', vtype=str, required=True)
            if config.eval.type == 'selfplay':
                assert config.check('eval.params.games', vtype=int, required=True)
                assert config.check('eval.params.elo', vtype=bool, required=True)

# vim: tabstop=4 shiftwidth=4 expandtab
