import warnings
import math
import random
import scipy.optimize

def make_optimizer(config, f, save, status):
    return NelderMead(config, f, save=save, status=status)

def dumb_callback(x):
    print('-> Dumb Callback:', x)

'''
Optimization with Nelder Mead method (simplex)
'''
class NelderMead:
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
            if type(self.config.pscale) == list:
                for pi, si in zip(self.config.pinits, self.config.pscale):
                    x0.append(pi)
                    start = pi - si
                    end   = pi + si
                    dimensions.append((start, end))
            else:
                for pi, si, ei in zip(self.config.pinits, self.config.pmin, self.config.pmax):
                    x0.append(pi)
                    dimensions.append((si, ei))
            games       = self.config.games
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
            for param in self.config.optimization.params:
                x0.append(param.ini)
                dimensions.append((param.min, param.max))
            games       = self.config.eval.params.games
            simul       = self.config.eval.type == 'simul'
            texel       = self.config.eval.type == 'texel'

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
        def minfun(x):
            return -f(x)
        self.func = minfun
        self.x0 = x0
        self.dimensions = dimensions
        # self.method = 'Nelder-Mead'
        self.method = 'Powell'

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
        print(f'Dimensions: {len(self.x0)}')
        if self.method == 'Nelder-Mead':
            options = { 'disp': True, 'maxfev': self.msteps, 'adaptive': True, 'xtol': 0.49 }
        else:
            options = { 'disp': True, 'maxiter': self.msteps, 'xtol': 0.49 }
        return scipy.optimize.minimize(self.func, self.x0, method=self.method, options=options, callback=dumb_callback)

    def report(self, vec, title=None, file='report.txt'):
        if title is None:
            title = 'Current best:'
        if file is None:
            print(title)
            print(vec)
            for n, v in zip(self.pnames, list(vec.x)):
                print(n, '=', v)
        else:
            with open(file, 'w', encoding='utf-8') as repf:
                print(title, file=repf)
                for n, v in zip(self.pnames, list(vec.x)):
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
            assert config.check('optimization.params', vtype=list, required=True)
            for param in config.optimization.params:
                assert type(param.name) == str
                assert type(param.ini) == int
                assert type(param.min) == int
                assert type(param.max) == int
            # self.n_jobs = config.n_jobs
            assert config.check('eval.type', vtype=str, required=True)
            if config.old_type and config.simul == 0.0 or not config.old_type and config.eval.type == 'selfplay':
                assert config.check('eval.params.games', vtype=int, required=True)
                assert config.check('eval.params.elo', vtype=bool, required=True)

# vim: tabstop=4 shiftwidth=4 expandtab
