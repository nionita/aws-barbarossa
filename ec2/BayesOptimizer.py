from skopt import Optimizer, expected_minimum
from skopt.space import Integer
# from sklearn.externals.joblib import Parallel, delayed

# Bayes optimization with Skopt
# We have to optimize a stochastic function of n integer parameters,
# but we can only get noisy results from measurements

'''
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
        if type(config.pscale) == list:
            for pi, si in zip(config.pinits, config.pscale):
                x0.append(pi)
                start = pi - si
                end   = pi + si
                dimensions.append(Integer(start, end))
        else:
            for pi, si, ei in zip(config.pinits, config.pmin, config.pmax):
                x0.append(pi)
                dimensions.append(Integer(si, ei))
        # print('Initial', x0, 'dims', dimensions)
        # When we start to use the regressor, we should have enough random points
        # for a good space exploration
        n_initial_points = max(self.msteps // 10, len(dimensions))
        self.optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=config.regressor,
                n_initial_points=n_initial_points)
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
            xm, _ = expected_minimum(last)
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
