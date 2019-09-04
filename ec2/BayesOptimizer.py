from skopt import Optimizer
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
        for i in range(len(config.pinits)):
            start = config.pinits[i] - config.pscale[i]
            end   = config.pinits[i] + config.pscale[i]
            dimensions.append(Integer(start, end))
        self.optimizer = Optimizer(dimensions=dimensions, base_estimator='gp')
        self.done = False
        if status is None:
            self.theta = None
            self.best = None
            self.xi = []
            self.yi = []
        else:
            self.theta = status['theta']
            self.best = status['best']
            self.xi = status['xi']
            self.yi = status['yi']
            for x, y in zip(self.xi, self.yi):
                # Here: becasue we maximize (wile the bayes optimizer minimize), negate!
                self.optimizer.tell(x, -y)
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
        while not (self.done or done):
            self.step_ask_tell()
            done = callback(self)
        return self.theta

    def step_ask_tell(self):
        if self.step >= self.msteps:
            self.done = True
            return
        print('Step:', self.step)
        x = self.optimizer.ask()
        print('Params:', x)
        y = self.func(x, self.base, self.config)
        print('Result:', y)
        # The x returned by ask is of type [np.int32], so we have to cast it
        self.xi.append(list(map(int, x)))
        self.yi.append(y)
        # Here: because we maximize, negate!
        self.optimizer.tell(x, -y)
        last = self.yi[-1]
        if self.best is None or last > self.best:
            self.theta = self.xi[-1]
            self.best = last
        print('Theta / value:', self.theta, '/', self.best)
        self.step += 1

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
