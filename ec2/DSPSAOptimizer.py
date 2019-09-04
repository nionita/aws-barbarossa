import math
import numpy as np

from Utils import Statistics

# DSPSA
# We have to optimize a stochastic function of n integer parameters,
# but we can only get noisy results from measurements

'''
Implementation of DSPSA
'''
class DSPSAOptimizer:
    def __init__(self, config, f, save=None, status=None):
        self.config = config
        self.pnames = config.pnames
        self.smalla = config.laststep * math.pow(1.1 * config.msteps + 1, config.alpha)
        self.biga = 0.1 * config.msteps
        self.alpha = config.alpha
        self.msteps = config.msteps
        if config.pscale is None:
            self.scale = None
        else:
            self.scale = np.array(config.pscale, dtype=np.float32)
        self.rend = config.rend
        self.save = save
        if status is None:
            self.done = False
            self.step = 0
            self.since = 0
            self.theta = np.array(config.pinits, dtype=np.float32)
            self.statistics = Statistics(self.theta)
        else:
            self.done = status['done']
            self.step = status['step']
            self.since = status['since']
            self.theta = np.array(status['theta'], dtype=np.float32)
            self.statistics = status['statistics']
        if self.rend is not None:
            self.rtheta = np.rint(self.theta)
        self.func = f

    # Generate next 2 points for gradient calculation
    def random_direction(self):
        p = self.theta.shape[0]
        delta = 2 * np.random.randint(0, 2, size=p) - np.ones(p, dtype=np.int)
        if self.scale is not None:
            delta = delta * self.scale
        pi = np.floor(self.theta) + np.ones(p, dtype=np.float32) / 2
        tp = np.rint(pi + delta / 2)
        tm = np.rint(pi - delta / 2)
        return tp, tm, delta

    '''
    Serialize (JSON) the changing parts of the DSPSA object (status)
    '''
    def get_status(self):
        return {
            'step': self.step,
            'since': self.since,
            'theta': self.theta.tolist(),
            'done': self.done,
            'statistics': self.statistics.get_status()
        }

    '''
    Optimize by the classical DSPSA method
    The callback is called after every step
    It is called with the DSPSA object and it must return True
    if the optimization has to be stopped immediately
    '''
    def optimize(self, callback):
        done = False
        while not (self.done or done):
            self.step_dspsa()
            done = callback(self)
        return self.theta

    def step_dspsa(self):
        if self.step >= self.msteps:
            self.done = True
            return
        print('Step:', self.step)
        tp, tm, delta = self.random_direction()
        print('Params +:', tp)
        print('Params -:', tm)
        df = self.func(tp, tm, self.config)
        gk = df / delta
        ak = self.smalla / math.pow(1 + self.biga + self.step, self.alpha)
        agk = ak * gk
        print('df:', df, 'ak:', ak)
        print('gk:', gk)
        print('ak * gk:', agk)
        # Here: + because we maximize!
        self.theta += agk
        self.statistics.step(agk)
        print('theta:', self.theta)
        print('pressure:', self.statistics.pressure())
        print('relevance:', self.statistics.relevance())
        self.step += 1
        if self.rend is not None:
            ntheta = np.rint(self.theta)
            if np.all(ntheta == self.rtheta):
                self.since += 1
                if self.since >= self.rend:
                    print('Rounded parameters unchanged for', self.rend, 'steps')
                    self.done = True
                    return
            else:
                self.rtheta = ntheta
                self.since = 0

    '''
    Momentum optimizer with friction
    beta1 + beta2 <= 1
    '''
    def momentum(self, f, config, beta1=0.8, beta2=0.1):
        p = self.theta.shape[0]
        gm = np.zeros(p, dtype=np.float32)
        for k in range(self.msteps):
            if k % 1 == 0:
                print('Step:', k)
            tp, tm, delta = self.random_direction()
            df = f(tp, tm, config)
            gk = df / delta
            gm = gm * beta1 + gk * beta2
            # We wouldn't need biga, as first steps are biased towards 0 anyway
            ak = self.smalla / math.pow(1 + self.biga + k, self.alpha)
            if k % 1 == 0:
                print('df:', df, 'ak:', ak)
            # Here: + because we maximize!
            self.theta += ak * gm
            if k % 1 == 0:
                print('theta:', self.theta)
            if k % 10 == 0:
                self.report(self.theta)
        return np.rint(self.theta)

    '''
    Adadelta should maintain different learning rates per dimension, but in our
    case all dimensions would have equal rates, because in every step only
    the sign is different, and we can't break the simmetry.
    Also, our gradient is just an estimate.
    To deal with these problems we maintain an average gradient and work with it
    as if it would be the current one
    '''
    def adadelta(self, f, config, mult=1, beta=0.9, gamma=0.9, niu=0.9, eps=1E-8):
        print('scale:', self.scale)
        p = self.theta.shape[0]
        gm = np.zeros(p, dtype=np.float32)
        eg2 = np.zeros(p, dtype=np.float32)
        ed2 = np.zeros(p, dtype=np.float32)
        for k in range(self.msteps):
            print('Step:', k)
            tp, tm, delta = self.random_direction()
            print('plus:', tp)
            print('mius:', tm)
            df = f(tp, tm, config)
            gk = df / delta
            # niu is for friction
            gm = (beta * gm + (1 - beta) * gk) * niu
            eg2 = gamma * eg2 + (1 - gamma) * gm * gm
            dtheta = np.sqrt((ed2 + eps) / (eg2 + eps)) * gm
            ed2 = gamma * ed2 + (1 - gamma) * dtheta * dtheta
            # Here: + because we maximize!
            self.theta += mult * dtheta
            print('df:', df, 'gm norm:', np.linalg.norm(gm), 'dt norm:', np.linalg.norm(dtheta))
            print('theta:', self.theta)
            if k % 10 == 0:
                self.report(self.theta)
        return np.rint(self.theta)

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
