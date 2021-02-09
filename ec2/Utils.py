import threading
from math import sqrt, log, exp
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

# Some constants
eps = 1e-6
qdecay = 0.95

'''
Single games duration is modelled as an exponential distribution, but in one chunk
we run a few games (tipically 10 or 20), so the total duration of one chunk follows
an Erlang distribution, which has 2 parameters, k (shape) and mu (mean of one game duration)

There is also a probability that the game does not terminate at all (e.g. because software bugs)
which we want to estimate and use in our calculation of the cost. Because of this non-termination
probability we must use a timeout in our play function.

Our goal is to minimize the cost of a timeout, by choosing the timeout such that we minimize
the total time loss.

Because we must have only one estimator for all threads, we need locking
'''
class TimeEstimator():
    def __init__(self):
        self.__inited = False
        self.__lock = threading.Lock()

    def init(self, k=10, mu=60, probto=0.02, debug=False):
        assert k > 1 and mu > 0 and probto > 0 and probto < 1
        with self.__lock:
            if not self.__inited:
                self.__inited = True
                self.n = 0
                self.s = 0
                self.k = k
                self.mu = mu
                self.timeouts = 0
                self.probto = probto
                self.q = 0.01
                self.messages = []
                self.debug = debug

    def __add(self, v):
        self.n += 1
        self.s += v

    def mean(self):
        self.__check_inited()
        with self.__lock:
            return self.__mean()

    def get_messages(self):
        self.__check_inited()
        with self.__lock:
            ms = self.messages
            self.messages = []
            return ms

    def __check_inited(self):
        if not self.__inited:
            raise RuntimeError('Not initialized')

    # This is the mean per chunk
    # To be used only internal after inited checked!
    def __mean(self):
        if self.n == 0:
            return self.mu * self.k
        else:
            return self.s / self.n

    def normal(self, duration):
        self.__check_inited()
        with self.__lock:
            self.__add(duration)
            self.__adjust_q()

    '''
    When aborted (timeout), we must re-estimate p and q
    '''
    def aborted(self, timeout):
        self.__check_inited()
        with self.__lock:
            # Estimated duration in case of abort
            # Even when we abort: because of memorylessness
            # we must expect that it would have taken that more time
            edur = timeout + self.__mean()
            if self.debug:
                self.messages.append('Estimated duration of timeout game: {}'.format(edur))
            # This must be corrected: we might have aborted an endless game
            ced = self.__correct_aborted_game_duration(edur)
            if self.debug:
                self.messages.append('Corrected estimated duration of timeout game: {}'.format(ced))
            self.__add(ced)
            self.timeouts += 1
            self.__adjust_q()

    # Adjust the probability of endless games: either the rest of the
    # sample abort probability, if it is greater than our timeout probability,
    # or a fraction of the previous one, if not (kind of estimate...)
    def __adjust_q(self):
        pq = self.timeouts / self.n
        if pq >= self.probto:
            self.q = pq - self.probto
        else:
            self.q = self.q * qdecay
        if self.debug:
            self.messages.append('New probability of endless games: {}'.format(self.q))

    '''
    Given estimated duration t, find x such that:
    - with prob q / (p + q): mean = s / n, i.e. aborted endless games do not affect statistics
    - with prob p / (p + q): mean = (s + t) / (n + 1), i.e. add estimated duration
    So we add always, but a corrected estimated duration
    If the timeout happens on the very first chunk played, the calculation simplifies a bit
    '''
    def __correct_aborted_game_duration(self, edur):
        pq = self.probto + self.q
        if self.n > 0:
            t1 = (self.n + 1) / self.n * self.s * self.q
            t2 = (self.s + edur) * self.probto
            return (t1 + t2) / pq - self.s
        else:
            return (self.mu * self.k * self.q + edur * self.probto) / pq

    def timeout(self):
        self.__check_inited()
        with self.__lock:
            return self.__timeout()

    def prob_of_timeout(self, timeout):
        self.__check_inited()
        with self.__lock:
            return self.__calc_p(timeout)

    def get_endless_proba(self):
        self.__check_inited()
        with self.__lock:
            return self.q

    def get_total_timeouts(self):
        self.__check_inited()
        with self.__lock:
            return self.timeouts

    '''
    Calculate timeout iterative by doubling and halving intervals
    We want timeout(p0) when we know p(timeout) = p0
    Our equation is: p(timeout) - p0 = 0, where 0 < p0 < 1
    So we want the root of f(x) = p(x) - p0

    Let's take a = 0; we know: p(0) = 1 (we timeout for sure when we set timeout to 0)
    That means: f(a) = 1 - p0 > 0
    Then we set b = self.mean() and as long as f(b) > 0, double b
    When we stop, we have f(a) > 0 and f(b) < 0
    Now we can halve the interval to find the root with some precision
    
    This could be further optimized, e.g. instead of halving:

    c = (b * fa - a * fb) / (fa - fb)

    where we already considered the negative sign of fb
    '''
    def __timeout(self):
        a = 0
        b = self.__mean()
        fb = self.__solver(b)
        while fb > 0:
            a = b
            b = 2 * b
            fb = self.__solver(b)
        # Now we have: f(a) > 0 and f(b) < 0
        done = False
        while not done and (b - a >= 1):
            c = (a + b) / 2
            fc = self.__solver(c)
            if fc > eps:
                a = c
            elif fc < -eps:
                b = c
            else:
                done = True
        return a

#    # Calculate timeout by Newton method - not working yet
#    def __timeout(self):
#        # Our initial guess is as if we had an exponential distribution
#        # with same mean as the chunk
#        t0 = - log(1 - self.probto) * self.__mean()
#        t1 = t0 - (self.__calc_p(t0) - self.probto) / self.__calc_p_der(t0)
#        return t1

    def __solver(self, x):
        return self.__calc_p(x) - self.probto

    '''
    Calculate the error function for the Erlang CDF
    (siehe https://en.wikipedia.org/wiki/Erlang_distribution)
    Use only after inited
    '''
    def __calc_p(self, timeout):
        lx = timeout * self.k / self.__mean()
        return exp(-lx) * self.__sum(lx)

    # This sum is needed for CDF and its derivative calculations
    def __sum(self, h):
        nif = 1 # factor 1 / n!
        hn = 1  # factor h ^ n
        s = 1   # sum value so far
        for n in range(1, self.k):
            nif = nif / n
            hn = hn * h
            s += nif * hn
        return s

#    '''
#    Calculate the derivative of the error function for the Erlang CDF
#    This is used to apply Newton formula to calculate the timeout patameter
#    from a given desired probability of timeout
#    '''
#    def __calc_p_der(self, timeout):
#        lx  = timeout * self.k / self.__mean()
#        elx = exp(-lx)
#        slx = self.__sum(lx)
#        return elx * (-lx * slx + self.sumd(lx))
#
#    # The derivative of the sum relative to h
#    def sumd(self, h):
#        nif = 1 # factor 1 / n!
#        hn = 1  # factor h ^ n
#        s = 0   # sum value so far
#        for n in range(1, self.k):
#            nif = nif / n
#            hn = hn * h
#            s += nif * n * hn
#        return s / h

if __name__ == '__main__':
    te = TimeEstimator()
    te.init(k=20, mu=10, probto=0.02)
    for t in (419, 342, 399, 428, 407, 399):
        te.normal(t)
    mean = te.mean()
    print('Mean', mean)
    for to in (0, mean, 500, 700, 900):
        print('Prob of TO for ', to, ':', te.prob_of_timeout(to))
    t = te.timeout()
    print('Timeout 1:', t)
    print('Prob of TO for ', t, ':', te.prob_of_timeout(t))
    te.aborted(t)
    t = te.timeout()
    print('Timeout 2:', t)
    print('Prob of TO for ', t, ':', te.prob_of_timeout(t))
    ms = te.get_messages()
    for mes in ms:
        print('Time estimator:', mes)

# vim: tabstop=4 shiftwidth=4 expandtab
