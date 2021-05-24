import datetime
import random
import math
import subprocess
import re
import os
import os.path
import shutil
import concurrent.futures
from concurrent.futures import FIRST_COMPLETED

from Utils import TimeEstimator

# Some constants
# To calculate Elo differences from game results
eps0 = 1e-2
eps1 = 1 - eps0

# To calculate node numbers for timeouts
# For 20000 nodes we need ~50 seconds in 10 games
nodes_depth_1 = 200
branching_factor = 2
nodes_constant = 4000

'''
Transforming a result in an elo difference - here frac is in the open interval (0, 1).
More than 0.5 means the first player is better.
We must have guards (eps0, eps1) for the excentric results 0 and 1, which can occur for a few games
'''
def elo(frac):
    frac = max(eps0, min(eps1, frac))
    return 400 * math.log10(frac / (1 - frac))

'''
The old function is essentially the same, but on a different scale
'''
def elowish(frac):
    frac = max(eps0, min(eps1, frac))
    return -math.log(1 / frac - 1)

resre = re.compile(r'End result')
wdlre = re.compile('[() ,]')

# We prepare the config only once
config = None

class Config:
    def __init__(self, cf):
        self.old_type = cf.old_type
        if self.old_type:
            self.name       = cf.name
            self.simul      = cf.simul
            self.pnames     = cf.pnames
            self.base       = cf.base
            self.selfplay   = cf.selfplay
            self.playdir    = cf.playdir
            self.ipgnfile   = cf.ipgnfile
            self.ipgnlen    = cf.ipgnlen
            self.depth      = cf.depth
            self.nodes      = cf.nodes
            self.play_chunk = cf.play_chunk
            self.games      = cf.games
            self.parallel   = cf.parallel
            self.probto     = cf.probto
            self.elo        = cf.elo
        else:
            self.name       = cf.name
            self.simul      = cf.eval.type == 'simul'
            self.pnames     = list(map(lambda param: param.name, cf.optimization.params))
            self.base       = cf.eval.params.base
            self.selfplay   = cf.eval.params.selfplay
            self.playdir    = cf.eval.params.playdir
            self.ipgnfile   = cf.eval.params.ipgnfile
            self.ipgnlen    = cf.eval.params.ipgnlen
            self.depth      = cf.eval.params.depth
            self.nodes      = cf.eval.params.nodes
            self.play_chunk = cf.eval.params.play_chunk
            self.games      = cf.eval.params.games
            self.parallel   = cf.eval.params.parallel
            self.probto     = cf.eval.params.probto
            self.elo        = cf.eval.params.elo

def init_config(cf):
    config = Config(cf)

# When we work with a base param (like in bayesian optimization)
# we want to be able to give a fixed configuration (in which we can have also
# other parameters as the optimized ones set different as from the default
# self playing program)
base_file = None

# When we have a ready base param file, this is given in the config
# and we probably must copy it to the playdir
def copy_base_file(source, dest):
    print('We have a base file:', source)
    base_file = os.path.basename(source)
    if os.path.dirname(source) != dest:
        print('Copy', source, 'to', dest)
        # We copy it only when it's not already there
        shutil.copy(source, dest)

# We play random openings from the given pgn file, and then more at once,
# for efficiency reason, so we must skip a random number of games before we start
# to play the given number of games
def random_skip(pgnlen, games):
    return random.randint(0, pgnlen - games + 1)

# We will estimate the timeout based on observed game durations
timeEst = TimeEstimator()

# Play a match with a given number of games between 2 param sets
# Player 1 is theta+ or the candidate
# Player 2 is theta- or the base param set
# Even if we optimize in real parameter, play will round the parameters
def play(tp, tm=None):
    # If the parameters are real, convert them:
    tp = list(map(round, tp))
    if config.simul:
        return rosenbrock(tp, config.simul)
    os.chdir(config.playdir)
    pla = config.name + '-playerp.cfg'
    with open(pla, 'w', encoding='utf-8') as plf:
        for p, v in zip(config.pnames, tp):
            plf.write('%s=%d\n' % (p, v))
    if tm is None:
        if base_file is None:
            copy_base_file(config.base, config.playdir)
        base = base_file
    else:
        # If the parameters are real, convert them:
        tm = list(map(round, tm))
        base = config.name + '-playerm.cfg'
        with open(base, 'w', encoding='utf-8') as plf:
            for p, v in zip(config.pnames, tm):
                plf.write('%s=%d\n' % (p, v))
    args = [config.selfplay, '-m', config.playdir, '-a', pla, '-b', base,
            '-i', config.ipgnfile, '-d', str(config.depth)]
    if config.nodes:
        args.extend(['-n', str(config.nodes)])
    games = config.play_chunk or config.games // config.parallel
    total_starts = config.games // games

    # Calculate the timeout:
    # This will be calculated based on nodes / move, everything empirical, of course
    if config.nodes:
        nodes = config.nodes
    else:
        nodes = nodes_depth_1 * branching_factor ** config.depth
    mu = nodes * games / nodes_constant

    # Initialize the time estimator - it will remain initialized between calls of play
    timeEst.init(k=games, mu=mu, probto=config.probto)
    timeout = int(timeEst.timeout())
    endless = timeEst.get_endless_proba()
    tto     = timeEst.get_total_timeouts()

    print('Play: starting', total_starts, 'times with', games, \
            'games each, timeout =', timeout, 'endless proba =', endless, \
            'timeouts', tto)

    # We could calculate the timeout individually before every chunk play by passing
    # a callback to worker function (play_one) - but the timeout calculation is heavy, so
    # we would have to define a kind of cache and cache expiration and calculate the
    # timeout only e.g. every 10 steps

    # Play the games in parallel, collecting and consolidating the results
    w, d, l = 0, 0, 0
    succ_ends = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel) as pool:
        pending = set([pool.submit(play_one, config, list(args), games, timeout=timeout, id=i) \
                          for i in range(total_starts)])
        cid = total_starts
        while len(pending) > 0:
            done, not_done = concurrent.futures.wait(pending, return_when=FIRST_COMPLETED)
            new_starts = set()
            for future in done:
                success = False
                data = future.result()
                if 'exception' in data:
                    print('Exception in game thread {}: {}'.format(data['id'], data['exception']))
                elif 'timeout' in data:
                    timeout = data['timeout']
                    # The time duration estimator takes care of endless games
                    timeEst.aborted(timeout)
                    print('Timeout in game thread {} ({})'.format(data['id'], timeout))
                    if data['stdout']:
                        print('Standard output:', data['stdout'])
                    if data['stderr']:
                        print('Standard error:', data['stderr'])
                elif 'incomplete' in data:
                    print('No result from game thread {}'.format(data['id']))
                else:
                    success = True
                    succ_ends += 1
                    dt = data['duration']
                    timeEst.normal(dt)
                    w1, d1, l1 = data['result']
                    s1 = 'Partial result {:3d}: {:2d} {:2d} {:2d}'.format(data['id'], w1, d1, l1)
                    s2 = '({} seconds, remaining chunks: {})'.format(int(dt), total_starts - succ_ends)
                    print(s1, '\t', s2)
                    w += w1
                    d += d1
                    l += l1
                if not success:
                    # We get an updated timeout, which should be better than the older
                    timeout = int(timeEst.timeout())
                    new_starts.add(pool.submit(play_one, config, list(args), games, timeout=timeout, id=cid))
                    cid += 1
            pending = not_done | new_starts
    # Get the messages from time estimator and print them
    ms = timeEst.get_messages()
    for mes in ms:
        print('Time estimator:', mes)

    if w + d + l == 0:
        print('Play: No result from self play')
        return 0
    else:
        print('Play:', w, d, l)
        if config.elo:
            return elo((w + 0.5 * d) / (w + d + l))
        else:
            return elowish((w + 0.5 * d) / (w + d + l))

# The timeout (in seconds?) is taken for 10 games with 20000 nodes
# and may be wrong for longer games
# We should keep statistics for current configuration and use them (with some margin)
def play_one(config, args, games, timeout=360, id=0):
    skip = random_skip(config.ipgnlen, games)
    args.extend(['-s', str(skip), '-f', str(games)])
    # print('Will start:')
    # print(args)
    w = None
    # For windows: shell=True to hide the window of the child process - seems not to be necessary!
    try:
        starttime = datetime.datetime.now()
        status = subprocess.run(args, capture_output=True, cwd=config.playdir, \
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as toe:
        return { 'id': id, 'timeout': timeout, 'stdout': toe.stdout, 'stderr': toe.stderr }
    except Exception as exc:
        return { 'id': id, 'exception': exc }
    else:
        endtime = datetime.datetime.now()
        dt = endtime - starttime
        for line in status.stdout.splitlines():
            #print('Line:', line)
            if resre.match(line):
                _, _, _, ws, ds, ls, _ = wdlre.split(line)
                w = int(ws)
                d = int(ds)
                l = int(ls)
                break
        if w is None:
            return { 'id': id, 'incomplete': True }
        return { 'id': id, 'result': (w, d, l), 'duration': dt.total_seconds() }

# Rosenbrock function for simulation
# Fixed parameters:
rose_a = 1.
rose_b = 10.
rose_offset = 0.

# No noise yet
# Global minimum of 0 at (rose_a, rose_a)
# Must be inverted, as it has a minimum
# We also give an Y offset and scale
# So now global maximum of rose_offset * scale an (rose_a, rose_a)
def rosenbrock(params, scale):
    x, y = params[:2]
    print('Rosenbrock with', x, y)
    y2 = y - x * x
    val = (rose_a - x) * (rose_a - x) + rose_b * y2 * y2
    return (-val + rose_offset) * scale

# vim: tabstop=4 shiftwidth=4 expandtab
