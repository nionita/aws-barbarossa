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

from Utils import Expo

eps0 = 1e-2
eps1 = 1 - eps0

'''
Transforming mean result in an elo difference involves the constants log(10) and 400,
but at the end those are multiplicative, so it is just a matter of step size
Here frac is between 0 and 1, exclusive! More than 0.5 means the first player is better
We have guards for very excentric results, eps0 and eps1, which also limit the gradient amplitude:
'''
def elowish(frac):
    frac = max(eps0, min(eps1, frac))
    return -math.log(1 / frac - 1)

resre = re.compile(r'End result')
wdlre = re.compile('[() ,]')

# When we work with a base param (like in bayesian optimization)
# we want to be able to give a fixed configuration (in which we can have also
# other parameters as the optimized ones set different as from the default
# self playing program)
base_file = None

# When we have a ready base param config, this is given in the config
# and we probably must copy it to the playdir
def copy_base_file(config):
    source = config.base
    print('We have a base file:', source)
    base_file = os.path.basename(source)
    dest = config.playdir
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
# The timeout will be calculated such that a given fraction of the games
# will terminate normally (i.e. not discarded via timeout) - probaq
expo = Expo()
probaq = 0.98

# Play a match with a given number of games between 2 param sets
# Player 1 is theta+ or the candidate
# Player 2 is theta- or the base param set
def play(config, tp, tm=None):
    os.chdir(config.playdir)
    pla = config.name + '-playerp.cfg'
    with open(pla, 'w', encoding='utf-8') as plf:
        for p, v in zip(config.pnames, tp):
            plf.write('%s=%d\n' % (p, v))
    if tm is None:
        if base_file is None:
            copy_base_file()
        base = base_file
    else:
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
    if expo.n == 0:
        timeout = 60 * games
    else:
        timeout = int(expo.quantile(probaq))

    print('Play: starting', total_starts, 'times with', games, 'games each, timeout =', timeout)

    # Play the games in parallel, collecting and consolidating the results
    w, d, l = 0, 0, 0
    succ_ends = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel) as pool:
        pending = set([pool.submit(play_one, config, list(args), games, timeout=timeout, id=i) for i in range(total_starts)])
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
                    # We add the timeout as a datapoint, otherwise this duration is not
                    # reflected in the statistics
                    # This still cuts the long tail of the exponential distribution, but on the other
                    # side, we must account for games which would take forever - a compensation
                    expo.add(timeout)
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
                    expo.add(dt)
                    w1, d1, l1 = data['result']
                    print('Partial result {:2d}: {:2d} {:2d} {:2d}\t({} seconds, remaining games: {})'.format(data['id'], w1, d1, l1, int(dt), total_starts - succ_ends))
                    w += w1
                    d += d1
                    l += l1
                if not success:
                    new_starts.add(pool.submit(play_one, config, list(args), games, timeout=timeout, id=cid))
                    cid += 1
            pending = not_done | new_starts
    if w + d + l == 0:
        print('Play: No result from self play')
        return 0
    else:
        print('Play:', w, d, l)
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
        status = subprocess.run(args, capture_output=True, cwd=config.playdir, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as toe:
        return { 'id': id, 'timeout': True, 'stdout': toe.stdout, 'stderr': toe.stderr }
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

# vim: tabstop=4 shiftwidth=4 expandtab
