import random
import math
import subprocess
import re
import os
import os.path
import shutil
import concurrent.futures

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
    print('Play: starting', total_starts, 'times with', games, 'games each')

    # Play the games in parallel, then collect and consolidate the results
    w, d, l = 0, 0, 0
    succ_starts = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel) as pool:
        while total_starts > succ_starts:
            to_start = total_starts - succ_starts
            executions = [ pool.submit(play_one, config, list(args), games) for _ in range(to_start) ]
            for future in concurrent.futures.as_completed(executions):
                try:
                    data = future.result()
                except Exception as exc:
                    print('Exception in one game thread:', exc)
                else:
                    if data is None:
                        print('No result from game thread')
                    else:
                        succ_starts += 1
                        w1, d1, l1 = data
                        print('Partial play result: {} {} {}\t(remaining: {})'.format(w1, d1, l1, total_starts - succ_starts))
                        w += w1
                        d += d1
                        l += l1
    if w + d + l == 0:
        print('Play: No result from self play')
        return 0
    else:
        print('Play:', w, d, l)
        return elowish((w + 0.5 * d) / (w + d + l))

# The timeout (in seconds?) is taken for 10 games with 20000 nodes
# and may be wrong for longer games
# We should keep statistics for current configuration and use them (with some margin)
def play_one(config, args, games, timeout=360):
    skip = random_skip(config.ipgnlen, games)
    args.extend(['-s', str(skip), '-f', str(games)])
    # print('Will start:')
    # print(args)
    w = None
    # For windows: shell=True to hide the window of the child process - seems not to be necessary!
    status = subprocess.run(args, capture_output=True, cwd=config.playdir, text=True, timeout=timeout)
    for line in status.stdout.splitlines():
        #print('Line:', line)
        if resre.match(line):
            _, _, _, ws, ds, ls, _ = wdlre.split(line)
            w = int(ws)
            d = int(ds)
            l = int(ls)
            break
    if w is None:
        return None
    return w, d, l

# vim: tabstop=4 shiftwidth=4 expandtab
