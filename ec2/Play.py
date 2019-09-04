import random
import math
import subprocess
import re
import os

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

# Play a match with a given number of games between theta+ and theta-
# Player 1 is theta+
# Player 2 is theta-
def play(tp, tm, config):
    # print('chdir to', config.playdir)
    os.chdir(config.playdir)
    with open('playerp.cfg', 'w', encoding='utf-8') as plf:
        for p, v in zip(config.pnames, tp):
            plf.write('%s=%d\n' % (p, v))
    with open('playerm.cfg', 'w', encoding='utf-8') as plf:
        for p, v in zip(config.pnames, tm):
            plf.write('%s=%d\n' % (p, v))
    skip = random.randint(0, config.ipgnlen - config.games + 1)
    #print('Skip = %d' % skip)
    args = [config.selfplay, '-m', config.playdir, '-a', 'playerp.cfg', '-b', 'playerm.cfg',
            '-i', config.ipgnfile, '-d', str(config.depth), '-s', str(skip), '-f', str(config.games)]
    # print('Will start:')
    # print(args)
    w = None
    # For windows: shell=True to hide the window of the child process
    with subprocess.Popen(args, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=config.playdir, universal_newlines=True, shell=True) as proc:
        for line in proc.stdout:
            #print('Got:', line)
            if resre.match(line):
                #vals = wdlre.split(line)
                #print(vals)
                _, _, _, ws, ds, ls, _ = wdlre.split(line)
                w = int(ws)
                d = int(ds)
                l = int(ls)
                #print('I found the result %d, %d, %d' % (w, d, l))
    if w == None or w + d + l == 0:
        #raise RuntimeError('No result from self play')
        return 0
    else:
        return elowish((w + 0.5 * d) / (w + d + l))
