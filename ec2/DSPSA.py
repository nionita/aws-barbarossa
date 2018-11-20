# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:59:24 2017

@author: nicu
"""

import numpy as np
import random
import math
import argparse
import subprocess
import re
import os
import os.path
import sys
import json
import zlib
import base64
import boto3

version = '0.1.0'
eps0 = 1e-2
eps1 = 1 - eps0

# DSPSA
# We have to optimize a stochastic function of n integer parameters,
# but we can only get noisy results from measurements

'''
Transforming mean result in an elo difference involves the constants log(10) and 400,
but at the end those are multiplicative, so it is just a matter of step size
Here frac is between 0 and 1, exclusive! More than 0.5 means the first player is better
We have guards for very excentric results, eps0 and eps1, which also limit the gradient amplitude:
'''
def elowish(frac):
    frac = max(eps0, min(eps1, frac))
    return -math.log(1 / frac - 1)

'''
Implementation of DSPSA
'''
class DSPSA:
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
            self.step = 0
            self.since = 0
            self.theta = np.array(config.pinits, dtype=np.float32)
        else:
            self.step = status['step']
            self.since = status['since']
            self.theta = np.array(status['theta'], dtype=np.float32)
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

    def save_status(self, file):
        dic = {
            'step': self.step,
            'since': self.since,
            'theta': self.theta.tolist()
        }
        jsonstr = json.dumps(dic)
        with open(file, 'w', encoding='utf-8') as of:
            print(jsonstr, file=of)

    '''
    Optimize by the classical DSPSA method
    '''
    def optimize(self):
        done = False
        while not done:
            done = self.step_dspsa()
        return self.theta

    def step_dspsa(self):
        if self.step >= self.msteps:
            return True
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
        print('theta:', self.theta)
        self.step += 1
        if self.rend is not None:
            ntheta = np.rint(self.theta)
            if np.all(ntheta == self.rtheta):
                self.since += 1
                if self.since >= self.rend:
                    print('Rounded parameters unchanged for', self.rend, 'steps')
                    return True
            else:
                self.rtheta = ntheta
                self.since = 0
        return False

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

'''
A config class for tuning configuration
It represents a config file with the following structure:

# This line is a comment
# The file begins with section 0, which defines match parameters
# (the selfplay program, the current directory for the selfplay execution, input pgn file
# for the games, search depth, number of games per match) and optimization hyper parameters
name: test
optdir: C:/Learn/dspsa
selfplay: C:/astra/SelfPlay-dnp.exe
playdir: C:/Learn/dspsa
ipgnfile: C:/astra/open-moves/open-moves.fen
depth: 4
games: 8
laststep: 10
msteps: 10

#alpha: 0.501
#rend: 300

[params]
# Section params defines the parameters to be optimized (can be empty or not appear at all)
# with param name, starting value and scale
epMovingMid:  156, 3
epMovingEnd:  156, 3
epMaterMinor: 1, 1

[weights]
# Section weights defines the weights to be optimized (can be empty or not appear at all)
# with param name, starting mid game value, starting end game value, and scale
kingSafe: 1, 0, 1
kingOpen: 2, 4, 1
kingPlaceCent: 8, 1, 1

'''
class Config:
    # These are acceptable fields in section 0, with their type and maybe default value
    # S is string, I integer and F float
    fields = {
        'name': 'S',
        'optdir': ('S', '.'),
        'selfplay': 'S',
        'playdir': ('S', '.'),
        'ipgnfile': 'S',
        'depth': ('I', 4),
        'games': ('I', 16),
        'laststep': ('F', 0.1),
        'alpha': ('F', 0.501),
        'msteps': ('I', 1000),
        'rend': 'I',
        'save': ('I', 10)
    }

    mandatory_fields = ['name', 'selfplay', 'ipgnfile']

    '''
    A config can be initialized either with a file name or with a dictionary
    When called with a file name, the file will be read and transformed into a dictionary which
    will be used for the config creation
    When called with a dictionary, the dictionary keys will be used as attributes of the
    new created config object
    '''
    def __init__(self, source):
        if type(source) != dict:
            # Called with an open file
            source = Config.readConfig(source)
        # Here source is a dictionary
        for name, val in source.items():
            self.__setattr__(name, val)

    @staticmethod
    def accept_data_type(field_name, field_type):
        if field_type not in ['S', 'I', 'F']:
            raise Exception('Config: wrong field type {} for field {}'.format(field_type, field_name))

    @staticmethod
    def create_defaults():
        values = dict()
        for field_name, field_spec in Config.fields.items():
            if type(field_spec) == str:
                Config.accept_data_type(field_name, field_spec)
                values[field_name] = None
            else:
                field_type, field_ini = field_spec
                Config.accept_data_type(field_name, field_type)
                values[field_name] = field_ini
        return values

    @staticmethod
    def readConfig(cof):
        # Transform the config file to a dictionary
        values = Config.create_defaults()
        seen = set()
        sectionNames = [dict(), dict()]
        section = 0
        lineno = 0
        error = False
        for line in cof:
            lineno += 1
            # split the comment path
            line = re.split('#', line)[0].lstrip().rstrip()
            if len(line) > 0:
                if line == '[params]':
                    section = 1
                elif line == '[weights]':
                    section = 2
                else:
                    parts = re.split(r':\s*', line, 1)
                    name = parts[0]
                    val = parts[1]
                    if section == 0:
                        if name in Config.fields:
                            field_type = Config.fields[name]
                            if type(field_type) == tuple:
                                field_type = field_type[0]
                            if field_type == 'S':
                                values[name] = val
                            elif field_type == 'I':
                                values[name] = int(val)
                            elif field_type == 'F':
                                values[name] = float(val)
                            else:
                                raise Exception('Cannot be here!')
                        else:
                            print('Config error in line {:d}: unknown config name {:s}'.format(lineno, name))
                            error = True
                    else:
                        vals = re.split(r',\s*', val)
                        if len(vals) == section + 1:
                            if name in seen:
                                print('Config error in line {:d}: name {:s} already seen'.format(lineno, name))
                                error = True
                            else:
                                seen.add(name)
                                sectionNames[section-1][name] = [int(v) for v in vals]
                        else:
                            print('Config error in line {:d}: should have {:d} values, it has {:d}'.format(lineno, section+1, len(vals)))
                            error = True
        for mand in Config.mandatory_fields:
            if mand not in values:
                print('Config does not contain mandatory field {}'.format(mand))
                error = True

        if error:
            raise Exception('Config file has errors')
        hasScale = False

        # Collect the eval parameters
        values['pnames'] = []
        values['pinits'] = []
        values['pscale'] = []
        for name, vals in sectionNames[0].items():
            val = vals[0]
            scale = vals[1]
            values['pnames'].append(name)
            values['pinits'].append(val)
            values['pscale'].append(scale)
            if scale != 1:
                hasScale = True

        # Collect the eval weights
        for name, vals in sectionNames[1].items():
            mid = vals[0]
            end = vals[1]
            scale = vals[2]
            values['pnames'].append('mid.' + name)
            values['pinits'].append(mid)
            values['pscale'].append(scale)
            values['pnames'].append('end.' + name)
            values['pinits'].append(end)
            values['pscale'].append(scale)
            if scale != 1:
                hasScale = True

        if not hasScale:
            values['pscale'] = None

        return values

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
    skip = random.randint(0, 25000)
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

def optimize(opt):
    done = False
    while not done:
        done = opt.step_dspsa()
        if opt.step % 10 == 0:
            opt.report(opt.theta)
        if opt.save is not None and opt.step % opt.save == 0:
            opt.save_status(opt.config.name + '-save.txt')
    return np.rint(opt.theta)

def get_sqs():
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName='aws-barbarossa-requests.fifo')
    print('Queue URL:', queue.url)
    return queue

'''
Gets & returns a request from the given AWS queue, sets the visibility timeout
'''
def get_request(queue, visibility=10):
    requests = []
    wait = 20
    attribute_names = ['MessageGroupId', 'MessageDeduplicationId', 'SequenceNumber']
    requests = queue.receive_messages(MaxNumberOfMessages=1, AttributeNames=attribute_names,
        WaitTimeSeconds=wait, VisibilityTimeout=visibility)
    return requests

'''
Requests are processed undefinitely, as log as we have some
'''
def process_all_requests(queue):
    while True:
        print('Getting a request from the queue')
        requests = get_request(queue)

        if len(requests) == 0:
            # Should we cancel the spot instance here?
            return

        print('Requests:')
        for request in requests:
            process_request(request)

def process_request(request):
    print(request.body)
    print('Process...')
    print('Delete message:')
    request.delete()
    print('Message deleted')

def encoding(inp: str):
    cbytes = zlib.compress(inp.encode())
    b64 = base64.b64encode(cbytes)
    return b64.decode()

def decoding(inp: str):
    zbytes = base64.b64decode(inp.encode())
    ebytes = zlib.decompress(zbytes)
    return ebytes.decode()

'''
Define the argument parser
'''
def argument_parser():
    parser = argparse.ArgumentParser(description='Parameter optimization for Barbarossa')
    parser.add_argument('--version', action='version', version=version)
    subparsers = parser.add_subparsers(dest='command', help='run sub-commands: local, cloud or aws')

    local = subparsers.add_parser('local', help='run an optimization request locally')
    local.add_argument('--save', type=int, help='save optimization status after given steps (default: 10)')
    local.add_argument('config', type=argparse.FileType('r'))

    aws = subparsers.add_parser('aws', help='run an optimization request on AWS (requests from SQS)')
    aws.add_argument('--save', type=int, help='save optimization status after given steps (default: 10)')

    cloud = subparsers.add_parser('cloud', help='send an optimization request to AWS')
    cloud.add_argument('config', type=argparse.FileType('r'))

    return parser

'''
Read the config file, encode it and create a SQS request for aws-barbarossa
'''
def send_to_cloud(args):
    cfc = args.config.read()
    a = encoding(cfc)
    print('CFC len:', len(cfc))
    print('Z64 len:', len(a))

    # queue = get_sqs()
    # process_all_requests(queue)

'''
This runs on AWS
'''
def run_on_aws(args):
    print('This must run on AWS')

'''
This runs a local optimization
'''
def run_local_optimization(args):
    print('Running local optimization')
    config = Config(args.config)
    os.chdir(config.optdir)
    if args.save:
        save = args.save
    else:
        save = config.save
    opt = DSPSA(config, play, save=save)
    r = optimize(opt)
    #r = opt.momentum(play, config)
    #r = opt.adadelta(play, config, mult=20, beta=0.995, gamma=0.995, niu=0.999, eps=1E-8)
    opt.report(r, title='Optimum', file=os.path.join(config.optdir, config.name + '-optimum.txt'))
    opt.report(r, title='Optimum', file=None)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    print('Parsed:', args)

    if args.command == 'cloud':
        send_to_cloud(args)
    elif args.command == 'aws':
        run_on_aws(args)
    else:
        run_local_optimization(args)

    # confFile = sys.argv[1]
    # config = Config(confFile)

    # # Test: to/from json should be identity
    # jsonstr = json.dumps(config.__dict__)
    # config1 = Config(json.loads(jsonstr))

    # if config.__dict__ != config1.__dict__:
    #     print('Not equal')

    # # Test: to JSON, compress, base 64
    # jsonstr = json.dumps(config.__dict__)
    # print('JSON len:', len(jsonstr))
    # cbytes = zlib.compress(jsonstr.encode())
    # b64 = base64.b64encode(cbytes)
    # print('Base64 len:', len(b64))
    # print('In base64:', b64)

    # # Real
