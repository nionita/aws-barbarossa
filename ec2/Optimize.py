# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:59:24 2017

@author: nicu
"""

import argparse
import os
import os.path
import sys
import subprocess
import json
import zlib
import base64
import re
from datetime import datetime, timedelta

from Play import play
from Config import Config
from DSPSAOptimizer import DSPSAOptimizer
from BayesOptimizer import BayesOptimizer
from AWS import get_sqs

version = '0.1.0'
aws_queue_name = 'aws-barbarossa-requests.fifo'

def run_build(branch, timeout=None):
    print('Building', branch)
    args = 'build.sh ' + branch
    # For windows: shell=True to hide the window of the child process
    # This could be better achieved by using STARTUPINFO
    # check=True: will raise an exception when not terminating normally
    try:
        err = False
        cp = subprocess.run(args, check=True, timeout=timeout, universal_newlines=True, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        print('Build error:', e)
        cp = e
        err = True
    finally:
        print('Arguments used to call build:')
        print(cp.args)
        print('Output:')
        print(cp.stdout)
        if err:
            raise cp

def optimize_callback_aws(opt):
    # On aws we will always have save
    return opt.step % opt.save == 0

def optimize_callback_local(opt):
    report_times(opt)
    if opt.step % 10 == 0:
        opt.report(opt.theta)
    if opt.save is not None and opt.step % opt.save == 0:
        jsonstr = json.dumps(opt.get_status())
        # We write to a new file, to avoid a crash while saving
        # and then rename to correct save file
        sfile = opt.config.name + '-save.txt'
        nfile = 'new-' + sfile
        with open(nfile, 'w', encoding='utf-8') as of:
            print(jsonstr, file=of)
        os.replace(nfile, sfile)
    return False

start_date = datetime.now()

def report_times(opt):
    global start_date
    now = datetime.now()
    td = now - start_date
    # The opt step is the next to execute, i.e. first time step == 1
    seconds_per_step = td.total_seconds() / opt.step
    remaining_steps = opt.msteps - opt.step
    remaining_seconds = remaining_steps * seconds_per_step
    remaining = timedelta(seconds=remaining_seconds)
    eta = now + remaining
    print('Since:', td.total_seconds(), 'seconds, per step:', seconds_per_step, 'seconds')
    print('Remaining:', remaining.seconds, 'seconds, ETA:', eta.strftime('%d.%m.%Y %H:%M:%S'))

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
Requests are processed undefinitely, as long as we have some
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

'''
Encode a dictionary to JSON, compress, and encode to base64
'''
def encoding(message):
    # Make a json string
    jbody = json.dumps(message)
    # Compress
    cbytes = zlib.compress(jbody.encode())
    # Encode base64
    b64 = base64.b64encode(cbytes)
    # Return the string
    return b64.decode()

'''
Decode from base64, decompress, decode from JSON
'''
def decoding(inp: str):
    # Decode from base64
    zbytes = base64.b64decode(inp.encode())
    # Decompress
    jstr = zlib.decompress(zbytes).decode()
    # From JSON
    return json.loads(jstr)

'''
Read the config file, encode it and create a SQS request for aws-barbarossa
'''
def send_to_cloud(args):
    print('Sending to cloud optimization')
    config = Config(args.config)
    message = {
        'config': config.__dict__
    }
    body = encoding(message)
    print('c+b64 len:', len(body))

    queue = get_sqs(aws_queue_name)
    resp = queue.send_message(MessageGroupId='original-request', MessageBody=body)
    print('Request was sent, response:', resp)

def rewrite_config_for_aws(home, cf):
    spe = re.split(r'/', cf['selfplay'])[-1]
    if not spe.startswith('SelfPlay'):
        raise Exception('Could not identify SelfPlay: ' + spe)
    spe = re.sub(r'^SelfPlay-', '', spe)
    branch = re.sub(r'\.exe$', '', spe)
    pgn = re.split(r'/', cf['ipgnfile'])[-1]
    rundir = os.path.join(home, 'run')
    cf['optdir'] = rundir
    cf['playdir'] = rundir
    cf['selfplay'] = os.path.join(home, 'engines', 'SelfPlay-' + branch)
    cf['ipgnfile'] = os.path.join(home, 'static', pgn)
    return branch, pgn

'''
This runs on AWS
'''
def run_on_aws(args):
    print('Running optimization part on AWS')
    home = os.environ['HOME']
    branch_old, pgn_old = '', ''
    queue = get_sqs(aws_queue_name)
    while True:
        reqs = get_request(queue)
        if len(reqs) == 0:
            print('No more requests, terminate')
            return
        req = reqs[0]
        message = decoding(req.body)
        cf = message['config']
        # Some config parameters must be adapted to run on AWS
        branch, pgn = rewrite_config_for_aws(home, cf)
        print('This is the new config')
        print(cf)
        try:
            if branch != branch_old or pgn != pgn_old:
                # Increase message visibility for the build time
                res = req.change_visibility(VisibilityTimeout=300)
                run_build(branch)
                branch_old, pgn_old = branch, pgn

            config = Config(cf)
            if 'status' in message:
                status = message['status']
            else:
                status = None

            # We should compute how long it takes per step and modify save to be around every hour or so
            if config.save:
                save = config.save
            else:
                save = 10

            # Changing message visibility: should be calculated
            res = req.change_visibility(VisibilityTimeout=3600)
            print('Message visibility result:', res)

            # Create the optimizer & optimize
            opt = DSPSAOptimizer(config, play, status=status, save=config.save)
            r = opt.optimize(optimize_callback_aws)

            if opt.done:
                opt.report(r, title='Optimum', file=config.name + '-optimum.txt')
                # Send the result to S3
                # TODO: why is not writing the report file?
            else:
                status = opt.get_status()
                # Send new request to SQS

        except Exception as e:
            print('Working on this request terminated with error:', e)

        finally:
            res = req.delete()
            print('Message delete result:', res)

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
    if config.method == 'DSPSA':
        opt = DSPSAOptimizer(config, play, save=save)
    elif config.method == 'Bayes':
        opt = BayesOptimizer(config, play, save=save)
    r = opt.optimize(optimize_callback_local)
    #r = opt.momentum(play, config)
    #r = opt.adadelta(play, config, mult=20, beta=0.995, gamma=0.995, niu=0.999, eps=1E-8)
    opt.report(r, title='Optimum', file=os.path.join(config.optdir, config.name + '-optimum.txt'))
    opt.report(r, title='Optimum', file=None)

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

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    # print('Parsed:', args)

    if args.command == 'cloud':
        send_to_cloud(args)
    elif args.command == 'aws':
        run_on_aws(args)
    else:
        run_local_optimization(args)

    # # Test: to/from json should be identity
    # jsonstr = json.dumps(config.__dict__)
    # config1 = Config(json.loads(jsonstr))

    # if config.__dict__ != config1.__dict__:
    #     print('Not equal')

# vim: tabstop=4 shiftwidth=4 expandtab
