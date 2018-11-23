# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:59:24 2017

@author: nicu
"""

import argparse
import os
import os.path
import sys
import json
import zlib
import base64
import boto3

from Play import play
from Config import Config
from Optim import DSPSA

version = '0.1.0'

def optimize_callback_local(opt):
    if opt.step % 10 == 0:
        opt.report(opt.theta)
    if opt.save is not None and opt.step % opt.save == 0:
        jsonstr = opt.serialize_status()
        # We write to a new file, to avoid a crash while saving
        # and then rename to correct save file
        nfile = 'new-' + opt.config.name + '-save.txt'
        sfile = opt.config.name + '-save.txt'
        with open(nfile, 'w', encoding='utf-8') as of:
            print(jsonstr, file=of)
        os.replace(nfile, sfile)
    return False

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
    print('Sending to cloud optimization')
    config = Config(args.config)
    message = {
        'config': config.__dict__
    }
    jbody = json.dumps(message)
    body = encoding(jbody)
    print('plain len:', len(jbody))
    print('c+b64 len:', len(body))

    queue = get_sqs()
    resp = queue.send_message( MessageGroupId='original-request', MessageBody=body)
    print('Request was sent, response:', resp)

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
    r = opt.optimize(optimize_callback_local)
    #r = opt.momentum(play, config)
    #r = opt.adadelta(play, config, mult=20, beta=0.995, gamma=0.995, niu=0.999, eps=1E-8)
    opt.report(r, title='Optimum', file=os.path.join(config.optdir, config.name + '-optimum.txt'))
    opt.report(r, title='Optimum', file=None)

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
