#!/bin/bash
set -v

HOME=/home/ubuntu
S3BASE=s3://storage.acons.at/chess/aws-barbarossa

EC2DIR=$S3BASE/ec2/
REQDIR=$S3BASE/requests/

# Sync bin scripts
aws s3 sync $EC2DIR $HOME/bin/
chmod +x $HOME/bin/*.sh

export BUILD=$HOME/bin/build.sh

python3 $HOME/bin/DSPSA.py aws
