#!/bin/bash
set -v

# First parameter: branch to test
BRANCH=$1

HOME=/home/ubuntu
TEMP=$HOME/temp
REPO=https://github.com/nionita/Barbarossa.git
DIR=Barbarossa
EXEDIR=s3://storage.acons.at/chess/aws-barbarossa/exe

cd $HOME
[ -d $TEMP ] && rm -rf $TEMP

mkdir $TEMP

cd $TEMP
git clone --branch $BRANCH --depth 1 $REPO $DIR

cd $DIR
stack build
LIR=$(stack path | grep local-install-root | sed 's/local-install-root: //')
aws s3 cp $LIR/bin/SelfPlay $EXEDIR/SelfPlay-$BRANCH
