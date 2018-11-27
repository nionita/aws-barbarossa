#!/bin/bash
set -v

# First parameter: branch to test
BRANCH=$1

# Some remote ressources
REPO=https://github.com/nionita/Barbarossa.git
S3HOME=s3://storage.acons.at/chess/aws-barbarossa

# Local directories and names
BUILDDIR=Barbarossa
BINARY=SelfPlay

HOME=/home/ubuntu
TEMP=$HOME/temp
STATIC=$HOME/static
ENGINES=$HOME/engines
RUN=$HOME/run

# Make sure we have the needed directories
[ -d $STATIC ] || mkdir $STATIC
[ -d $ENGINES ] || mkdir $ENGINES
[ -d $RUN ] || mkdir $RUN

# Static files (pgns)
cd $STATIC
aws s3 sync $S3HOME/static/ .

# Find/Copy/Build the SelfPlay binary
if [ ! -f $ENGINES/$BINARY-$BRANCH ]
then
    cd $ENGINES
    if aws s3 cp $S3HOME/exe/$BINARY-$BRANCH .
    then
        chmod +x $BINARY-$BRANCH
    else
        cd $HOME
        [ -d $TEMP ] && rm -rf $TEMP

        mkdir $TEMP

        cd $TEMP
        git clone --branch $BRANCH --depth 1 $REPO $DIR

        cd $BUILDDIR
        stack build
        LIR=$(stack path | grep local-install-root | sed 's/local-install-root: //')
        mv $LIR/bin/$BINARY $ENGINES/$BINARY-$BRANCH
        aws s3 cp $ENGINES/$BINARY-$BRANCH $S3HOME/exe/
    fi
fi