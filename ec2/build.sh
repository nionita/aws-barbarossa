#!/bin/bash
set -vx

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
echo Synchronising static files
cd $STATIC
aws s3 sync $S3HOME/static/ .

# Find/Copy/Build the SelfPlay binary
if [ ! -x $ENGINES/$BINARY-$BRANCH ]
then
    echo Binary $ENGINES/$BINARY-$BRANCH does not exist locally
    cd $ENGINES
    if aws s3 cp $S3HOME/exe/$BINARY-$BRANCH .
    then
        echo Binary $ENGINES/$BINARY-$BRANCH copied from s3
        chmod +x $BINARY-$BRANCH
    else
        echo Binary $ENGINES/$BINARY-$BRANCH does not exist on s3
        cd $HOME
        [ -d $TEMP ] && rm -rf $TEMP

        mkdir $TEMP
        cd $TEMP

        echo Clone branch $BRANCH from repo $REPO
        if git clone --branch $BRANCH --depth 1 $REPO
        then
            if cd $BUILDDIR && stack build
            then
                LIR=$(stack path | grep local-install-root | sed 's/local-install-root: //')
                mv $LIR/bin/$BINARY $ENGINES/$BINARY-$BRANCH
                aws s3 cp $ENGINES/$BINARY-$BRANCH $S3HOME/exe/
            else
                echo Cannot build SelfPlay binary
                exit 1
            fi
        else
            echo Cannot clone branch $BRANCH
            exit 1
        fi
    fi
fi