#!/bin/bash
set -v

S3BASE=s3://storage.acons.at/chess/aws-barbarossa

EC2DIR=$S3BASE/ec2/
REQDIR=$S3BASE/requests/

# Sync bin scripts
# aws s3 sync $EC2DIR bin

aws s3 ls $REQDIR | while read d t l n
do
    if [ "$l" -gt 0 ]
    then
        echo $n
    fi
done