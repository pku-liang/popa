#!/bin/bash

# BASH_SOURCE is this script
if [ $0 != $BASH_SOURCE ]; then
    echo "Error: The script should be directly run, not sourced"
    return
fi

if [ "$1" != "a10" -a "$1" != "s10"  -a  "$1" != "gen9" ]; then
    echo "Usage: ./devcloud-jobs.sh (a10|s10|gen9) [bitstream]"
    exit
else
    target=$1
fi

if [ "$HOSTNAME" != "login-2" ]; then
    echo "Error: The script should be run on the head node of DevCloud, i.e. login-2."
    exit
fi

# The path to this script.
PATH_TO_SCRIPT="$( cd "$(dirname "$BASH_SOURCE" )" >/dev/null 2>&1 ; pwd -P )"

cur_dir=$PWD
cd $PATH_TO_SCRIPT

if [ "$target" == "gen9" ]; then
    ./devcloud-job.sh gemm $target large hw
    ./devcloud-job.sh conv $target large hw
    ./devcloud-job.sh capsule $target large hw
    ./devcloud-job.sh pairhmm $target large hw
    ./devcloud-job.sh gemv $target large hw
    ./devcloud-job.sh gbmv $target large hw
elif [ "$target" == "a10" -o "$target" == "s10" ]; then
    ./devcloud-job.sh gemm $target large hw $2
    ./devcloud-job.sh conv $target large hw $2
    ./devcloud-job.sh capsule $target large hw $2
    ./devcloud-job.sh pairhmm $target large hw $2
    ./devcloud-job.sh gemv $target large hw $2
    ./devcloud-job.sh gbmv $target large hw $2
else
    echo "Performance testing on $target not supported yet in this release"
    exit
fi

cd $cur_dir
