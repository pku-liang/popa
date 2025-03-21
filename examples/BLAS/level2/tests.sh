#!/bin/bash

function show_usage {
    echo "Usage:"
    echo "  ./tests.sh (devcloud|local) (a10|s10|gen9|gen12) [bitstream]"
}       

# BASH_SOURCE is this script
if [ $0 != $BASH_SOURCE ]; then
    echo "Error: The script should be directly run, not sourced"
    return
fi

if [ "$HOSTNAME" == "login-2" ]; then
    echo "Error: The script should be run on a compute node of $location or a local machine, not on the head node of $location (login-2)."
    exit
fi

if [ "$1" != "devcloud" -a "$1" != "local" ]; then
    show_usage
    exit
else
    location="$1"
fi

if [ "$2" != "a10" -a "$2" != "s10"  -a  "$2" != "gen9" -a  "$2" != "gen12" ]; then
    show_usage
    exit
else
    target="$2"
fi

# The path to this script.
PATH_TO_SCRIPT="$( cd "$(dirname "$BASH_SOURCE" )" >/dev/null 2>&1 ; pwd -P )"

cur_dir=$PWD
cd $PATH_TO_SCRIPT

if [ "$target" == "a10" -o "$target" == "s10" ]; then
    # FPGA: Verify correctness with tiny problem sizes and emulator
    ./test.sh $location gemv $target tiny emulator
    ./test.sh $location ger $target tiny emulator
    ./test.sh $location gerc $target tiny emulator
    ./test.sh $location geru $target tiny emulator
    ./test.sh $location her $target tiny emulator
    ./test.sh $location her2 $target tiny emulator
    ./test.sh $location syr $target tiny emulator
    ./test.sh $location syr2 $target tiny emulator
    ./test.sh $location hemv $target tiny emulator
    ./test.sh $location symv $target tiny emulator
    ./test.sh $location hbmv $target tiny emulator
    ./test.sh $location sbmv $target tiny emulator
    ./test.sh $location trmv $target tiny emulator
    ./test.sh $location trsv $target tiny emulator
else
    echo "Performance testing on $target not supported yet in this release"
    exit
fi

cd $cur_dir
