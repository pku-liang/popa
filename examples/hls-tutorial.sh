#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  Generate MLIR and SchedIR files: ./hls-tutorial.sh generate [basic/SA/IO]"
    echo "  Run vanilla version: ./hls-tutorial.sh run [basic/SA]"
    exit 1
fi

run_popa() {
    if [[ ! -e "matrix_multiply" ]]; then
        g++ matrix_multiply.cpp -g -I../install/include -L../install/lib -lHalide -std=c++17 -o matrix_multiply
    fi
    env LD_LIBRARY_PATH=../install/lib ./matrix_multiply $1
}

run_hector() {
    which hector-opt
    which hestia
    hector-opt $1 --canonicalize --hls-unroll --affine-loop-normalize --canonicalize --new-array-partition --canonicalize --remove-access=mode=aggressive --lower-affine \
        --convert-input="top-function=_kernel_C_s0_run_on_device resource=./examples/resource_dynamatic.json" --dump-scf --scf-to-tor="pipeline" --schedule-tor --split-schedule --dump-tor="json=tor.json" &>/dev/null
    hestia tor.tcl
}

if [[ "$1" == "generate" ]]; then
    run_popa $2
elif [[ "$1" == "run" ]]; then
    if [[ ! -e "$POPA_DIR/examples/SCF_$2.mlir" ]]; then
        echo "Please first generate the MLIR file SCF_$2.mlir"
        exit 1
    fi
    run_hector SCF_$2.mlir
else
    echo "Invalid command"
fi
