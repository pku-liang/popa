#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  Generate MLIR files: ./hls-tutorial.sh generate"
    echo "  Run vanilla version: ./hls-tutorial.sh run vanilla"
    echo "  Run optimized version: ./hls-tutorial.sh run optimized"
fi

POPA_DIR=${POPA_DIR:=popa}
HECTOR_DIR=${HECTOR_DIR:=hector}
HESTIA_DIR=${HESTIA:=hestia}

run_popa() {
    pushd $POPA_DIR/examples >/dev/null
    if [[ ! -e "tutorial" ]]; then
        g++ tutorial.cpp -g -I../install/include -L../install/lib -lHalide -std=c++17 -o tutorial
    fi
    env LD_LIBRARY_PATH=../install/lib ./tutorial $1
    popd >/dev/null
}

run_hector() {
    pushd $HECTOR_DIR >/dev/null
    which hector-opt
    which hestia
    hector-opt $1 --canonicalize --hls-unroll --affine-loop-normalize --canonicalize --new-array-partition --canonicalize --remove-access=mode=aggressive --lower-affine \
        --convert-input="top-function=_kernel_C_s0_run_on_device resource=./examples/resource_dynamatic.json" --dump-scf --scf-to-tor="pipeline" --schedule-tor --split-schedule --dump-tor="json=tor.json" &>/dev/null
    hestia tor.tcl
    popd >/dev/null
}

if [[ "$1" == "generate" ]]; then
    run_popa vanilla
    run_popa optimized
fi

if [[ "$1" == "run" ]]; then
    if [[ "$2" == "vanilla" ]]; then
        cp $POPA_DIR/examples/SCF_vanilla.mlir $HECTOR_DIR/examples/popa/
        run_hector examples/popa/SCF_vanilla.mlir
    elif [[ "$2" == "optimized" ]]; then
        cp $POPA_DIR/examples/SCF_optimized.mlir $HECTOR_DIR/examples/popa/
        run_hector examples/popa/SCF_optimized.mlir
    fi
fi
