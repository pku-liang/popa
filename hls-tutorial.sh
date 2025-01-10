#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  Generate MLIR files: ./hls-tutorial.sh generate"
    echo "  Run vanilla version: ./hls-tutorial.sh run vanilla"
    echo "  Run optimized version: ./hls-tutorial.sh run"
fi

POPA_DIR=${POPA_DIR:=popa}
HECTOR_DIR=${HECTOR_DIR:=hector}
HESTIA_DIR=${HESTIA:=hestia}

run_popa() {
    pushd $POPA_DIR
    if [[ ! -e "tutorial" ]]; then
        g++ tutorial.cpp -g -I./install/include -L./install/lib -lHalide -std=c++17
    fi
    env LD_LIBRARY_PATH=./install/lib ./tutorial $1
    popd
}

run_hector() {
    hestia_run=$(readlink -f $HESTIA_DIR/target/release/hestia)
    pushd $HECTOR_DIR
    ./build/bin/hector-opt $1 --canonicalize --hls-unroll --affine-loop-normalize --canonicalize --new-array-partition --canonicalize --remove-access=mode=aggressive --lower-affine \
        --convert-input="top-function=_kernel_C_s0_run_on_device resource=./examples/resource_dynamatic.json" --dump-scf --scf-to-tor="pipeline" --schedule-tor --split-schedule --dump-tor="json=tor.json" 2>/dev/null
    $hestia_run tor.tcl
    popd
}

if [[ "$1" == "generate" ]]; then
    for i in {0..3}; do
        run_popa $i
    done
fi

if [[ "$1" == "run" ]]; then
    if [[ "$2" == "vanilla" ]]; then
        cp $POPA_DIR/mm_0.mlir $HECTOR_DIR/examples/popa/mm.mlir
        run_hector examples/popa/mm.mlir
    else
        cp $POPA_DIR/mm_3.mlir $HECTOR_DIR/examples/popa/mm_optimized.mlir
        run_hector examples/popa/mm_optimized.mlir
    fi
fi