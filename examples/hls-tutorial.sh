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
    pushd $POPA_DIR/examples >/dev/null
    if [[ ! -e "tutorial" ]]; then
        g++ tutorial.cpp -g -I../install/include -L../install/lib -lHalide -std=c++17 -o tutorial
    fi
    env LD_LIBRARY_PATH=../install/lib ./tutorial $1
    popd >/dev/null
}

run_hector() {
    hestia_run=$(readlink -f $HESTIA_DIR/target/release/hestia)
    pushd $HECTOR_DIR >/dev/null
    ./build/bin/hector-opt $1 --canonicalize --hls-unroll --affine-loop-normalize --canonicalize --new-array-partition --canonicalize --remove-access=mode=aggressive --lower-affine \
        --convert-input="top-function=_kernel_C_s0_run_on_device resource=./examples/resource_dynamatic.json" --dump-scf --scf-to-tor="pipeline" --schedule-tor --split-schedule --dump-tor="json=tor.json" 2>/dev/null
    $hestia_run tor.tcl
    popd >/dev/null
}

if [[ "$1" == "generate" ]]; then
    for i in {0..3}; do
	echo "Function exp_$i"
        run_popa $i
    done
fi

if [[ "$1" == "run" ]]; then
    if [[ "$2" == "vanilla" ]]; then
        cp $POPA_DIR/examples/mm_0.mlir $HECTOR_DIR/examples/popa/
        run_hector examples/popa/mm_0.mlir
    else
        cp $POPA_DIR/examples/mm_3.mlir $HECTOR_DIR/examples/popa/
        run_hector examples/popa/mm_3.mlir
    fi
fi
