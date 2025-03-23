# POPA
A framework for portable performance across FPGAs and GPUs.

# Introduction
This paper aims at high and portable performance for tensor computations across spatial (e.g., FPGAs) and vector architectures (e.g., GPUs). The state-of-the-art usually address performance portability across vector architectures (CPUs and GPUs). However, they either miss FPGAs or do not achieve high performance. Without a common architectural abstraction, they program and optimize spatial and vector devices separately, causing low portability. We propose a unified programming framework, POPA, which achieves portability via architectural abstraction and performance via specialization. A parallel dataflow machine is proposed as a unified, abstract hardware target that hides differences of concrete architectures. The machine consists of software-defined systolic arrays and a tensor-specific cache hierarchy, which captures pipeline parallelism and customizable memories on FPGAs, as well as multithreading parallelism on GPUs. The machine is specified in a unified programming model as two dataflow graphs for scheduling compute and data movement, respectively. A compiler then specializes the abstract machine to exploit the properties of FPGAs and GPUs, bridging the gap between the abstract machine and a concrete architecture. We evaluate POPA on several Intel FPGAs and GPUs with high-profile tensor kernels, and this is the first system that achieves >=80% performance of expert-written code or machine peak across architectures, to the best of our knowledge.

# Quick Start Guide

1. Acquiring repositories
```
git clone https://github.com/llvm/llvm-project.git
git checkout cbc378ecb87e3f31dd5aff91f2a621d500640412
git clone -b tutorial-aspdac https://github.com/pku-liang/Hector.git
git clone -b mlir https://github.com/pku-liang/popa
```

2. Build LLVM and MLIR
```
cd llvm-project
git apply -p1 ../popa/mlir_link_issue.patch
cmake -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_LLD=ON \
        -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -G Ninja -S llvm -B build
cmake --build build
cmake --install build --prefix install
cd ..
```

2. Build Hector
```
cd Hector
git submodule update --init --recursive
cmake -G Ninja -DMLIR_DIR=../llvm-project/build/lib/cmake/mlir -B build
cmake --build build
cd ..
```

3. Build POPA
```
export PATH=$PWD/llvm-project/install/bin:$PATH
cd popa
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
cmake --install build --prefix install
cd ..
```

# Publications

+ **Productively Generating a High-Performance Linear Algebra Library on FPGAs**.  
Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. TRETS, 2025. [Link](https://doi.org/10.1145/3723046)

+ **POPA: Expressing High and Portable Performance across Spatial and Vector Architectures for Tensor Computations**.  
Xiaochen Hao, Hongbo Rong, Mingzhe Zhang, Ce Sun, Hong Jiang, Yun Liang. FPGA, 2024. [Link](https://doi.org/10.1145/3626202.3637566)

+ **Lasa: Abstraction and Specialization for Productive and Performant Linear Algebra on FPGAs**.  
Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. FCCM, 2023. [Link](https://ieeexplore.ieee.org/abstract/document/10171577)

+ **SuSy: a programming model for productive construction of high-performance systolic arrays on FPGAs**.  
Yi-Hsiang Lai, Hongbo Rong, Size Zheng, Weihao Zhang, Xiuping Cui, Yunshan Jia, Jie Wang, Brendan Sullivan, Zhiru Zhang, Yun Liang, Youhui Zhang, Jason Cong, Nithin George, Jose Alvarez, Christopher Hughes, and Pradeep Dubey. 2020.  ICCAD 2020. [Link](https://ieeexplore.ieee.org/document/9256583) 

+ **T2S-Tensor: Productively Generating High-Performance Spatial Hardware for Dense Tensor Computations**.  
Nitish Srivastava, Hongbo Rong, Prithayan Barua, Guanyu Feng, Huanqi Cao, Zhiru Zhang, David Albonesi,Vivek Sarkar, Wenguang Chen, Paul Petersen, Geoff Lowney, Adam Herr, Christopher Hughes,Timothy Mattson, Pradeep Dubey. FCCM, 2019. [Link](https://ieeexplore.ieee.org/document/8735529)

