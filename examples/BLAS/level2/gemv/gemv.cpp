/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the BSD-2-Clause Plus Patent License (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSDplusPatent
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: BSD-2-Clause-Patent
*******************************************************************************/
#include "Halide.h"
#include "util.h"

// Constant parameters (inner loop bounds) of the design
#include "const-parameters.h"

using namespace Halide;

int main()
{
    // Dependences
    #define P               iii,   ii, kk,      k,   i
    #define P_iii_minus_1   iii-1, ii, kk,      k,   i
    #define P_kk_minus_1    iii,   ii, kk-1,    k,   i
    #define P_k_minus_1     iii,   ii, kk+KK-1, k-1, i
    #define P_Out           iii,   ii,               i

    // Linearized addresses
    #define total_i         (iii + III * ii + III * II * i)
    #define total_k         (kk + KK * k)

    // Outer loop bounds, which are determined by input sizes
    #define I (A.dim(1).extent() / (III * II))
    #define K (A.dim(0).extent() / KK)

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), X("X", TTYPE, 1);

    // UREs
    Var kkk("kkk"), iii("iii"), kk("kk"), ii("ii"), k("k"), i("i");
    URE uA("uA", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ("uZ", TTYPE, {P}), Out("Out");
    uA(P) = A(total_k, total_i);
    uX(P) = select(iii == 0, X(total_k), uX(P_iii_minus_1));
    uZ(P) = select(kk == 0 && k == 0, 0,
                  select(kk == 0, uZ(P_k_minus_1), uZ(P_kk_minus_1))
                  ) + uA(P) * uX(P);
    Out(P_Out) = select(kk == KK-1 && k == K-1, uZ(P));

    // Put all the UREs inside the same loop nest of X.
    uA.merge_ures(uX, uZ, Out);

    // Explicitly set the loop bounds
    uA.set_bounds(iii, 0, III)
      .set_bounds(kk,  0, KK,  ii,  0, II)
      .set_bounds(k,   0, K,   i,   0, I);

    // Create a systolic array
    uA.space_time_transform(iii);

    // I/O network
    Stensor DA("aLoader", DRAM);
    Stensor DX("xLoader", DRAM), SX("xFeeder", SRAM);
    Stensor DY("unloader", DRAM), Y("deserializer");
    A >> DA.out(iii) >> FIFO(256);
    X >> DX >> FIFO(256)
      >> SX.scope(k) >> FIFO(256);
    Out >> FIFO(256) >> DY >> Y(total_i);

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    Y.compile_to_host("gemv-interface", { A, X }, "gemv", IntelFPGA);
    printf("Success\n");
    return 0;
}
