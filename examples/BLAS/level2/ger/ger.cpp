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
    #define P               jj,   ii, j, i
    #define P_jj_minus_1    jj-1, ii, j, i

    // Linearized addresses
    #define total_i         (ii + II * i)
    #define total_j         (jj + JJ * j)

    // Outer loop bounds, which are determined by input sizes
    #define I (A.dim(1).extent() / II)
    #define J (A.dim(0).extent() / JJ)

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), X("X", TTYPE, 1), Y("Y", TTYPE, 1);

    // UREs
    Var jj("jj"), ii("ii"), j("j"), i("i");
    URE uY("uY", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ("uZ", TTYPE, {P});
    uX(P) = select(jj == 0, X(total_i), uX(P_jj_minus_1));
    uY(P) = Y(total_j);
    uZ(P) = A(total_j, total_i) + uX(P) * uY(P);

    // Put all the UREs inside the same loop nest of X.
    uX.merge_ures(uY, uZ);

    // Explicitly set the loop bounds
    uX.set_bounds(jj, 0, JJ, ii, 0, II)
      .set_bounds(j,  0, J,  i,  0, I);

    // Create a systolic array
    uX.space_time_transform(jj);

    // I/O network
    Stensor DA("aLoader", DRAM);
    Stensor DX("xLoader", DRAM);
    Stensor DY("yLoader", DRAM);
    Stensor DZ("unloader", DRAM), Z("deserializer");
    A >> DA >> FIFO(256);
    X >> DX >> FIFO(256);
    Y >> DY >> FIFO(256);
    uZ >> FIFO(256) >> DZ >> Z(total_i);

    // Compile the kernel to an FPGA bitstream, a d expose a C interface for the host to invoke
    Z.compile_to_host("ger-interface", { A, X, Y }, "ger", IntelFPGA);
    printf("Success\n");
    return 0;
}
