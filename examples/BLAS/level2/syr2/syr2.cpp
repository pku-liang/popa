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
    #define P               jjj,   iii, jj, ii, j, i
    #define P_jjj_minus_1   jjj-1, iii, jj, ii, j, i

    // Linearized addresses
    #define total_i         (iii + III*ii + III*II*i)
    #define total_j         (jjj + JJJ*jj + JJJ*JJ*j)

    // Outer loop bounds, which are determined by input sizes
    #define I (A.dim(1).extent() / (II*III))
    #define J (A.dim(0).extent() / (JJ*JJJ))

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), X("X", TTYPE, 1), Y("Y", TTYPE, 1);

    // UREs
    Var jjj("jjj"), iii("iii"), jj("jj"), ii("ii"), j("j"), i("i");
    URE uY("uY", TTYPE, {P}), uX("uX", TTYPE, {P}), uZ("uZ", TTYPE, {P});
    uX(P) = select(jjj == 0, X(total_i), uX(P_jjj_minus_1));
    uY(P) = Y(total_j);
    uZ(P) = uX(P) * uY(P);

    URE uY_T("uY_T", TTYPE, {P}), uX_T("uX_T", TTYPE, {P}), uZ_T("uZ_T", TTYPE, {P});
    uX_T(P) = X(total_j);
    uY_T(P) = select(jjj == 0, Y(total_i), uY_T(P_jjj_minus_1));
    uZ_T(P) = uX_T(P) * uY_T(P);

    // Put all the UREs inside the same loop nest of X.
    uX.merge_ures(uY, uZ);
    uX_T.merge_ures(uY_T, uZ_T);

    // Explicitly set the loop bounds
    uX.set_bounds(jjj, 0, JJJ, iii, 0, III)
      .set_bounds(jj,  0, JJ,   ii, 0, II)
      .set_bounds(j,   i, J-i,   i, 0, I);
    uX_T.set_bounds(jjj, 0, JJJ, iii, 0, III)
        .set_bounds(jj,  0, JJ,   ii, 0, II)
        .set_bounds(j,   i, J-i,   i, 0, I);

    // Create a systolic array
    uX.space_time_transform(jjj);
    uX_T.space_time_transform(jjj);

    URE Add("Add");
    Add(P) = uZ(P) + uZ_T(P) + A(total_j, total_i);
    Add.set_bounds(jjj, 0, JJJ, iii, 0, III)
       .set_bounds(jj,  0, JJ,   ii, 0, II)
       .set_bounds(j,   i, J-i,   i, 0, I);
    Add.space_time_transform(jjj);

    // I/O network
    Stensor DX("xLoader", DRAM), DY("yLoader", DRAM);
    Stensor SX("xFeeder", SRAM), SY("yFeeder", SRAM);
    X >> DX >> FIFO(256) >> SX.scope(j) >> FIFO(256) >> uX;
    Y >> DY >> FIFO(256) >> SY.scope(j).out(jjj) >> FIFO(256) >> uY;

    Stensor DX_T("xLoader_T", DRAM), DY_T("yLoader_T", DRAM);
    Stensor SX_T("xFeeder_T", SRAM), SY_T("yFeeder_T", SRAM);
    X >> DX_T >> FIFO(256) >> SX_T.scope(j).out(jjj) >> FIFO(256) >> uX_T;
    Y >> DY_T >> FIFO(256) >> SY_T.scope(j) >> FIFO(256) >> uY_T;

    Stensor DA("aLoader", DRAM, CHANNEL_1);
    A >> DA.out(jjj) >> FIFO(256) >> Add;

    Stensor DZ("unloader", DRAM, CHANNEL_2), Z("deserializer");
    Add >> FIFO(256) >> DZ.out(jjj) >> Z;

    // Compile the kernel to an FPGA bitstream, a d expose a C interface for the host to invoke
    Z.compile_to_host("syr2-interface", { A, X, Y }, "syr2", IntelFPGA);
    printf("Success\n");
    return 0;
}
