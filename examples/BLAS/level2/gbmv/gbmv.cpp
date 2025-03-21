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
    // Dependences for a single thread
    #define P                         vi,       iii,    kkk,    ii,  kk,  i,  k
    #define P_vi_minus_1              vi-1,     iii,    kkk,    ii,  kk,  i,  k
    #define P_kkk_minus_1_vi_plus_1   vi+1,     iii,    kkk-1,  ii,  kk,  i,  k
    #define P_kkk_minus_1_iii_plus_1  vi-VI+1,  iii+1,  kkk-1,  ii,  kk,  i,  k
    #define P_top                                       kkk,    ii,  kk,  i,  k
    #define P_right                   vi,       iii,            ii,  kk,  i,  k

    // Linearized addresses
    #define total_i           (vi + VI*iii + VI*III*ii + VI*III*II*i)
    #define total_k           (kkk + KKK*kk + KKK*KK*k)

    // Outer loop bounds, which are determined by input sizes
    #define I               (A.dim(1).extent() / (VI * III * II))
    #define K               (A.dim(0).extent() / (KKK * KK))

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), X("X", TTYPE, 1);

    // UREs for MV
    Var vi("vi"), iii("iii"), kkk("kkk"), kk("kk"), ii("ii"), k("k"), i("i");
    URE fX("fX", TTYPE, {P}), MV("MV", TTYPE, {P});
    URE TopOut("TopOut"), RightOut("RightOut");
    fX(P) = select(vi == 0, X(total_k), fX(P_vi_minus_1));
    MV(P) = select(kkk == 0 || (vi == VI-1 && iii == III-1), 0,
                    select(vi == VI-1, MV(P_kkk_minus_1_iii_plus_1),
                                       MV(P_kkk_minus_1_vi_plus_1))
                  ) + A(total_k, total_i) * fX(P);
    TopOut(P_top) = select(vi == 0 && iii == 0, MV(P));
    RightOut(P_right) = select(kkk == KKK-1, select(vi == 0 && iii == 0, 0, MV(P)));

    fX.merge_ures(MV, TopOut, RightOut);
    fX.set_bounds(iii, 0, III, kkk, 0, KKK)
      .set_bounds(ii,  0, II,  kk,  0, KK)
      .set_bounds(i,   0, I,   k,   0, K)
      .set_bounds(vi,  0, VI);
    fX.space_time_transform(vi);

    // GPU can have many threads running in parallel.
#ifdef GPU
    X.gpu_blocks(j, i).gpu_threads(jj, ii);
#endif

    // I/O network
    Stensor DA("DA", DRAM), DX("DX", DRAM), SA("SA", SRAM), SX("SX", SRAM);
    Stensor DTopOut("DTopOut", DRAM), DRightOut("DRightOut", DRAM);
    A >> DA.out(vi) >> SA.scope(kk).out(vi) >> fX;
    X >> DX >> SX.scope(kk) >> fX;
    TopOut >> DTopOut;
    RightOut >> DRightOut;
    Stensor::realize(IntelFPGA);

    Func Out("Out", Place::Host);
    Func DTopOutWrapper = DTopOut.get_wrapper_func();
    Func DRightOutWrapper = DRightOut.get_wrapper_func();
    Var flat_dim;
    RDom col(0, VI, 0, III), row(0, KKK);
    Out(flat_dim, ii, kk, i, k) = 0.0f;
    Out(row.x, ii, kk, i, k) += DTopOutWrapper(row.x, ii, kk, i, k);
    Out(col.x + VI*col.y + KKK-1, ii, kk, i, k) += DRightOutWrapper(col.x, col.y, ii, kk, i, k);
    Out.set_bounds(flat_dim, 0, VI*III+KKK-1)
       .set_bounds(ii, 0, II, kk, 0, KK)
       .set_bounds(i, 0, I, k, 0, K);

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    Target acc = get_host_target();
    acc.set_feature(Target::IntelFPGA);
    acc.set_feature(Target::EnableSynthesis);
    Out.compile_to_host("gbmv-interface", { A, X }, "gbmv", acc);
    printf("Success\n");

    return 0;
}
