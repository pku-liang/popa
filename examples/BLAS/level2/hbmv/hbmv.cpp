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
* See the License for the specific language gov
erning permissions
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
    // Dependences for LowMV
    #define P                       vi,       ii,    kk,       i,    k
    #define P_vi_minus_1            vi-1,     ii,    kk,       i,    k
    #define P_kk_minus_1_vi_plus_1  vi+1,     ii,    kk-1,     i,    k
    #define P_kk_minus_1_ii_plus_1  vi-VI+1,  ii+1,  kk-1,     i,    k
    #define P_top                                    kk,       i,    k
    #define P_right                 vi,       ii,              i,    k

    #define P_T                     vi,       ii,    kk,       k,    i
    #define P_T_kk_minus_1          vi,       ii,    kk-1,     k,    i
    #define P_T_k_minus_1           vi,       ii,    kk+KK-1,  k-1,  i
    #define P_up                    vi,       ii,                    i

    #define total_i                 (vi + VI*ii + VI*II*i)
    #define total_k                 (kk + KK*k)

    // Type of the data to process in C and T2S
    #define TTYPE Complex(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), x("x", TTYPE, 1);
    Var vi("vi"), vk("vk"), kk("kk"), ii("ii"), k("k"), i("i");

    // UREs for UpMV
    #define I                       (A.dim(0).extent() / (VI * II))
    #define K                       (A.dim(1).extent() / (KK))
    URE UpMV("UpMV", TTYPE, {P_T}), UpOut("UpOut");
    UpMV(P_T) = select(k == 0 && kk <= 1, 0,
                  select(kk == 0, UpMV(P_T_k_minus_1), UpMV(P_T_kk_minus_1)))
                  + conjugate(A(total_k, total_i)) * x(total_k + total_i);
    UpOut(P_up) = select(kk == KK-1 && k == K-1, UpMV(P_T));

    UpMV.merge_ures(UpOut);
    UpMV.set_bounds(ii,  0, II,  kk,  0, KK)
        .set_bounds(i,   0, I,   k,   0, K)
        .set_bounds(vi,  0, VI);
    UpMV.space_time_transform(vi);
    #undef I
    #undef K

    // UREs for LowMV
    #define I                       (A.dim(1).extent() / (VI * II))
    #define K                       (A.dim(0).extent() / (KK))
    URE LowFx("LowFx", TTYPE, {P}), LowMV("LowMV", TTYPE, {P});
    URE LowTop("LowTop"), LowOut("LowOut");
    LowFx(P) = select(vi == 0, x(total_k), LowFx(P_vi_minus_1));
    LowMV(P) = select(kk == 0 || (vi == VI-1 && ii == II-1), 0,
                    select(vi == VI-1, LowMV(P_kk_minus_1_ii_plus_1),
                                       LowMV(P_kk_minus_1_vi_plus_1))
                    ) + A(total_k, total_i) * LowFx(P);
    LowTop(P_top) = select(ii == 0 && vi == 0, LowMV(P));
    LowOut(P_right) = select(kk == KK-1, select(ii == 0 && vi == 0, 0, LowMV(P)));

    LowFx.merge_ures(LowMV, LowTop, LowOut);
    LowFx.set_bounds(ii,  0, II,  kk,  0, KK)
         .set_bounds(i,   0, I,   k,   0, K)
         .set_bounds(vi,  0, VI);
    LowFx.space_time_transform(vi);

    Stensor DA("DA", DRAM), SA("SA", SRAM);
    Stensor DX_Up("DX_Up", DRAM), DX_Low("DX_Low", DRAM);
    Stensor SX_Up("SX_Up", SRAM), SX_Low("SX_Low", SRAM);
    Stensor DLowTop("DLowTop", DRAM), DLowOut("DLowOut", DRAM), DUpOut("DUpOut", DRAM);
    A >> DA.out(vi) >> FIFO(128) >> vector<FuncOrStensor>{LowMV, SA};
    SA.scope(k).transpose().out(vi) >> FIFO(128) >> UpMV;
    x >> vector<FuncOrStensor>{DX_Up, DX_Low};
    DX_Up >> SX_Up.scope(k) >> FIFO(128) >> UpMV;
    DX_Low >> SX_Low.scope(i) >> FIFO(128) >> LowFx;
    UpOut >> FIFO(128) >> DUpOut;
    LowTop >> FIFO(128) >> DLowTop, LowOut >> FIFO(128) >> DLowOut;
    Stensor::realize(IntelFPGA);

    Func Out("Out", Place::Host);
    Func DLowTopWrapper = DLowTop.get_wrapper_func();
    Func DLowOutWrapper = DLowOut.get_wrapper_func();
    Func DUpOutWrapper = DUpOut.get_wrapper_func();
    Var flat_dim;
    RDom lt(0, KK, 0, I, 0, K), lo(0, VI, 0, II, 0, I, 0, K), uo(0, VI, 0, II, 0, K);
    Out(flat_dim) = complex32_t(0.0f, 0.0f);
    Out(lt.x + lt.y*VI*II + lt.z*KK) += DLowTopWrapper(lt.x, lt.y, lt.z);
    Out(lo.x + lo.y*VI + KK-1 + lo.z*I*II + lo.w*KK) += DLowOutWrapper(lo.x, lo.y, lo.z, lo.w);
    Out(uo.x + uo.y*VI + uo.z*VI*II) += DUpOutWrapper(uo.x, uo.y, uo.z);
    Out.set_bounds(flat_dim, 0, VI*II*I+K*KK-1);

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    Target acc = get_host_target();
    acc.set_feature(Target::IntelFPGA);
    acc.set_feature(Target::EnableSynthesis);
    Out.compile_to_host("hbmv-interface", { A, x }, "hbmv", acc);
    printf("Success\n");

    return 0;
}
