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
#include "const-parameters.h"

using namespace Halide;

int main()
{
   // Dependences
    #define P               kkk,        kk,       k,    b
    #define P_kkk_minus_1   kkk-1,      kk,       k,    b
    #define P_kk_minus_1    kkk+KKK-1,  kk-1,     k,    b
    #define P_k_minus_1     kkk+KKK-1,  kk+KK-1,  k-1,  b
    #define P_out                                       b
    // Linearized addresses
    #define total_k         (kkk + KKK * kk + KKK * KK * k)

    // Outer loop bounds, which are determined by input sizes
    #define K (X.dim(0).extent() / (KK * KKK))
    #define B (X.dim(1).extent())

    // Inputs
    ImageParam X("X", TTYPE, 2);

    // UREs
    Var kkk("kkk"), kk("kk"), k("k"), b("b");
    URE uX("uX", TTYPE, {P}), uZ("uZ", TTYPE, {P}), Out("Out");

    uX(P) = X(total_k, b);
    uZ(P) = select(k == 0 && kk == 0 && kkk == 0, 0,
                  select(kkk == 0, select(kk == 0, uZ(P_k_minus_1), uZ(P_kk_minus_1)), uZ(P_kkk_minus_1))) + abs(uX(P));
    Out(P_out) = select(kkk == KKK-1 && kk == KK-1 && k == K-1, uZ(P));

    // Put all the UREs inside the same loop nest of X.
    uX.merge_ures(uZ, Out);
    uX.set_bounds(kkk,  0, KKK, kk,  0, KK,  k,  0, K)
      .set_bounds(b,    0, 1);
    uX.space_time_transform(kkk);

    // I/O network
    Stensor DX("xLoader", DRAM), DC("unloader", DRAM), C("deserializer");
    X >> DX.out(kkk) >> FIFO(256);
    Out >> FIFO(256) >> DC >> C(b);

    C.compile_to_host("asum-interface", { X }, "asum", IntelFPGA);
    printf("Success\n");
    return 0;
}
