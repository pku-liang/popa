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
    #define P_1             ii,   i,   b
    #define P_1_i_minus_1   ii,   i-1, b
    #define P_2             ii,        b
    #define P_2_ii_minus_1  ii-1,      b

    // Linearized addresses
    #define total_i         (ii + II*i)
    // Outer loop bounds, which are determined by input sizes
    #define I               (x.dim(0).extent()/II)

    #define TTYPE           Float(32)

    // Inputs
    ImageParam x("x", TTYPE, 1);

    // UREs
    Var ii("ii"), i("i"), b("b");
    URE X1("X1"), cmp1v("cmp1v", TTYPE, {P_1}), cmp1i("cmp1i", Int(32), {P_1}), cmp1vout("cmp1vout"), cmp1iout("cmp1iout");
    X1(P_1) = x(total_i);
    cmp1v(P_1) = select(i==0, X1(P_1),
                    select(X1(P_1) > cmp1v(P_1_i_minus_1), X1(P_1), cmp1v(P_1_i_minus_1)));
    cmp1i(P_1) = select(X1(P_1) == cmp1v(P_1), total_i, cmp1i(P_1_i_minus_1));
    cmp1vout(P_2) = select(i == I-1, cmp1v(P_1));
    cmp1iout(P_2) = select(i == I-1, cmp1i(P_1));

    URE X2("X2"), X2i("X2i"), cmp2v("cmp2v", TTYPE, {P_2}), cmp2i("cmp2i", Int(32), {P_2}), out("out");
    X2(P_2) = cmp1vout(P_2);
    X2i(P_2) = cmp1iout(P_2);
    cmp2v(P_2) = select(ii==0, X2(P_2),
                    select(X2(P_2) > cmp2v(P_2_ii_minus_1), X2(P_2), cmp2v(P_2_ii_minus_1)));
    cmp2i(P_2) = select(X2(P_2) == cmp2v(P_2), X2i(P_2), cmp2i(P_2_ii_minus_1));
    out(b) = select(ii == II-1, cmp2i(P_2));

    // Explicitly set the loop bounds
    X1.merge_ures(cmp1v, cmp1i, cmp1vout, cmp1iout);
    X2.merge_ures(X2i, cmp2v, cmp2i, out);
    X1.late_fuse(X2);

    X1.set_bounds(ii, 0, II, i, 0, I, b, 0, 1);
    X1.space_time_transform(ii);
    X2.set_bounds(ii, 0, II, b, 0, 1);
    X2.space_time_transform(ii);

    // I/O network
    Stensor DX("xLoader", DRAM), DZ("unloader", DRAM), imax("imax");
    x >> DX.out(ii) >> FIFO(256);
    out >> DZ >> FIFO(256) >> imax;
    imax.compile_to_host("iamax-interface", {x}, "iamax", IntelFPGA);

    printf("Success\n");
    return 0;
}
