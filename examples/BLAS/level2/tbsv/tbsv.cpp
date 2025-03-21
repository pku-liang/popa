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
    #define P                     ii,      i,   k
    #define P_ii_minus_1          ii-1,    i,   k
    #define P_i_minus_1           ii,      i-1, k
    #define P_prev_psum           ii+1,    i,   k-1
    #define P_prev_boundary_psum  ii-II+1, i+1, k-1
    #define P_Out                               k

    // Linearized addresses
    #define total_i               (ii+II*i)

    // Type of the data to process in C and T2S
    #define CTYPE float
    #define TTYPE Float(32)

    // Inputs
    ImageParam A("A", TTYPE, 2), b("b", TTYPE, 1);

    // UREs
    Var ii("ii"), k("k"), i("i");
    URE uA("uA", TTYPE, {P}), uX("uX", TTYPE, {P}), uY("uY", TTYPE, {P}), Out("Out");
    uA(P) = A(total_i, k);
    uX(P) = select(i == 0,
                select(ii == 0,
                    (b(k) - select(k == 0, 0, uY(P_prev_psum))) / uA(P),
                    uX(P_ii_minus_1)),
                uX(P_i_minus_1));
    uY(P) = select(k == 0, 0,
                select(ii == II-1,
                    select(i == I-1, 0, uY(P_prev_boundary_psum)),
                    uY(P_prev_psum))
            ) + uA(P) * uX(P);
    Out(P_Out) = select(ii==0 && i==0, uX(P));

    // Put all the UREs inside the same loop nest of X.
    uA.merge_ures(uX, uY, Out);

    // Explicitly set the loop bounds
    uA.set_bounds(ii, 0, II,  i, 0, I)
      .set_bounds(k,  0, K);

    // Create a systolic array
    uA.space_time_transform(ii);

    // I/O network
    Stensor DA("aLoader", DRAM);
    Stensor DB("bLoader", DRAM);
    Stensor DX("unloader", DRAM), x("deserializer");
    A >> DA.out(ii) >> FIFO(256);
    b >> DB >> FIFO(256);
    Out >> FIFO(256) >> DX >> x(k);

    // Compile the kernel to an FPGA bitstream, and expose a C interface for the host to invoke
    x.compile_to_host("tbsv-interface", { A, b }, "tbsv", IntelFPGA);
    printf("Success\n");
    return 0;
}
