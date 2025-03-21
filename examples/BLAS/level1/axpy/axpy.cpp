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
    #define P               iii, ii, i
    // Linearized addresses
    #define total_i         (iii + III*ii + III*II*i)
    // Outer loop bounds, which are determined by input sizes
    #define I               (X.dim(0).extent()/(III*II))

    #define TTYPE           Float(32)

    // Inputs
    ImageParam X("X", TTYPE, 1), Y("Y", TTYPE, 1);
    Param<float> alpha;

    // UREs
    Var iii("iii"), ii("ii"), i("i");
    URE Out("out");
    Out(P) = alpha*X(total_i) + Y(total_i);

    // Explicitly set the loop bounds
    Out.set_bounds(iii, 0, III, ii, 0, II, i, 0, I);
    Out.space_time_transform(iii);

    // I/O network
    Stensor DX("xLoader", DRAM, CHANNEL_1), DY("yLoader", DRAM, CHANNEL_2);
    Stensor DC("unloader", DRAM), C("deserializer");
    X >> DX.out(iii) >> FIFO(64);
    Y >> DY.out(iii) >> FIFO(64);
    Out >> FIFO(64) >> DC >> C(total_i);

    C.compile_to_host("axpy-interface", { X, Y, alpha }, "axpy", IntelFPGA);
    printf("Success\n");
    return 0;
}
