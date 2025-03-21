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
    #define P               ii, i
    // Linearized addresses
    #define total_i         (ii + II*i)
    // Outer loop bounds, which are determined by input sizes
    #define I               (X.dim(0).extent()/II)

    #define TTYPE           Float(32)

    // Inputs
    ImageParam X("X", TTYPE, 1);
    Param<float> alpha;

    // UREs
    Var ii("ii"), i("i");
    URE Out("out");
    Out(P) = alpha*X(total_i);

    // Explicitly set the loop bounds
    Out.set_bounds(ii, 0, II, i, 0, I);
    Out.space_time_transform(ii);

    // I/O network
    Stensor DX("xLoader", DRAM, CHANNEL_1), DC("unloader", DRAM, CHANNEL_2);
    Stensor C("deserializer");
    X >> DX.out(ii) >> FIFO(256);
    Out >> FIFO(256) >> DC >> C(total_i);

    C.compile_to_host("scal-interface", { X, alpha }, "scal", IntelFPGA);
    printf("Success\n");
    return 0;
}
