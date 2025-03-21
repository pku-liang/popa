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
    #define I               (x.dim(0).extent()/II)

    #define TTYPE           Float(32)

    // Inputs
    ImageParam x("x", TTYPE, 1), y("y", TTYPE, 1);
    Param<float> c, s;

    // UREs
    Var ii("ii"), i("i");
    URE X("X"), Y("Y"), OutX("OutX"), OutY("OutY");
    X(P) = x(total_i);
    Y(P) = y(total_i);
    OutX(P) = c*X(P) + s*Y(P);
    OutY(P) = c*Y(P) - s*X(P);

    // Explicitly set the loop bounds
    X.merge_ures(Y, OutX, OutY);
    X.set_bounds(ii, 0, II, i, 0, I);
    X.space_time_transform(ii);

    // I/O network
    Stensor DX("xLoader", DRAM, CHANNEL_1), DY("yLoader", DRAM, CHANNEL_2);
    Stensor DOutX("xUnloader", DRAM, CHANNEL_1), DOutY("yUnloader", DRAM, CHANNEL_2);
    x >> DX.out(ii) >> FIFO(256);
    y >> DY.out(ii) >> FIFO(256);
    OutX >> FIFO(256) >> DOutX.out(ii);
    OutY >> FIFO(256) >> DOutY.out(ii);
    Stensor::realize(IntelFPGA);

    Func fX = DOutX.get_wrapper_func();
    Func fY = DOutY.get_wrapper_func();
    Target acc = get_host_target();
    acc.set_feature(Target::IntelFPGA);
    acc.set_feature(Target::EnableSynthesis);
    Pipeline({fX, fY}).compile_to_host("rot-interface", { x, y, c, s }, "rot", acc);

    printf("Success\n");
    return 0;
}
