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
#ifndef T2S_STANDARDIZE_IR_FOR_OPENCL_H
#define T2S_STANDARDIZE_IR_FOR_OPENCL_H

/** \file
 * Standardize the IR so that later it is straightforward to generate OpenCL code
 */

#include "../../Halide/src/Substitute.h"
#include "../../Halide/src/Scope.h"
#include "../../Halide/src/IR.h"
#include "../../Halide/src/IRMutator.h"
#include "Utilities.h"

namespace Halide {
namespace Internal {

/* Standardize IR so that generating OpenCL code is straightforward: the code generator
 * simply prints whatever the IR is, without doing any smart tricks. Not only this simplifies
 * the code generator, but also improves code readability, as no immediate variable would be
 * blindly generated. */
extern Stmt standardize_ir_for_opencl_code_gen(Stmt s);

/* Halide isolates device code by replacing device for loops with an external call on the host.
 * However, some device-specific constructs are declared outside the device loops. These constructs
 * must be collected by the device codegen to ensure safe removal, thereby facilitating host codegen.*/
class RemoveDeviceDeclaration : public IRMutator {
    using IRMutator::visit;
    SmallStack<std::string> kernels;

public:
    Stmt visit(const Realize *op) override {
        if (kernels.empty()) {
            // Remove nodes out of the scope of kernels
            return mutate(op->body);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const For *op) override {
        if (ends_with(op->name, ".run_on_device")) {
            kernels.push(op->name);
        }
        Stmt s = IRMutator::visit(op);
        if (ends_with(op->name, ".run_on_device")) {
            kernels.pop();
        }
        return s;
    }
};

}
}

#endif
