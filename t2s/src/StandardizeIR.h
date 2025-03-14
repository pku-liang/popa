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

class RemoveIfStmt : public IRMutator {
    using IRMutator::visit;
    bool in_kernel_scope = false;
    std::vector<const For*> for_ops;
    std::set<const For*> boundary_loops;
    Stmt body_of_if_stmt;

    const For* get_target_loop() {
        for (auto l : for_ops) {
            if (boundary_loops.find(l) != boundary_loops.end())
                return l;
        }
        return nullptr;
    }

public:
    Stmt visit(const For *op) override {
        if (ends_with(op->name, ".run_on_device")) {
            in_kernel_scope = true;
        }
        for_ops.push_back(op);
        Stmt body = mutate(op->body);
        auto target_loop = get_target_loop();
        if (op == target_loop) {
            if (body.defined()) {
                // Rebuild unroll loops
                for (auto it = for_ops.rbegin(); *it != target_loop; it++) {
                    const For *cur_op = *it;
                    if (boundary_loops.find(cur_op) == boundary_loops.end()) {
                        std::string loop_name = unique_name("dummy");
                        body_of_if_stmt = substitute(cur_op->name, Variable::make(Int(32), loop_name), body_of_if_stmt);
                        body_of_if_stmt = For::make(loop_name, cur_op->min, cur_op->extent, cur_op->for_type, op->device_api, body_of_if_stmt);
                    }
                }
                body = For::make(op->name, op->min, op->extent, op->for_type, op->device_api, body);
                return Block::make(body, body_of_if_stmt);
            } else {
                return body_of_if_stmt;
            }
        }
        if (ends_with(op->name, ".run_on_device")) {
            in_kernel_scope = false;
            for_ops.clear();
        }
        return For::make(op->name, op->min, op->extent, op->for_type, op->device_api, body);
    }

    Stmt visit(const IfThenElse *op) override {
        if (in_kernel_scope) {
            auto conjuction = break_logic_into_conjunction(op->condition);
            for (auto c : conjuction) {
                auto eq = c.as<EQ>();
                internal_assert(eq);
                auto eq_a = eq->a.as<Variable>();
                internal_assert(eq_a);
                auto it = std::find_if(for_ops.begin(), for_ops.end(), [&](const For *lp){ return lp->name == eq_a->name; });
                boundary_loops.insert(*it);
            }
            internal_assert(!op->else_case.defined());
            body_of_if_stmt = op->then_case;
            return Stmt();
        }
        return IRMutator::visit(op);
    }
};

}
}

#endif
