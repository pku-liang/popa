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
#include "../../Halide/src/IRMutator.h"
#include "../../Halide/src/IRVisitor.h"
#include "../../Halide/src/Simplify.h"
#include "../../Halide/src/IREquality.h"
#include "./StructType.h"
#include "./PatternMatcher.h"
#include "./Utilities.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;

/* The original inner product operation is expressed in UREs as follows: Z(k, ...) = select(k == 0, 0, Z(k-1, ...)) + A * B
 * The guarding condition (k == 0) is tested at each iteration, which may confuse the backend compilers to find an optimized IP.
 * We automatically detect such pattern and eliminate the guarding condition. Specifially, the lowered code seems like:
 * Z.temp = 0                   // (1) This temporary variable is used for reduction and initialized with 0 (true value in select)
 * for (k, 0, K) {
 *  Z.temp = Z.temp + A * B     // (2) The inner product operation.
 * }
 * Z(0, ...) = Z.temp           // Write back. After MinimizeShregs phase, only one register is allocated for reduction.
 */
class InnerProductMatcher : public IRMutator
{
    struct InnerProduct {
        string name;            // The temporary variable used for reduction
        string sink_loop;       // Move the initial part outside of this loop
        Type type;              // Type of the variable
        Expr init_value;        // Expr to initialize temporary variable (1)
        Expr update_value;      // Expr to update temporary variable (2)
        const Call *ori_call;   // The original write_shift_reg call for inner product
    };
    vector<string> loops;
    vector<InnerProduct> inner_products;
    vector<std::pair<string, Type>> allocs;
    Stmt update;                // Stmt to replace write_shift_reg call (passed to enclosing Evaluate node)

    bool find_inner_product(string w_name, Expr w_value, vector<Expr> w_dims) {
        // An inner product usually contains Add node whose lhs is a select
        auto add = w_value.as<Add>();
        auto sel = add ? add->a.as<Select>() : 0;
        if (!add || !sel) {
            return false;
        }
        vector<Expr> conds = break_logic_into_conjunction(sel->condition);
        string sink_loop = "";
        // This process is similar to loop-invariant hoisting
        // The statement executed only once can be moved outside the loop body
        for (auto l = loops.rbegin(); l != loops.rend(); ++l) {
            string temp = sink_loop;
            // A boundary conjuction like l==0 and l is local to a register (does not appear in w_dims)
            for (auto c = conds.begin(); c != conds.end(); ++c) {
                const EQ *eq = c->as<EQ>();
                if (!eq) {
                    continue;
                }
                auto a = eq->a.as<Variable>();
                auto b = eq->b.as<IntImm>();
                if ((a && a->name == *l) && (b && b->value == 0)) {
                    bool find_var = false;
                    for (auto &d : w_dims) {
                        auto v = d.as<Variable>();
                        if (v && v->name == a->name) {
                            find_var = true;
                        }
                    }
                    if (!find_var) {
                        sink_loop = a->name;
                        c = conds.erase(c);
                        break;
                    }
                }
            }
            if (temp == sink_loop) {
                // We cannot further hoist the statement as no boundary conjunction exists.
                break;
            }
        }
        // Check if the false expr is to read the last value
        bool is_reduce = true;
        auto read_call = sel->false_value.as<Call>();
        if (read_call && read_call->is_intrinsic(Call::read_shift_reg)) {
            string r_name = read_call->args[0].as<StringImm>()->value;
            vector<Expr> r_dims(read_call->args.begin()+1, read_call->args.end());
            // Read the same register with the same dimensions
            if (r_name == w_name) {
                internal_assert(w_dims.size() == r_dims.size());
                for (size_t i = 0; i < w_dims.size(); i++) {
                    if (!equal(w_dims[i], r_dims[i])) {
                        is_reduce = false;
                    }
                }
            }
        }
        if (!sink_loop.empty() && is_reduce) {
            // The remaining conjuctions (after removing boundary conjunctions)
            // are used to generate new guarding condition
            Expr new_cond = const_true();
            for (auto &c : conds) {
                new_cond = new_cond && c;
            }

            InnerProduct tmp;
            tmp.type = add->a.type();
            tmp.name = unique_name(w_name + ".temp");
            tmp.sink_loop  = sink_loop;
            Expr fpga_reg  = Call::make(tmp.type, Call::IntrinsicOp::fpga_reg, {sel->false_value, 1}, Call::CallType::PureIntrinsic);
            tmp.init_value = Select::make(simplify(new_cond), sel->true_value, fpga_reg);
            tmp.update_value = Call::make(tmp.type, tmp.name, {}, Call::Intrinsic) + add->b;
            tmp.ori_call = NULL; // To be instantiated later.
            inner_products.push_back(std::move(tmp));
            return true;
        }
        return false;
    }

public:
    using IRMutator::visit;

    Expr visit(const Call *op) override {
        if (op->is_intrinsic(Call::write_shift_reg)) {
            string name = op->args[0].as<StringImm>()->value;
            vector<Expr> dims(op->args.begin()+1, op->args.end()-1);
            Expr value = op->args.back();
            if (find_inner_product(name, value, dims)) {
                // The initial and write-back parts are hoisted outside the loop,
                // so only the update part stays here
                auto &tmp = inner_products.back();
                tmp.ori_call = op;
                update = Provide::make(tmp.name, { tmp.update_value }, {});
                return 0;
            }
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Evaluate *op) override {
        Expr value = mutate(op->value);
        if (update.defined()) {
            // The original write_shift_reg is replaced with the update statement
            Stmt tmp = update;
            update = Stmt();
            return tmp;
        }
        return Evaluate::make(value);
    }

    Stmt visit(const For *op) override {
        vector<InnerProduct> backup;
        inner_products.swap(backup);

        loops.push_back(op->name);
        Stmt body = mutate(op->body);
        loops.pop_back();

        if (ends_with(op->name, "run_on_device")) {
            for (auto &p : allocs) {
                // The allocation of temporary variables is inserted at the top of a kernel
                body = Realize::make(p.first, {p.second}, MemoryType::Auto, {}, const_true(), body);
            }
            allocs.clear();
        }
        // Breaks up dot-8 and larger into dot-4s using fpga_reg
        for (auto it = inner_products.begin(); it != inner_products.end(); ++it) {
            if (it->sink_loop != op->name) continue;
            Expr tmp = Call::make(it->type, it->name, {}, Call::Intrinsic);
            Expr fpga_reg = Call::make(it->type, Call::IntrinsicOp::fpga_reg, {tmp, 1}, Call::CallType::PureIntrinsic);
            Stmt tmp_self = Provide::make(it->name, { fpga_reg }, {});
            Expr cond = Variable::make(Int(32), op->name) % 4 == 3;
            Stmt if_stmt = IfThenElse::make(cond, tmp_self);
            body = Block::make(body, if_stmt);
        }
        body = For::make(op->name, op->min, op->extent,
                         op->for_type, op->device_api, body);
        for (auto it = inner_products.begin(); it != inner_products.end(); ) {
            if (it->sink_loop != op->name) {
                ++it;
                continue;
            }
            // Above the loop body, we initialize the temporary variable
            Expr value = it->init_value;
            Stmt init = Provide::make(it->name, { it->init_value }, {});
            body = Block::make(init, body);
            // Below the loop body, we write back the temporary variable
            auto call = it->ori_call;
            vector<Expr> call_args(call->args.begin(), call->args.end()-1);
            call_args.push_back(Call::make(it->type, it->name, {}, Call::Intrinsic));
            Expr write_back = Call::make(call->type, Call::write_shift_reg, call_args, Call::Intrinsic);
            body = Block::make(body, Evaluate::make(write_back));
            // Putting the allocation of temporary variables togther
            allocs.push_back({ it->name, it->type });
            it = inner_products.erase(it);
        }
        inner_products.insert(inner_products.begin(), backup.begin(), backup.end());
        return body;
    }
};

class PartitionMatcher : public IRMutator
{
    vector<PartitionItem> v_param;
    std::map<string, Expr> original_node;
    const std::map<string, Function> &env;
public:
    using IRMutator::visit;
    PartitionMatcher(const std::map<string, Function> &_e) : env(_e) {}

    Stmt visit(const ProducerConsumer *op) override {
        Function func;
        if (op->is_producer && function_is_in_environment(op->name, env, func)) {
            auto &param = func.definition().schedule().partition_params();
            if (!param.empty()) {
                internal_assert(param.size() == 1);
                v_param.push_back(param[0]);
            }
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const For *op) override {
        Stmt body = mutate(op->body);
        string func_name = extract_first_token(op->name);
        auto it = std::find_if(v_param.begin(), v_param.end(), [&](const PartitionItem &p){ return p.consumer == func_name; });
        if (it != v_param.end() && extract_last_token(op->name) == it->loop_name) {
            internal_assert(original_node.count(func_name) > 0);
            Expr write_val = original_node.at(func_name);
            string tmp_array_name = func_name + ".temp";
            string tmp_loop_name = op->name + ".t";
            Stmt write_temp;
            auto extent = op->extent.as<IntImm>();
            internal_assert(extent);
            for (int i = 0; i < extent->value; i += it->stride) {
                Expr curr = write_val;
                if (write_val.as<Select>()) {
                    Expr expected_cond = Variable::make(Int(32), op->name) < (i+it->stride);
                    internal_assert(equal(expected_cond, write_val.as<Select>()->condition));
                    curr = write_val.as<Select>()->true_value;
                    write_val = write_val.as<Select>()->false_value;
                }
                Expr write_idx = Variable::make(Int(32), tmp_loop_name) + i;
                Stmt write_node = Provide::make(tmp_array_name, {curr}, {write_idx});
                write_temp = write_temp.defined() ? Block::make(write_temp, write_node) : write_node;
            }
            write_temp = For::make(tmp_loop_name, 0, it->stride, ForType::Unrolled, op->device_api, write_temp);
            body = For::make(op->name, op->min, op->extent, op->for_type, op->device_api, body);
            body = Block::make(write_temp, body);
            Stmt realize_temp = Realize::make(tmp_array_name, { write_val.type() }, MemoryType::Auto,
                                              { Range(op->min, op->extent) }, const_true(), body);
            return realize_temp;
        }
        return For::make(op->name, op->min, op->extent, op->for_type, op->device_api, body);
    }

    Stmt visit(const Provide *op) override {
        auto it = std::find_if(v_param.begin(), v_param.end(), [&](const PartitionItem &p){ return p.consumer == op->name; });
        if (it != v_param.end()) {
            // Replace the reference to the original value with the temporary value
            internal_assert(op->values.size() == 1 && op->values[0].as<Select>());
            original_node[op->name] = op->values[0];
            auto tmp_array_name = it->consumer + ".temp";
            auto loop_var = Variable::make(Int(32), it->consumer + ".s0." + it->loop_name);
            Expr read_temp = Call::make(op->values[0].type(), tmp_array_name, { loop_var }, Call::Intrinsic);
            return Provide::make(op->name, { read_temp }, op->args);
        }
        return IRMutator::visit(op);
    }

};

Stmt rewrite_memory_partition(Stmt s, const std::map<string, Function> &env) {
    PartitionMatcher pm(env);
    s = pm.mutate(s);
    return s;
}

Stmt match_patterns(Stmt s) {
    InnerProductMatcher ipm;
    s = ipm.mutate(s);
    return s;
}

}
}
