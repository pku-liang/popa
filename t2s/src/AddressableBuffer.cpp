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

#include <set>
#include <string>
#include <queue>
#include <map>
#include "../../Halide/src/Definition.h"
#include "../../Halide/src/Expr.h"
#include "../../Halide/src/Func.h"
#include "../../Halide/src/Function.h"
#include "../../Halide/src/IREquality.h"
#include "../../Halide/src/IRMutator.h"
#include "../../Halide/src/IRVisitor.h"
#include "../../Halide/src/Schedule.h"
#include "../../Halide/src/Simplify.h"
#include "../../Halide/src/Substitute.h"
#include "../../Halide/src/Util.h"
#include "./AddressableBuffer.h"
#include "./DebugPrint.h"
#include "./Utilities.h"

namespace Halide{

Func &Func::addressable_buffer(Func f, VarOrRVar buffer_loop, vector<Expr> write_indices, vector<Expr> read_indices, BufferStrategy strategy) {
    invalidate_cache();
    user_assert(this->defined()) << "Func " << this->name() << " is undefined";
    user_assert(f.defined()) << "Func " << f.name() << " is undefined";
    user_assert(!write_indices.empty()) << "Access index of the buffer is undefined.";
    if (read_indices.empty()) {
        read_indices = write_indices;
    }
    // Indices are from inner outer, but buffer dimensions are from outer inner
    std::reverse(write_indices.begin(), write_indices.end());
    std::reverse(read_indices.begin(), read_indices.end());
    std::vector<Internal::AddressableBufferItem> &buffer_params = func.definition().schedule().addressable_buffer_params();
    user_assert(buffer_params.empty())
        << "Inserting more than 1 buffer to Func " << func.name() << " is unexpected. We support only one buffer in a function so far\n";
    buffer_params.push_back(Internal::AddressableBufferItem(f.name(), buffer_loop.name(), write_indices, read_indices, strategy));
    return *this;
}

namespace Internal{

namespace {

using std::tuple;

// Info in a function tha buffers and/or scatters data
class AddressableBufferArg{
public:
    string producer = "";          // The Func that sends data to buffer and/or scatter
    string buffer_loop = "";       // Loop under which a buffer is to be inserted. Undocorated name without func and stage prefix, e.g. "x"
    string scatter_loop = "";      // Loop along which data are to be scattered.  Undocorated name without func and stage prefix, e.g. "y"
    vector<Expr> write_indices;    // The indices to write into the buffer, which indicates the storage layout
    vector<Expr> read_indices;     // The indices to read from the buffer
    Expr read_node = nullptr;      // The expression that reads the data from the producer
    Expr write_node = nullptr;     // // The expression that writes the data to the consumer
    Expr read_condition = nullptr; // Path conditioin to read_node. It is the same condition the producer writes into its output channel.
    BufferStrategy buffer_strategy = BufferStrategy::Double;
    ScatterStrategy scatter_strategy = ScatterStrategy::Up; 

    // Full names, mins, extents and types of the loops around the read node.
    vector<tuple<string, Expr, Expr, ForType>> loops;
};
using AddressableBufferArgs = map<string, AddressableBufferArg>;

void get_AddressableBufferArgs(const map<string, Function> &env, AddressableBufferArgs& args) {
    for (const auto &e : env) {
        Function func =  e.second;
        auto buffer_params = func.definition().schedule().addressable_buffer_params();
        auto scatter_params = func.definition().schedule().scatter_params();
        if (buffer_params.empty()) {
            continue;
        }
        // Scatter is optional
        internal_assert(buffer_params.size() == 1 && scatter_params.size() <= 1);

        if (!scatter_params.empty()) {
            user_assert(buffer_params[0].func_name == scatter_params[0].func_name)
                <<"Func " << func.name() << " buffers data from Func " << buffer_params[0].func_name
                << ", but scatters data from another Func " << scatter_params[0].func_name
                << ". We require the data to buffer and scatter are from the same producer.";
        }

        AddressableBufferArg tmp;
        tmp.producer = buffer_params[0].func_name;
        tmp.buffer_loop = buffer_params[0].buffer_loop;
        tmp.write_indices = buffer_params[0].write_indices;
        tmp.read_indices = buffer_params[0].read_indices;
        tmp.buffer_strategy = buffer_params[0].strategy;
        if (!scatter_params.empty()) {
            tmp.scatter_loop = scatter_params[0].loop_name;
            tmp.scatter_strategy = scatter_params[0].strategy;
        }

        debug(4) <<  tmp.producer << " --> " << func.name() << ": "
                 << "parallel access buffer at loop ("<< tmp.buffer_loop << "), "
                 << "scatter along loop (" << tmp.scatter_loop << ")\n";
        args.insert({func.name(), tmp});
    }
    return;
}

// Check assumptions and collect info to be used for inserting parallel access buffer
class AddressableBufferChecker: public IRVisitor {
    using IRVisitor::visit;

private:
    string func_name; // A function where a buffer is to be inserted
    AddressableBufferArgs &args;
    const map<string,Function>& env;
    vector<tuple<string, Expr, Expr, ForType>> &all_loops; // Full name, min, extent and type of all loops in the IR

    // Temporaries for the current Func
    vector<tuple<string, Expr, Expr, ForType>> loops; // Full name, min, extent and type of the loops
                                                      // around the read node in the current Func
    bool buffer_loop_seen; // True after a buffer loop has been seen for the current Func
    Expr path_condition; // Path condition to the current IR.

public:
    AddressableBufferChecker(AddressableBufferArgs& _args, const map<string, Function>& _env,
                         vector<tuple<string, Expr, Expr, ForType>> &_all_loops):
        args(_args), env(_env), all_loops(_all_loops) { func_name = ""; }

    void visit(const ProducerConsumer *op) override {
        if (op->is_producer && args.find(op->name) != args.end()) {
            func_name = op->name;
            loops.clear();
            buffer_loop_seen = false;
            path_condition = const_true();
            IRVisitor::visit(op);
            func_name = "";
            return;
        }
        IRVisitor::visit(op);
        return;
    }

    void visit(const Call *op) override {
        if (func_name != "") {
            auto iter = args.find(func_name);
            string producer = iter->second.producer;
            if (op->is_intrinsic(Call::read_channel)) {
                string chn_name = producer + "." + func_name + ".channel";
                internal_assert(op->args[0].as<StringImm>());
                if (chn_name == op->args[0].as<StringImm>()->value) {
                    internal_assert(!iter->second.read_node.defined());
                    iter->second.read_node = op;
                    iter->second.read_condition = path_condition;
                    iter->second.loops = loops;
                }
            }
            if (op->is_intrinsic(Call::write_channel)) {
                internal_assert(op->args[0].as<StringImm>());
                string chn_name = op->args[0].as<StringImm>()->value;
                if (extract_first_token(chn_name) == func_name) {
                    iter->second.write_node = op;
                }
            }
        }
        IRVisitor::visit(op);
        return;
    }

    void visit(const For* op) override {
        // The scattering/buffering phase is placed after vectorization and vectorized loops are out of concern.
        internal_assert(op->for_type != ForType::Vectorized);

        IRMutator mutator;
        Expr min = mutator.mutate(op->min);
        Expr extent = mutator.mutate(op->extent);
        if (!ends_with(op->name, ".run_on_device")) {
            all_loops.push_back(tuple<string, Expr, Expr, ForType>(op->name, min, extent, op->for_type));
        }

        if (func_name != "") {
            auto iter = args.find(func_name);
            string scatter_loop = iter->second.scatter_loop;
            string buffer_loop = iter->second.buffer_loop;
            auto remove_iter = env.find(iter->second.producer);
            if (remove_iter != env.end()) {
                auto remove_params = remove_iter->second.definition().schedule().remove_params();
                for (auto remove_param : remove_params) {
                    if (ends_with(op->name,"." + remove_param)) {
                        // This loop is removed in the producer side. In the consumer side, data should
                        // be buffered first, and then this loop READS the buffer repetitively so that
                        // the removal of the loop in the producer does not affect the data read.
                        // In other words, the buffer loop should be before this loop.
                        user_assert(buffer_loop != "")
                            << "Loop " << remove_param << " is removed in Func " << iter->second.producer
                            << " (producer), and a buffer is expected to insert in Func " << func_name
                            << " (consumer).\n";
                        user_assert(buffer_loop_seen)
                            << "Loop " << remove_param << " is removed in Func " << iter->second.producer
                            << " (producer), and a buffer is expected to insert in Func " << func_name
                            << " (consumer) at an outer loop level. However, the current loop to insert the buffer"
                            << ", " << buffer_loop << ", is not at an outer loop of loop " << remove_param << "\n";
                    }
                }
            }
            if (scatter_loop != "" && ends_with(op->name,"." + scatter_loop)) {
                user_assert(op->for_type == ForType::Unrolled)
                    << func_name << " scatters data from " << iter->second.producer << " along loop " << scatter_loop
                    << ". The loop is expected to be unrolled.\n";
                user_assert(op->min.as<IntImm>() && op->extent.as<IntImm>())
                    << func_name << " scatters data from " << iter->second.producer << " along loop " << scatter_loop
                    << ". The loop is expected to have constant bounds.\n";
            }
            if (!ends_with(op->name, ".run_on_device")) {
                loops.push_back(tuple<string, Expr, Expr, ForType>(op->name, min, extent, op->for_type));
                if (buffer_loop != "" && ends_with(op->name,"." + buffer_loop)) {
                    buffer_loop_seen = true;
                }
            }

            IRVisitor::visit(op);

            if (!ends_with(op->name, ".run_on_device")) {
                loops.pop_back();
            }
        } else {
            IRVisitor::visit(op);
        }
    }

    void visit(const Load *op) override {
        if (func_name != "") {
            auto iter = args.find(func_name);
            if (iter->second.buffer_loop != "") {
                string producer = iter->second.producer;
                if (producer == op->name + "_im" || producer == op->name) {
                    user_assert(false)
                        << iter->second.producer << " sends data to " << func_name << " to buffer "
                        << " via memory, which is unexpected. The data should be transferred via a channel "
                        << " instead. To avoid this error, make sure both " << iter->second.producer
                        << " and " << func_name << " are declared with Place::Device.";
                }
            }
            // if (iter->second.scatter_loop != "") {
            //     string producer = iter->second.producer;
            //     if (producer == op->name + "_im" || producer == op->name) {
            //         internal_assert(!iter->second.read_node.defined());
            //         iter->second.read_node = op;
            //         iter->second.read_condition = path_condition;
            //         iter->second.loops = loops;
            //     }
            // }
        }
        IRVisitor::visit(op);
        return;
    }

    void visit(const IfThenElse *op) override {
        if(func_name != ""){
            Expr old_condition = path_condition;

            path_condition = equal(old_condition, const_true()) ? op->condition : old_condition && op->condition;
            op->then_case.accept(this);

            if (op->else_case.defined()) {
                path_condition = equal(old_condition, const_true()) ? !op->condition : old_condition && !op->condition;
                op->else_case.accept(this);
            }

            path_condition = old_condition;
        } else {
            IRVisitor::visit(op);
        }
    }

    void visit(const Select *op) override {
        if(func_name != ""){
            Expr old_condition = path_condition;

            path_condition = equal(old_condition, const_true()) ? op->condition : old_condition && op->condition;
            op->true_value.accept(this);

            if (op->false_value.defined()) {
                path_condition = equal(old_condition, const_true()) ? !op->condition : old_condition && !op->condition;
                op->false_value.accept(this);
            }

            path_condition = old_condition;
        } else {
            IRVisitor::visit(op);
        }
    }
};

// Transform IR to buffer and scatter incoming data. The code closely follows this document:
//   ../doc/compiler_design/buffer.md
// So read the document first in order to understand the code.
class AddressableBuffer: public IRMutator{
    using IRMutator::visit;
private:
    // Input info of the entire IR, not just the current Func.
    const map<string, Function>& envs;
    const vector<tuple<string, Expr, Expr, ForType>> &all_loops; // Full names, mins, extents and types of loops

    // Info of the current Func
    const string &func_name; // The function that buffers and scatters incoming data
    const string &producer;  // The Func that sends data to buffer and scatter
    const string &buffer_loop; // Loop under which a buffer is to be inserted. Undocorated name without func and stage prefix, e.g. "x"
    const string &scatter_loop; // Loop along which data are to be scattered.  Undocorated name without func and stage prefix, e.g. "y"
    const vector<Expr> &write_indices;  // The indices to write into the buffer
    const vector<Expr> &read_indices;   // The indices to read from the buffer
    const Expr &original_read_node; // The expression in the consumer that reads the data from the producer
    const Expr &original_write_node; // The expression in the consumer that writes the data to the next consumer
    const Expr &original_read_condition; // Path condition to the original_read_node. It is the same condition the producer writes to its output channel.
    const BufferStrategy buffer_strategy;
    const ScatterStrategy scatter_strategy;
    const vector<tuple<string, Expr, Expr, ForType>> &loops; // Full names, mins, extents and types of the loops around the read node.

private:
    // Frequently used constants and variables used in transforming the IR.
    Type TYPE;               // Type of the incoming data
    Type vector_type;        // Type of the incoming array of data, only available for ForwardVector strategy
    vector<int> WRITE_LOOPS; // Serial loops in the producer below the buffer insertion point. Elements: indices to loops
    vector<int> READ_LOOPS;  // Serial loops in the consumer below the buffer insertion point. Elements: indices to loops
    uint32_t WRITES;         // Product of the extents of WRITE_LOOPS
    uint32_t READS;          // Product of the extents of READ_LOOPS
    int32_t BUFFERS;         // Number of double buffers (i.e. extent of the scatter loop)
    uint32_t CYCLES_PER_PERIOD;
    uint32_t INIT;           // The cycle in a period when buffer writing should start.
    Expr PERIODS;            // Total periods
    Expr cycle;              // Current cycle
    Expr in_v;               // Incoming value read from the input channel in the current cycle
    Expr out_v;              // Outgoing value written into the output channel
    Expr value;              // Incoming value stored in shift registers for scattering
    Expr time_stamp;         // Cycle stored in shift registers for scattering
    Expr buf_loop_var;
    Expr period;
    Expr offset;
    Expr time_to_write_buffer;

    int original_scatter_loop;             // Index to loops for the full name of the scatter loop in the consumer
    bool scatter_loop_removed_in_producer; // The scatter loop is removed in the producer?
    Expr original_scatter_loop_var;

    // As an extension to our design document, besides the scatter loop, we allow other unroll loops to exist.
    // Every iteration of these other unroll loops has its own double buffers that store and scatter its own
    // data. All the iterations of these other unroll loops are independent from each other.
    // See the comments for visit_innermost_loop() for details.
    vector<int> unroll_loops;                 // All unroll loops. Elements are indices to loops.
    vector<Expr> unroll_loop_vars;            // Variables of unroll_loops
    Region unroll_loop_dims;                  // Bounds of unroll_loops
    vector<int> nonscatter_unroll_loops;      // All unroll loops except the scatter loop. Elements are indices to loops.
    vector<Expr> nonscatter_unroll_loop_vars; // Variables of nonscatter_unroll_loops
    Region nonscatter_unroll_loop_dims;       // Bounds of nonscatter_unroll_loops

    // Variables for a single buffer:
    Expr _cycle;
    Expr _period;
    Expr _offset;
    Expr _time_to_write_buffer;
    Expr _owner;
    Expr _idx;
    Expr _time_to_read;

    // The incoming data type might be a compiler-generated struct, which contains
    // multiple fields, and each field might differ in their degrees of reuse. Therefore,
    // we can allocate separate buffers for them, with sizes decided by their degrees
    // of reuse separately. This will minimize the total sizes of memory allocation.
    typedef struct {
        string name;             // Name of this buffer (for one field)
        Type   type;             // Element type
        Region dims;             // [2][extents of NonScatter_NonReuse_Write_Loops][extents of nonscatter_unroll_loops][BANKS],
                                 // where NonScatter_NonReuse_Write_Loops = (WRITE_LOOPS - REUSE_WRITE_LOOPS - scatter loop),
                                 // where REUSE_WRITE_LOOPS are subset of WRITE_LOOPS that are not referred by the field.
        int32_t num_banks;
        int32_t bank_bits;
        bool parallel_access;    // If read with transposed layout, it is implemented as a parallel access buffer
        Expr skew_factor;        // For the first, second, ... rows, skew left by one, two, ... times
        vector<Expr> write_args; // [_idx][WRITE_TO(_offset)][nonscatter_unroll_loops][buf], where WRITE_TO(_offset) is the address determined by
                                 // the NonScatter_NonReuse_Write_Loops out of the WRITE_LOOPS
        vector<Expr> read_args;  // DB[!_idx][READ_FROM(_offset)][nonscatter_unroll_loops][buf], where READ_FROM(_offset) is the address determined by
                                 // the NonScatter_NonReuse_Write_Loops out of the READ_LOOPS
    } buffer_info;
    vector<buffer_info> buffers_info; // Buffer info for all fields

public:
    AddressableBuffer(
        const map<string, Function>& envs,
        const vector<tuple<string, Expr, Expr, ForType>> &all_loops,
        const string &func_name,
        const string &producer,
        const string &buffer_loop,
        const string &scatter_loop,
        const vector<Expr> &write_indices,
        const vector<Expr> &read_indices,
        const Expr &original_read_node,
        const Expr &original_write_node,
        const Expr &original_read_condition,
        const BufferStrategy buffer_strategy,
        const ScatterStrategy scatter_strategy,
        const vector<tuple<string, Expr, Expr, ForType>> &loops) :
            envs(envs), all_loops(all_loops), func_name(func_name), producer(producer),
            buffer_loop(buffer_loop), scatter_loop(scatter_loop),
            write_indices(write_indices), read_indices(read_indices),
            original_read_node(original_read_node), original_write_node(original_write_node),
            original_read_condition(original_read_condition),
            buffer_strategy(buffer_strategy), scatter_strategy(scatter_strategy), loops(loops) {
                initialize_common_constants_vars();
    }

private:
    const string &var_name(const Expr &e) const {
        const Variable *v = e.as<Variable>();
        internal_assert(v);
        return v->name;
    }

    string replace_postfix(const string &str, const string &postfix, const string &replacement) {
        internal_assert(ends_with(str, postfix));
        return str.substr(0, str.size() - postfix.size()) + replacement;
    }

    // This class does not really mutate anything. It inherits IRMutator to take advantage of mutate() functions.
    class VarReferred : public IRMutator {
        using IRMutator::visit;
    private:
        const string &var;
    public:
        bool referred;
        VarReferred(const string &var) : var(var) {
            referred = false;
        }
    public:
        Expr mutate(const Expr &e) override {
            if (referred) {
                return e;
            }
            const Variable *v = e.as<Variable>();
            if (v) {
                if (v->name == var) {
                    referred = true;
                    return e;
                }
            }
            return IRMutator::mutate(e);
        }
        Stmt mutate(const Stmt &s) override {
            if (referred) {
                return s;
            }
            return IRMutator::mutate(s);
        }
    };

    // See if a loop variable is used in the expression
    bool loop_referred(const string &loop, const Expr &e) {
        VarReferred ref(loop);
        ref.mutate(e);
        return ref.referred;
    }

    // The correponding loop name.
    string producer_loop_name(const string &loop_name) {
        const Function &func = envs.find(producer)->second;
        string ploop = func.name() + ".s0." + extract_after_tokens(loop_name, 2);
        return ploop;
    }

    // The loop is removed?
    bool loop_is_removed(const string &loop_name) {
        for (auto &l : all_loops) {
            if (loop_name == std::get<0>(l)) {
                return false;
            }
        }
        return true;
    }

    void intialize_common_constants() {
        internal_assert(original_read_node.defined());
        TYPE = original_read_node.type();
        WRITES = 1, READS = 1, PERIODS = 1;
        scatter_loop_removed_in_producer = false;
        bool inside_buffer_loop = false;

        for (size_t i = 0; i < loops.size(); i++) {
            auto &l = loops[i];
            string loop_name = std::get<0>(l);
            Expr loop_extent = std::get<2>(l);
            ForType for_type = std::get<3>(l);
            if (inside_buffer_loop) {
                string producer_loop = producer_loop_name(loop_name); // Corresponding loop in producer
                if(!loop_is_removed(producer_loop)) {
                    if (for_type != ForType::Unrolled) {
                        // NOTE: we check if a loop is serial based on its current for_type in the consumer.
                        // So WRITE_LOOPS are those that are not removed in the producer, and currently
                        // serial in the consumer. Scatter loop is an exception: it is excluded here,
                        // but may be added in the end.
                        string undecorated_loop_name = extract_after_tokens(loop_name, 2);
                        user_assert(loop_extent.as<IntImm>()) << "Loop " << undecorated_loop_name
                                <<" in func " << envs.find(producer)->second.name()
                                << " is below the buffer loop, and is expected to have a constant extent.";
                        int loop_extent_val = loop_extent.as<IntImm>()->value;
                        WRITES = WRITES * loop_extent_val;
                        WRITE_LOOPS.push_back(i);
                    }
                } else {
                    if (ends_with(loop_name, "." + scatter_loop)) {
                        scatter_loop_removed_in_producer = true;
                    }
                }
                if (for_type != ForType::Unrolled) {
                    // NOTE: we check if a loop is serial based on its current for_type in the consumer.
                    // So READ_LOOPS are those that are currently serial in the consumer.
                    string undecorated_loop_name = extract_after_tokens(loop_name, 2);
                    user_assert(loop_extent.as<IntImm>()) << "Loop " << undecorated_loop_name
                            <<" in func " << func_name
                            << " is below the buffer loop, and is expected to have a constant extent.";
                    int loop_extent_val = loop_extent.as<IntImm>()->value;
                    READS = READS * loop_extent_val;
                    READ_LOOPS.push_back(i);
                }
            } else {
                PERIODS = PERIODS * loop_extent;
            }
            if (ends_with(loop_name, "." + buffer_loop)) {
                inside_buffer_loop = true;
            }
            if (ends_with(loop_name, "." + scatter_loop)) {
                original_scatter_loop = i;
                internal_assert(loop_extent.as<IntImm>());
                BUFFERS = loop_extent.as<IntImm>()->value;
            }
        }
        vector_type = generate_array(TYPE, {Range(0, BUFFERS)});

        if (!scatter_loop_removed_in_producer && scatter_strategy != ScatterStrategy::ForwardVector) {
            WRITES = WRITES * BUFFERS;
            WRITE_LOOPS.push_back(original_scatter_loop);
        }

        debug(4) << "Buffer WRITES: " << WRITES << ", WRITE_LOOPS: ";
        for (auto W : WRITE_LOOPS) {
            debug(4) << std::get<0>(loops[W]) << " ";
        }
        debug(4) << "\nBuffer READS: " << READS << ",  READ_LOOPS: ";
        for (auto R : READ_LOOPS) {
            debug(4) << std::get<0>(loops[R]) << " ";
        }
        debug(4) << "\nPERIODS: " << PERIODS << "\n";

        if (READS < WRITES) {
            user_warning << "Buffering in func " << func_name << ": READS (" << READS << ")" << " < WRITES("
                    << WRITES << "). This buffer might not be able to hide the latency of reading from the main memory\n";
        }

        CYCLES_PER_PERIOD = std::max(READS, WRITES);
        INIT = (READS >= WRITES) ? (READS - WRITES) : 0;

        for (size_t i = 0; i < loops.size(); i++) {
            auto &l = loops[i];
            ForType for_type = std::get<3>(l);
            if (for_type == ForType::Unrolled) {
                string loop_name = std::get<0>(l);
                Expr loop_min = std::get<1>(l);
                Expr loop_extent = std::get<2>(l);
                unroll_loops.push_back(i);
                unroll_loop_vars.push_back(Variable::make(Int(32), loop_name));
                unroll_loop_dims.push_back(Range(loop_min, loop_extent));
                if (!ends_with(loop_name,"." + scatter_loop)) {
                    nonscatter_unroll_loops.push_back(i);
                    nonscatter_unroll_loop_vars.push_back(Variable::make(Int(32), loop_name));
                    nonscatter_unroll_loop_dims.push_back(Range(loop_min, loop_extent));
                }
            }
        }
    }

    // Loop variables referred in an isolated operand are the original loop variables
    // before space time transform. Replace them with those after the transform
    Expr isolated_operand_with_loop_vars_after_stt(const Expr &isolated_opnd) {
        debug(4) << "Isolated operand: " << to_string(isolated_opnd) << "\n";
        const Function &f = envs.at(func_name);
        auto &params = f.definition().schedule().transform_params();
        internal_assert(params.size() <= 1); // Currently only 1 STT is allowed in a function
        Expr opnd = isolated_opnd;
        if (!params.empty()) {
            const auto &reverse = params[0].reverse;
            for (const auto &r : reverse) {
                opnd = substitute(r.first, r.second, opnd);
                debug(4) << "  Replace " << r.first << " with " << r.second << " and get " << opnd << "\n";
            }
        }
        return opnd;
    }

    Range calculate_buffer_range(const Expr index) {
        Expr min = index, max = index;
        vector<int> write_loops(WRITE_LOOPS.begin(), WRITE_LOOPS.end());
        // In ForwardVector, scatter loop is not one of WRITE_LOOPS
        write_loops.push_back(original_scatter_loop);
        for (auto W : write_loops) {
            auto &l = loops[W];
            string loop_name = std::get<0>(l);
            Expr loop_min = std::get<1>(l);
            Expr loop_extent = std::get<2>(l);

            auto var = Variable::make(Int(32), loop_name);
            min = simplify(substitute(var, loop_min, min));
            max = simplify(substitute(var, loop_extent-1, max));
        }
        max = simplify(min + max + 1);
        user_assert(min.as<IntImm>())
            << "Buffer " << func_name << " has non-const lower bound " << min << ", "
            << "Please check if its indices have extra loops\n";
        user_assert(max.as<IntImm>())
            << "Buffer " << func_name << " has non-const upper bound " << max << ", "
            << "Please check if its indices have extra loops\n";
        return Range(min, max);
    }

    int calculate_bank_bits(const vector<Expr> &indices, const buffer_info &buf) {
        map<string, Expr> loop2val;
        for (auto W : WRITE_LOOPS) {
            auto &l = loops[W];
            string name = std::get<0>(l);
            loop2val[extract_after_tokens(name, 2)] = 0;
        }
        string bank_name = std::get<0>(loops[original_scatter_loop]);
        loop2val[extract_after_tokens(bank_name, 2)] = 1;
        bool found_bank_loop = false;
        internal_assert(indices.size() == buf.dims.size()-1);
        // Find the lowest bank bit
        int lowest_bank_bit = 0;
        for (int i = indices.size()-1; i >= 0; i--) {
            Expr extent = simplify(substitute(loop2val, indices[i]));
            if (is_zero(extent)) {
                if (found_bank_loop) continue;
                // The first dimension is used to index buffers
                extent = buf.dims[i+1].extent;
            } else {
                user_assert(!found_bank_loop)
                    << "Failed to partition " << func_name << " into banks with non-uniform addresses";
                found_bank_loop = true;
            }
            auto extent_imm = extent.as<IntImm>();
            internal_assert(extent_imm);
            user_assert(is_power_of_two(extent_imm->value))
                << "Failed to partition " << func_name << " into banks with non-power-of-two addresses";
            lowest_bank_bit += int(log2(extent_imm->value));
        }
        internal_assert(found_bank_loop);
        loop2val[extract_after_tokens(bank_name, 2)] = buf.num_banks;
        found_bank_loop = false;
        // Find the higest bank bit
        int highest_bank_bit = 0;
        for (int i = indices.size()-1; i >= 0; i--) {
            Expr extent = simplify(substitute(loop2val, indices[i]));
            if (is_zero(extent)) {
                if (found_bank_loop) continue;
                extent = buf.dims[i+1].extent;
            } else {
                found_bank_loop = true;
            }
            auto extent_imm = extent.as<IntImm>();
            internal_assert(extent_imm);
            user_assert(is_power_of_two(extent_imm->value))
                << "Failed to partition " << func_name << " into banks with non-power-of-two addresses";
            highest_bank_bit += int(log2(extent_imm->value));
        }
        // If bank bits are consecutive
        user_assert(1 << (highest_bank_bit-lowest_bank_bit) == buf.num_banks)
            << "Failed to partition " << func_name << " into banks with non-uniform addresses";

        return lowest_bank_bit;
    }

    void calculate_parallel_access_arguments(buffer_info &buf) {
        if (buf.bank_bits == 0) {
            // Banks are partitioned along columns (write along columns and read along rows)
            buf.skew_factor = buf.write_args[1] % Expr(BUFFERS);
            Expr row_expr_without_bank_loop = substitute(original_scatter_loop_var, 0, buf.read_args[1]);
            Expr col_expr_aligned_with_bank = (buf.read_args[2] / Expr(BUFFERS)) * Expr(BUFFERS);
            // The write arguments are unchanged
            buf.read_args[1] = simplify(row_expr_without_bank_loop + (original_scatter_loop_var - buf.skew_factor) % Expr(BUFFERS));
            buf.read_args[2] = simplify(col_expr_aligned_with_bank + original_scatter_loop_var);
        } else {
            // Banks are partitioned along rows (write along rows and read along columns)
            buf.skew_factor = buf.write_args[2] % Expr(BUFFERS);
            Expr col_expr_without_bank_loop = substitute(original_scatter_loop_var, 0, buf.read_args[2]);
            Expr row_expr_aligned_with_bank = (buf.read_args[1] / Expr(BUFFERS)) * Expr(BUFFERS);
            // The write arguments are unchanged
            buf.read_args[1] = simplify(row_expr_aligned_with_bank + original_scatter_loop_var);
            buf.read_args[2] = simplify(col_expr_without_bank_loop + (original_scatter_loop_var - buf.skew_factor) % Expr(BUFFERS));
        }
        debug(4) << "The arguments to read a parallel access buffer " << buf.name << "\n"
                 << "\tRow argument: " << buf.read_args[1] << "\n"
                 << "\tCol argument: " << buf.read_args[2] << "\n";
    }

    void calculate_buffer_dims_args(const Expr &isolated_opnd, buffer_info &buf) {
        // Loop vars referred in the isolated operand have no prefix. Thus we search
        // undecorated name in the isolated operand.
        Expr new_isolated_opnd = isolated_operand_with_loop_vars_after_stt(isolated_opnd);
        debug(4) << "calculate_buffer_dims_args with isolated operand " << to_string(new_isolated_opnd) << "\n";

        if (buffer_strategy == BufferStrategy::Double) {
            buf.dims.push_back(Range(0, 2));
            buf.write_args.push_back(_idx);
            buf.read_args.push_back(!_idx);
        }

        std::map<std::string, Expr> names_to_decorated_vars;
        for (size_t i = 0; i < loops.size(); i++) {
            auto loop_name = std::get<0>(loops[i]);
            auto undecorated_name = extract_after_tokens(loop_name, 2);
            names_to_decorated_vars[undecorated_name] = Variable::make(Int(32), loop_name);
        }
        internal_assert(write_indices.size() == read_indices.size());
        for (size_t i = 0; i < write_indices.size(); i++) {
            auto decorated_write = substitute(names_to_decorated_vars, write_indices[i]);
            auto decorated_read = substitute(names_to_decorated_vars, read_indices[i]);
            auto write_range = calculate_buffer_range(decorated_write);
            auto read_range = calculate_buffer_range(decorated_read);
            debug(4) << "Buffer " << func_name << " write index " << i << ": " << decorated_write << "\n";
            debug(4) << " Inferred buffer range: (" << write_range.min << ", " << write_range.extent << ")\n";
            // internal_assert(write_range.min <= read_range.min && write_range.extent >= read_range.extent);
            buf.dims.push_back(write_range);
            buf.write_args.push_back(decorated_write);
            buf.read_args.push_back(decorated_read);
        }
        internal_assert(!scatter_loop_removed_in_producer);
        buf.num_banks = (int32_t)closest_power_of_two((uint32_t)BUFFERS);
        buf.bank_bits = calculate_bank_bits(write_indices, buf);
        buf.parallel_access = false;
        if (write_indices.size() == 2
            && equal(write_indices[0], read_indices[1]) && equal(write_indices[1], read_indices[0])) {
            // Implement a parallel access buffer
            buf.parallel_access = true;
            calculate_parallel_access_arguments(buf);
        }
    }

    void calculate_buffer_dims_args() {
        // TODO: so far, we assume that the buffer is inserted to a Func that was isolated
        // out of another Func as that Func's producer. Should remove this limitation in future.
        const vector<Expr> &isolated_operands = envs.at(func_name).isolated_operands_as_producer();
        buffer_info buf;
        if (TYPE.is_generated_struct()) {
            int type_id = TYPE.bits();
            const std::pair<string, vector<Type>> &entry = GeneratedStructType::structs[type_id];
            internal_assert(entry.second.size() == isolated_operands.size());
            for (size_t i = 0; i < isolated_operands.size(); i++) {
                auto opnd = isolated_operands[i];
                // The following assertion might not hold: After isolation, we might have done
                // vectorization, so the original isolated operand's type might not be current.
                // internal_assert(opnd.type() == entry.second[i]);
                buf.type = entry.second[i];
                buf.name = func_name + ".DB_f" + std::to_string(i) + ".ibuffer";
                calculate_buffer_dims_args(opnd, buf);
                buffers_info.push_back(buf);
            }
        } else {
            internal_assert(isolated_operands.size() == 1);
            buf.type = original_read_node.type();
            buf.name = func_name + ".DB.ibuffer";
        }
        calculate_buffer_dims_args(isolated_operands[0], buf);
        buffers_info.push_back(buf);
    }

    void initialize_common_constants_vars() {
        intialize_common_constants();

        value = Variable::make(TYPE, func_name + "_value.shreg");
        time_stamp = Variable::make(UInt(32), func_name + "_time_stamp.shreg");
        cycle = Variable::make(UInt(32), func_name + "_cycle.temp");
        in_v = Variable::make(TYPE, func_name + "_in_v.temp");
        out_v = Variable::make(TYPE, func_name + "_out_v.temp");
        buf_loop_var = Variable::make(Int(32), func_name + ".s0.buf");
        const string &original_scatter_loop_name = std::get<0>(loops[original_scatter_loop]);
        original_scatter_loop_var = Variable::make(Int(32), original_scatter_loop_name);
        period = Variable::make(UInt(32), "period");
        offset = Variable::make(UInt(32), "offset");
        time_to_write_buffer = Variable::make(Bool(1), "time_to_write_buffer");

        _cycle = Variable::make(UInt(32), "_cycle");
        _period = Variable::make(UInt(32), "_period");
        _offset = Variable::make(UInt(32), "_offset");
        _time_to_write_buffer = Variable::make(Bool(1), "_time_to_write_buffer");
        _owner = Variable::make(UInt(32), "_owner");
        _idx = Variable::make(Bool(1), "_idx");
        _time_to_read = Variable::make(Bool(1), "_time_to_read");

        calculate_buffer_dims_args();
    }

public:
    Stmt visit(const For *op) override{
        if(ends_with(op->name,".run_on_device")){
            return visit_kernel(op);
        }
        Stmt new_body = mutate(op->body);
        if(op->name == std::get<0>(loops.back())) { //innermost loop
            visit_innermost_loop(op, new_body);
        }
        return new_body;
    }

    /* Make IR for the following declarations:
     *   TYPE value[unroll_loop_dims];
         int  time_stamp[unroll_loop_dims];
         TYPE in_v[nonscatter_unroll_loop_dims];
         TYPE out_v[scatter_loop_dims] (for parallel access only)
         int  cycle[nonscatter_unroll_loop_dims];
       Note this code pattern is an extension to the design doc to allow unroll loops
       besides the scatter loop.

       The corresponding IR for the code pattern looks like this:
         realize value.shreg, time_stamp.shreg
           parallel<OpenCL> (loop name, 0, 1) // Dummy loop
             parallel<OpenCL> value.run_on_device // Dummy loop
               parallel<OpenCL> time_stamp.run_on_device // Dummy loop
                 realize cycle.temp, in_v.temp, DB.ibuffer
                   body
       The last two dummy loops are needed for the OpenCL code generator
       to declare value.shreg and time_stamp.shreg
    */
    Stmt visit_kernel(const For *op) {
        internal_assert(ends_with(op->name,".run_on_device"));
        Stmt new_body = IRMutator::mutate(op->body);
        for (auto b : buffers_info) {
            Type bank_type = Int(b.bank_bits, b.num_banks);
            new_body = Realize::make(b.name, {bank_type, b.type}, MemoryType::Auto, b.dims, const_true(), new_body);
        }
        Type value_t = (scatter_strategy == ScatterStrategy::ForwardVector) ? vector_type : TYPE;
        new_body = Realize::make(var_name(cycle), {UInt(32)}, MemoryType::Register, nonscatter_unroll_loop_dims, const_true(), new_body);
        new_body = Realize::make(var_name(in_v), {value_t}, MemoryType::Register, nonscatter_unroll_loop_dims, const_true(), new_body);
        if (buffers_info[0].parallel_access) {
            new_body = Realize::make(var_name(out_v), {vector_type}, MemoryType::Register, {} , const_true(), new_body);
        }
        // new_body = For::make(replace_postfix(var_name(time_stamp), ".shreg", ".run_on_device"), 0, 1, ForType::Parallel, DeviceAPI::OpenCL, new_body);
        new_body = For::make(replace_postfix(var_name(time_stamp), ".shreg", ".run_on_device"), 0, 1, ForType::Parallel, op->device_api, new_body); 
        // new_body = For::make(replace_postfix(var_name(value), ".shreg", ".run_on_device"), 0, 1, ForType::Parallel, DeviceAPI::OpenCL, new_body);  
        new_body = For::make(replace_postfix(var_name(value), ".shreg", ".run_on_device"), 0, 1, ForType::Parallel, op->device_api, new_body);
        new_body = For::make(op->name, op->min, op->extent, op->for_type, op->device_api, new_body);
        new_body = Realize::make(var_name(time_stamp), {UInt(32)}, MemoryType::Register, unroll_loop_dims, const_true(), new_body);
        new_body = Realize::make(var_name(value), {value_t}, MemoryType::Register, unroll_loop_dims, const_true(), new_body);
        return new_body;
    }

    /* Generate IR like this:
     *   initialize cycle
     *   while (1)
     *      all unroll loops except the scatter loop // NEW to the design doc
     *          get input // All variables in this step needs an instance for
     *                    // each iteration of the above unroll loops
     *          unroll for buf = 0 : BUFFERS
     *                   original_scatter_loop = buf + scatter loop's min
     *                   broadcast input by scattering
     *                   write buffer
     *                   read buffer
     *          cycle++
     * A difference from the document is that we allow other unroll loops to exist
     * besides the scatter loop in the consumer. So all the variables used between
     * the above "all unrolled loops except the scatter loop" and "unroll for buf" loop,
     * i.e. "get input", need an instance for every iteration of the other unroll loops,
     * and need indices of these unroll loops. Such variables include cycle and in_v
     * (The other 3 variables: period, offset, and time_to_write_buffer can be made as
     * the target variables of LetStmts and thus avoid using the indices explicitly).
     */
    void visit_innermost_loop(const For *op, Stmt &new_body) {
        internal_assert(!ends_with(op->name,".run_on_device")); // not kernel loop
        internal_assert(op->name == std::get<0>(loops.back())); // is innermost loop

        // Build new loop body bottom up
        read_from_buffer(new_body);
        write_to_buffer(new_body);
        broadcast_input_by_scattering(new_body);

        // Add scatter loop
        const Expr &original_scatter_loop_min = std::get<1>(loops[original_scatter_loop]);
        const Expr &original_scatter_loop_extent = std::get<2>(loops[original_scatter_loop]);
        new_body = LetStmt::make(var_name(original_scatter_loop_var), buf_loop_var + original_scatter_loop_min, new_body);
        new_body = For::make(var_name(buf_loop_var), 0, original_scatter_loop_extent, ForType::Unrolled,op->device_api,new_body);

        // Read input value before the scatterring
        get_input(op, new_body);
        if (buffers_info[0].parallel_access) {
            rotate_out_v_for_parallel_access(op, new_body);
        }

        Expr single_PE_cond;
        vector<string> unrolled_loops_without_terms;
        vector<string> unrolled_loops_name;
        for (auto &v : nonscatter_unroll_loop_vars) {
            unrolled_loops_name.push_back(v.as<Variable>()->name);
        }
        Stmt inc_cycle = Provide::make(var_name(cycle), {Call::make(UInt(32), var_name(cycle), nonscatter_unroll_loop_vars, Call::PureIntrinsic) + 1}, nonscatter_unroll_loop_vars);
        if (check_is_single_PE(true, original_read_condition, unrolled_loops_name, {}, single_PE_cond, unrolled_loops_without_terms)) {
            inc_cycle = IfThenElse::make(single_PE_cond, inc_cycle);
        }
        new_body = Block::make(new_body, inc_cycle);

        add_nonscatter_unroll_loops(op->device_api, new_body);

        // TODO: change here into while(1)
        new_body = For::make(func_name + ".s0.outermost_loop.infinite", 0, 10, ForType::Serial, op->device_api, new_body);  
        // new_body = For::make(func_name + ".s0.outermost_loop", 0, Expr(PERIODS + 1) * Expr(CYCLES_PER_PERIOD), ForType::Serial, op->device_api, new_body);

        initialize(op->device_api, new_body);
    }

    /* Make IR like this:
     *    unroll for nonscatter_unroll_loop_vars
     *                cycle[nonscatter_unroll_loop_vars] = INIT;
     */
    void initialize(const DeviceAPI device_api, Stmt &new_body) {
        vector<Expr> init_args;
        for (auto v : nonscatter_unroll_loop_vars) {
            init_args.push_back(Variable::make(Int(32), var_name(v) + "_init"));
        }

        Expr single_PE_cond, original_cond = original_read_condition;
        vector<string> unrolled_loops_without_terms;
        vector<string> unrolled_loops_name;
        for (auto &v : nonscatter_unroll_loop_vars) {
            string name = v.as<Variable>()->name;
            unrolled_loops_name.push_back(name + "_init");
            original_cond = substitute(name, Variable::make(Int(32), name + "_init"), original_cond);
        }
        Stmt init_cycle = Provide::make(var_name(cycle), {Expr(INIT)}, init_args);
        if (check_is_single_PE(true, original_cond, unrolled_loops_name, {}, single_PE_cond, unrolled_loops_without_terms)) {
            init_cycle = IfThenElse::make(single_PE_cond, init_cycle);
        }

        for (int i = nonscatter_unroll_loops.size() - 1; i >= 0; i--) {
            auto &l = loops[nonscatter_unroll_loops[i]];
            string name = std::get<0>(l);
            int min = std::get<1>(l).as<IntImm>()->value;
            int extent = std::get<2>(l).as<IntImm>()->value;
            init_cycle = For::make(name + "_init", min, extent, ForType::Unrolled, device_api, init_cycle);
        }
        new_body = Block::make(init_cycle, new_body);
    }

    // Wrap the new body with all unroll loops except the scatter loop
    void add_nonscatter_unroll_loops(const DeviceAPI device_api, Stmt &new_body) {
        for (int i = nonscatter_unroll_loops.size() - 1; i >= 0; i--) {
            auto &l = loops[nonscatter_unroll_loops[i]];
            string loop_name = std::get<0>(l);
            Expr loop_min = std::get<1>(l);
            Expr loop_extent = std::get<2>(l);
            new_body = For::make(loop_name, loop_min, loop_extent, ForType::Unrolled, device_api,new_body);
        }
    }

    void rotate_out_v_for_parallel_access(const For *op, Stmt &new_body) {
        const auto &buf = buffers_info[0];
        string new_loop_name = var_name(buf_loop_var) + ".t";
        Expr new_loop_var = Variable::make(Int(32), new_loop_name);
        Expr skew_expr = simplify((original_scatter_loop_var + buf.skew_factor) % Expr(BUFFERS));
        Expr new_read_node = Call::make(TYPE, Call::read_array, { var_name(out_v), skew_expr }, Call::PureIntrinsic);
        Expr new_write_node = substitute(original_read_node, new_read_node, original_write_node);

        Stmt loop_body = Evaluate::make(new_write_node);
        loop_body = IfThenElse::make(original_read_condition, loop_body);
        loop_body = IfThenElse::make(_time_to_read, loop_body);
        // Recover the variables of the sequential loops
        Expr reads = _offset; // Total reads in the current period
        for(int i = READ_LOOPS.size() - 1; i >= 0; --i){
            auto &l = loops[READ_LOOPS[i]];
            string loop_name = std::get<0>(l);
            Expr loop_min = std::get<1>(l);
            Expr loop_extent = std::get<2>(l);
            Expr var;
            if (i == 0) {
                // Optimization: reads must already be within loop_extent. No need to %.
                var = reads + loop_min;
            } else {
                var = (reads % loop_extent) + loop_min;
            }
            loop_body = LetStmt::make(loop_name, var, loop_body);
            reads = reads / loop_extent;
        }
        // Expr periods_unfinished = (_period <= PERIODS);
        Expr val = (READS >= WRITES) ? (_period > 0) :
                   ((_period > 0) && (_offset < Expr(READS)));
        loop_body = LetStmt::make("_time_to_read", val, loop_body);

        Expr cur_cycle = Call::make(UInt(32), var_name(cycle), nonscatter_unroll_loop_vars, Call::PureIntrinsic);
        loop_body = LetStmt::make(var_name(_period), cur_cycle / Expr(CYCLES_PER_PERIOD), loop_body);
        loop_body = LetStmt::make(var_name(_offset), cur_cycle % Expr(CYCLES_PER_PERIOD), loop_body);
        loop_body = substitute(original_scatter_loop_var, new_loop_var, loop_body);

        Stmt for_loop = For::make(new_loop_name, op->min, op->extent, ForType::Unrolled, op->device_api, loop_body);
        new_body = Block::make(new_body, for_loop);
    }

    /* Make IR as:
     *   int period = cycle[nonscatter_unroll_loop_vars] / CYCLES_PER_PERIOD; // current period
         int offset = cycle[nonscatter_unroll_loop_vars] % CYCLES_PER_PERIOD; // relative position of the current cycle in the current period
         bool time_to_write_buffer = (offset >= INIT);
         if ((period < PERIODS) && time_to_write_buffer) {
             in_v[nonscatter_unroll_loop_vars] = read_channel_intel(IN_CHANNEL);
         }
     */
    void get_input(const For *op, Stmt &new_body) {
        // The original scatter loop is unroll loop in the consumer, and thus every
        // iteration of this loop reads from the producer with a seperate channel.
        // With scattering, however, we let only the first iteration reads from
        // the producer.
        const string &original_scatter_loop_name = std::get<0>(loops[original_scatter_loop]);
        Expr scatter_loop_min = std::get<1>(loops[original_scatter_loop]);
        Stmt read_input;
        if (scatter_strategy == ScatterStrategy::ForwardVector) {
            string scatter_loop_name = std::get<0>(loops[original_scatter_loop]);
            Expr scatter_loop_min = std::get<1>(loops[original_scatter_loop]);
            Expr scatter_loop_extent = std::get<2>(loops[original_scatter_loop]);
            Expr read_node = Call::make(TYPE, Call::write_array, { var_name(in_v), original_read_node, original_scatter_loop_var }, Call::PureIntrinsic);
            read_input = Evaluate::make(read_node);
            read_input = For::make(scatter_loop_name, scatter_loop_min, scatter_loop_extent, ForType::Unrolled, op->device_api, read_input);
        } else {
            Expr read_node = substitute(original_scatter_loop_name, scatter_loop_min, original_read_node);
            read_input = Provide::make(var_name(in_v), {read_node}, nonscatter_unroll_loop_vars);
        }

        if (!equal(original_read_condition, const_true())) {
            // This piece of "get_input" IR will be wrapped around by all the unrolled loops, except
            // the scatter loop, which has been serialized in the producer, and thus in the consumer
            // side, we can only know the scatter loop is in which iteration by decoding the cycle.
            // This is necessary when the scatter loop var is used in the original condition.
            const string &original_scatter_loop_name = std::get<0>(loops[original_scatter_loop]);
            Expr scatter_loop_tmp = Variable::make(Int(32), original_scatter_loop_name + "_tmp");
            Expr new_condition = substitute(original_scatter_loop_name, scatter_loop_tmp, original_read_condition);
            read_input = IfThenElse::make(new_condition, read_input);
            Expr total_writes_so_far = Variable::make(UInt(32), "total_writes_so_far");
            Expr scatter_loop_val = (total_writes_so_far  % Expr((uint32_t)BUFFERS)) + scatter_loop_min;
            read_input = LetStmt::make(var_name(scatter_loop_tmp), scatter_loop_val, read_input);
            read_input = LetStmt::make(var_name(total_writes_so_far), period * Expr(WRITES) + offset - Expr(INIT), read_input);
        }

        Expr condition = time_to_write_buffer;
        read_input = IfThenElse::make(condition, read_input);

        read_input = LetStmt::make(var_name(time_to_write_buffer), (offset >= Expr(INIT)), read_input);

        Expr offset_val = Call::make(UInt(32), var_name(cycle), nonscatter_unroll_loop_vars, Call::PureIntrinsic) % Expr(CYCLES_PER_PERIOD);
        read_input = LetStmt::make(var_name(offset), offset_val, read_input);

        Expr period_val = Call::make(UInt(32), var_name(cycle), nonscatter_unroll_loop_vars, Call::PureIntrinsic) / Expr(CYCLES_PER_PERIOD);
        read_input = LetStmt::make(var_name(period), period_val, read_input);

        new_body = Block::make(read_input, new_body);
    }

    /**
     * if(buf == 0) {
     *     value     [unroll_loop_vars] = in_v [nonscatter_unroll_loop_vars];
     *     time_stamp[unroll_loop_vars] = cycle[nonscatter_unroll_loop_vars];
     * } else {
     *     value     [unroll_loop_vars] = value     [unroll_loop_vars with buf - 1];
     *     time_stamp[unroll_loop_vars] = time_stamp[unroll_loop_vars with buf - 1];
     * }
     */
    void broadcast_input_by_scattering(Stmt &new_body) {
        bool strategy_up = (scatter_strategy != ScatterStrategy::Down);
        Type type = (scatter_strategy == ScatterStrategy::ForwardVector) ? vector_type : TYPE;

        // If branch
        vector<Expr> write_value0_args(unroll_loop_vars);
        write_value0_args.insert(write_value0_args.begin(), var_name(value));
        write_value0_args.push_back(Call::make(type, var_name(in_v), nonscatter_unroll_loop_vars, Call::PureIntrinsic));
        Stmt write_value0 = Evaluate::make(Call::make(type, Call::write_shift_reg, write_value0_args, Call::Intrinsic));

        vector<Expr> write_time_stamp0_args(unroll_loop_vars);
        write_time_stamp0_args.insert(write_time_stamp0_args.begin(), var_name(time_stamp));
        write_time_stamp0_args.push_back( Call::make(UInt(32), var_name(cycle), nonscatter_unroll_loop_vars, Call::PureIntrinsic));
        Stmt write_time_stamp0 = Evaluate::make(Call::make(UInt(32), Call::write_shift_reg, write_time_stamp0_args, Call::Intrinsic));

        Stmt base = Block::make(write_value0, write_time_stamp0);

        // Else branch
        Expr prev_buf =  strategy_up ? Sub::make(buf_loop_var, 1) : Add::make(buf_loop_var, 1);
        const string &original_scatter_loop_name = std::get<0>(loops[original_scatter_loop]);
        vector<Expr> prev_args(unroll_loop_vars);
        for (auto &a : prev_args) {
            if (var_name(a) == original_scatter_loop_name) {
                a = prev_buf;
                break;
            }
        }
        vector<Expr> read_value_args(prev_args);
        read_value_args.insert(read_value_args.begin(), var_name(value));
        Expr read_value = Call::make(type, Call::read_shift_reg, read_value_args, Call::PureIntrinsic);

        vector<Expr> shift_value_args(unroll_loop_vars);
        shift_value_args.insert(shift_value_args.begin(), var_name(value));
        shift_value_args.push_back(read_value);
        Stmt shift_value =  Evaluate::make(Call::make(type, Call::write_shift_reg, shift_value_args, Call::Intrinsic));

        vector<Expr> read_time_stamp_args(prev_args);
        read_time_stamp_args.insert(read_time_stamp_args.begin(), var_name(time_stamp));
        Expr read_time_stamp = Call::make(UInt(32), Call::read_shift_reg, read_time_stamp_args, Call::PureIntrinsic);

        vector<Expr> shift_time_stamp_args(unroll_loop_vars);
        shift_time_stamp_args.insert(shift_time_stamp_args.begin(), var_name(time_stamp));
        shift_time_stamp_args.push_back(read_time_stamp);
        Stmt shift_time_stamp =  Evaluate::make(Call::make(UInt(32), Call::write_shift_reg, shift_time_stamp_args, Call::Intrinsic));

        Stmt recursion = Block::make(shift_value, shift_time_stamp);

        // Condition
        Expr condition = strategy_up ? (buf_loop_var == 0) : (buf_loop_var == Expr(BUFFERS - 1));

        // Compose together
        Stmt broadcast_incoming_value = IfThenElse::make(condition, base, recursion);
        if (!equal(original_read_condition, const_true())) {
            broadcast_incoming_value = IfThenElse::make(original_read_condition, broadcast_incoming_value);
        }
        new_body = Block::make(broadcast_incoming_value, new_body);
    }

    /* Make IR as:
        int  _cycle = time_stamp[unroll_loop_vars];
        int  _period = _cycle / CYCLES_PER_PERIODS;
        int  _offset = _cycle % CYCLES_PER_PERIODS;
        bool _time_to_write_buffer = (_offset >= INIT);
        int  _owner = _cycle % BUFFERS;
        bool _idx = _period & 1;
        if (buf == _owner) // Note: needed only when the scatter loop is not removed in the producer
          if (_time_to_write_buffer)
            TYPE _tmp =  = value[unroll_loop_vars];
            DB_f0[_idx][WRITE_TO(_offset)][buf] = _tmp.f0;
            DB_f1[_idx][WRITE_TO(_offset)][buf] = _tmp.f1;
            ...
            // Or simply: DB[_idx][WRITE_TO(_offset)][buf] = value[unroll_loop_vars]; if there is only 1 type of buffer.
    */
    void write_to_buffer(Stmt &new_body) {
        Stmt write_buffer;
        vector<Expr> read_value_args(unroll_loop_vars);
        read_value_args.insert(read_value_args.begin(), var_name(value));
        Expr read_value;
        if (scatter_strategy != ScatterStrategy::ForwardVector) {
            read_value = Call::make(TYPE, Call::read_shift_reg, read_value_args, Call::PureIntrinsic);
        } else {
            // Read the vector first, then read the current element
            Expr read_vector = Call::make(vector_type, Call::read_shift_reg, read_value_args, Call::PureIntrinsic);
            Expr ele_index =  original_scatter_loop_var;
            if (buffers_info[0].parallel_access) {
                ele_index = simplify((original_scatter_loop_var - buffers_info[0].skew_factor) % Expr(BUFFERS));
            }
            read_value = Call::make(TYPE, Call::read_array, { read_vector, ele_index }, Call::PureIntrinsic);
        }

        int num_types_of_buffers = buffers_info.size();
        internal_assert(num_types_of_buffers >= 1);
        if (num_types_of_buffers == 1) {
            // DB[_idx][WRITE_TO(_offset)][buf] = value[unroll_loop_vars]
            write_buffer = Provide::make(buffers_info[0].name, {read_value}, buffers_info[0].write_args);
        } else {
            // TYPE _tmp = value[unroll_loop_vars];
            // DB_f0[_idx][WRITE_TO(_offset)][buf] = _tmp.f0;
            // DB_f1[_idx][WRITE_TO(_offset)][buf] = _tmp.f1;
            // ...
            Expr _tmp = Variable::make(TYPE, "_tmp");
            vector<Stmt> writes(buffers_info.size());
            for (size_t i = 0; i < buffers_info.size(); i++) {
                Expr field = Call::make(buffers_info[i].type, Call::read_field, {_tmp, IntImm::make(Int(32), i)}, Call::PureIntrinsic);
                writes[i] = Provide::make(buffers_info[i].name, {field}, buffers_info[i].write_args);
            }
            write_buffer = Block::make(writes);
            write_buffer = LetStmt::make("_tmp", read_value, write_buffer);
        }
        write_buffer = IfThenElse::make(original_read_condition, write_buffer);

        // Calculate the write loop vars for write buffer.
        // These loops will be serial in the producer. Some of them might be unrolled
        // originally, and such a loop is added separately in two places:
        // (1) as the new innermost loop if it is the scatter loop, (2) as the loops
        // right below the outermost loop (see add_nonscatter_unroll_loops).
        Expr writes = (_offset - Expr(INIT)) % Expr(WRITES); // Total writes in the current period
        for(int i = WRITE_LOOPS.size() - 1; i >= 0; i--){
            int loop_index = WRITE_LOOPS[i];
            auto &l = loops[loop_index];
            Expr loop_extent = std::get<2>(l);
            if(loop_index == original_scatter_loop && !scatter_loop_removed_in_producer){
                writes = writes / loop_extent;
                continue;
            }
            if (std::get<3>(l) == ForType::Unrolled) {
                continue;
            }
            string loop_name = std::get<0>(l);
            Expr loop_min = std::get<1>(l);
            Expr var;
            if (i == 0) {
                // Optimization: writes must already be within loop_extent. No need to %.
                var = writes + loop_min;
            } else {
                var = (writes % loop_extent) + loop_min;
            }
            writes = writes / loop_extent;
            write_buffer = LetStmt::make(loop_name, var, write_buffer);
        }

        Expr _owner_value;
        // If scatter loop is not removed in producer, BUFFERS must be a factor of WRITES(i.e. _period * WRITES % BUFFERS == 0)
        // We can optimize the calculation of owner.  
        if (!scatter_loop_removed_in_producer && scatter_strategy != ScatterStrategy::ForwardVector) {
            write_buffer = IfThenElse::make(buf_loop_var == _owner, write_buffer);
            _owner_value = (_offset - Expr((uint32_t)INIT)) % Expr((uint32_t)BUFFERS);
        } else {
            _owner_value = (_period * Expr((uint32_t)WRITES) + _offset - Expr((uint32_t)INIT)) % Expr((uint32_t)BUFFERS);
        }
        write_buffer = IfThenElse::make(_time_to_write_buffer, write_buffer);
        new_body = Block::make(write_buffer, new_body);
        new_body = LetStmt::make(var_name(_idx), Cast::make(Bool(), _period & 0x1), new_body);
        new_body = LetStmt::make(var_name(_owner), _owner_value, new_body);
        new_body = LetStmt::make(var_name(_time_to_write_buffer), _offset >= Expr(INIT), new_body);
        new_body = LetStmt::make(var_name(_offset), _cycle % Expr(CYCLES_PER_PERIOD), new_body);
        new_body = LetStmt::make(var_name(_period), _cycle / Expr(CYCLES_PER_PERIOD), new_body);
        vector<Expr> read_time_stamp_args(unroll_loop_vars);
        read_time_stamp_args.insert(read_time_stamp_args.begin(), var_name(time_stamp));
        new_body = LetStmt::make(var_name(_cycle), Call::make(UInt(32), Call::read_shift_reg, read_time_stamp_args, Call::PureIntrinsic), new_body);
    }

    /* Make IR as:
     *       bool _time_to_read = (READS >= WRITES) ? (_period > 0 && _period <= PERIODS) :
     *                                                (_period > 0 && _period <= PERIODS && _offset < READS)
             if (_time_to_read) {
                write_channel_intel(OUT_CHANNEL[buf], _tmp) where
                    _tmp = { DB_f0[!_idx][READ_FROM(_offset)][buf], DB_f1[!_idx][READ_FROM(_offset)][buf], ...}
                    or simply _tmp = DB[!_idx][READ_FROM(_offset)][buf] if there is only 1 type of buffer.
             }
     */
    void read_from_buffer(Stmt &new_body) {
        Expr buf_value;
        int num_types_of_buffers = buffers_info.size();
        internal_assert(num_types_of_buffers >= 1);
        if (num_types_of_buffers == 1) {
            // DB[!_idx][READ_FROM(_offset)][buf]
            buf_value = Call::make(TYPE, buffers_info[0].name, buffers_info[0].read_args, Call::PureIntrinsic);
        } else {
            // { DB_f0[!_idx][READ_FROM(_offset)][buf], DB_f1[!_idx][READ_FROM(_offset)][buf], ...}
            vector<Expr> fields;
            for (size_t i = 0; i < buffers_info.size(); i++) {
                Expr field = Call::make(buffers_info[i].type, buffers_info[i].name, buffers_info[i].read_args, Call::PureIntrinsic);
                fields.push_back(field);
            }
            buf_value = Call::make(TYPE, Call::make_struct, fields, Call::PureIntrinsic);
        }
        if (!buffers_info[0].parallel_access) {
            new_body = substitute(original_read_node, buf_value, new_body);
        } else {
            // Replace the write_channel node with a write_array
            buf_value = Call::make(TYPE, Call::write_array, { var_name(out_v), buf_value, original_scatter_loop_var }, Call::PureIntrinsic);
            new_body = substitute(original_write_node, buf_value, new_body);
        }

        // Recover the variables of the sequential loops
        Expr reads = _offset; // Total reads in the current period
        for(int i = READ_LOOPS.size() - 1; i >= 0; --i){
            auto &l = loops[READ_LOOPS[i]];
            string loop_name = std::get<0>(l);
            Expr loop_min = std::get<1>(l);
            Expr loop_extent = std::get<2>(l);
            Expr var;
            if (i == 0) {
                // Optimization: reads must already be within loop_extent. No need to %.
                var = reads + loop_min;
            } else {
                var = (reads % loop_extent) + loop_min;
            }
            new_body = LetStmt::make(loop_name, var, new_body);
            reads = reads / loop_extent;
        }

        new_body = IfThenElse::make(_time_to_read, new_body);

        // Expr periods_unfinished = (_period <= PERIODS);
        Expr val = (READS >= WRITES) ? (_period > 0) :
                   ((_period > 0) && (_offset < Expr(READS)));
        new_body = LetStmt::make("_time_to_read", val, new_body);
    }
};

class AddressableBufferInserter : public IRMutator {
    using IRMutator::visit;
private:
    const map<string, Function>& envs;
    const AddressableBufferArgs &args;
    const vector<tuple<string, Expr, Expr, ForType>> &all_loops;
public:
    AddressableBufferInserter(const map<string, Function>& _envs,
                          const AddressableBufferArgs &_args,
                          const vector<tuple<string, Expr, Expr, ForType>> &_all_loops):
        envs(_envs), args(_args), all_loops(_all_loops) {}

    Stmt visit(const ProducerConsumer *op) override {
        if (op->is_producer && args.find(op->name) != args.end()) {
            auto iter = args.find(op->name);
            Stmt new_body;
            AddressableBuffer addressable_buffer(envs, all_loops, op->name,
                                        iter->second.producer,
                                        iter->second.buffer_loop,
                                        iter->second.scatter_loop,
                                        iter->second.write_indices,
                                        iter->second.read_indices,
                                        iter->second.read_node,
                                        iter->second.write_node,
                                        iter->second.read_condition,
                                        iter->second.buffer_strategy,
                                        iter->second.scatter_strategy,
                                        iter->second.loops);
            new_body =  addressable_buffer.mutate(op->body);
            return ProducerConsumer::make(op->name,true,new_body);
        }
        return IRMutator::visit(op);
    }
};

// Send one more period of dummy data from a producer to its consumer to buffer
class OneMorePeriod : public IRMutator {
    using IRMutator::visit;
private:
    bool in_producer;       // In the definition of a func that produces data for its consumer to buffer
    string producer_name;   // The name of the func that produces data for its consumer to buffer
    string buffer_loop;     // The loop under which a buffer is inserted in the consumer side
    string innermost_loop;  // The innermost loop name
    vector<string> loops;   // All the loops
    vector<Expr> mins;      // The original mins of the loops
    vector<Expr> extents;   // The original extents of the loops

    const AddressableBufferArgs     &args;
    const map<string, Function> &env;

public:
    Stmt visit(const ProducerConsumer *op) override {
        bool old_in_producer = in_producer;
        if (op->is_producer) {
            for (auto a : args) {
                if (a.second.producer == op->name) {
                    in_producer = true;
                    producer_name = op->name;
                    buffer_loop = a.second.buffer_loop;
                }
            }
        }
        if (in_producer) {
            Stmt s = ProducerConsumer::make(op->name, op->is_producer, mutate(op->body));
            in_producer = old_in_producer;
            return s;
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const For *op) override {
        if (!in_producer) {
            return IRMutator::visit(op);
        }

        // Skip the dummy loops that only tell the compiler this is a device function
        bool dummy_loop = ends_with(op->name, ".run_on_device");
        if (!dummy_loop) {
            loops.push_back(op->name);
            mins.push_back(op->min);
            extents.push_back(op->extent);
        }

        Stmt body = mutate(op->body);
        Expr min = op->min, extent = op->extent;
        if (!dummy_loop) {
            if (innermost_loop == op->name) {
                // if (first loop < its extent || (second until buffer loop all equal 0) body
                Expr cond1 = Variable::make(Int(32), loops[0]) < extents[0];
                Expr cond2 = const_true();
                if (extract_after_tokens(loops[0], 2) != buffer_loop) {
                    for (size_t i = 1; i < loops.size(); i++) {
                        Expr loop_var = Variable::make(Int(32), loops[i]);
                        cond2 = cond2 && (loop_var == mins[i]);
                        if (extract_after_tokens(loops[i], 2) == buffer_loop) {
                            break;
                        }
                    }
                }
                body = IfThenElse::make(cond1 || cond2, body);
            }
            if (loops.size() == 1) {
                // The outermost loop. Increase its extent by 1
                extent += 1;
            } else if (!is_const(min) && min.as<Variable>() && min.as<Variable>()->name == loops[0]) {
                Expr original_outer_extent = extents[0];
                Expr outer_loop_var = Variable::make(Int(32), loops[0]);
                extent = extent + outer_loop_var / original_outer_extent;
            }
            loops.pop_back();
            mins.pop_back();
            extents.pop_back();
        }
        return For::make(op->name, min, extent, op->for_type, op->device_api, body);
    }

    Expr visit(const Call *op) override {
        // TODO: here we should catch calls with side effects, or memory reads, and change them into
        // something like select(cond, op, dummy) instead. No need to change write_channel.
        // Also put the cond at the beginning of the innermost loop.
        if (op->is_intrinsic(Call::write_channel) && extract_first_token(op->args[0].as<StringImm>()->value) == producer_name) {
            // Change the value to write to the channel as
            //      select(first loop < its extent, current value, a dummy value);
            innermost_loop = loops.back();
            Expr value = op->args[1];
            Expr condition = Variable::make(Int(32), loops[0]) < extents[0];
            Expr dummy = cast(value.type(), Expr(0));
            vector<Expr> args= { op->args[0], Select::make(condition, value, dummy) };
            std::copy(op->args.begin()+2, op->args.end(), std::back_inserter(args)); 
            return Call::make(op->type, op->name, args, op->call_type);
        } else {
            return IRMutator::visit(op);
        }
    }

public:
    OneMorePeriod(AddressableBufferArgs& _args, const map<string, Function>& _env):
        args(_args), env(_env){ in_producer = false; }
};

class ModifyScatterLoop : public IRMutator{
    using IRMutator::visit;
    const vector<string> &all_loops_to_serialize;
    map<string, Expr> loop2min; // Loop to serialize--> its min
public:
    ModifyScatterLoop(const vector<string> &all_loops_to_serialize) :
        all_loops_to_serialize(all_loops_to_serialize) {}

    Stmt visit(const For *op) override{
        if (std::find(all_loops_to_serialize.begin(), all_loops_to_serialize.end(), op->name) != all_loops_to_serialize.end()) {
            loop2min[op->name] = op->min;
            return For::make(op->name, op->min, op->extent, ForType::Serial, op->device_api, mutate(op->body));
        }
        return For::make(op->name, op->min, op->extent, op->for_type, op->device_api, mutate(op->body));
    }

    Expr visit(const Call *op) override{
        if(op->is_intrinsic(Call::write_channel) || op->is_intrinsic(Call::read_channel)) {
            vector<Expr> new_args;
            new_args.push_back(op->args[0]); // channel name
            size_t offset = 1;
            if (op->is_intrinsic(Call::write_channel)) {
                new_args.push_back(mutate(op->args[1])); // data
                offset = 2;
            }
            for (size_t i = offset; i < op->args.size(); i++) {
                Expr arg = op->args[i];
                if (arg.as<Variable>()) {
                    const string &loop = arg.as<Variable>()->name;
                    if (loop2min.find(loop) != loop2min.end()) {
                        Expr min = loop2min.at(loop);
                        arg = substitute(loop, min, arg);
                    }
                }
                new_args.push_back(arg);
            }
            return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index, op->image, op->param);
        }
        return IRMutator::visit(op);
    }
};

// Suppose func p is isolated out of func c, and both funcs are on device. Both funcs have the same
// loop structure, because of isolation. Suppose there is an unrolled loop l with the extent of L,
// and here we want to scatter along dimension l in func c. Without scattering, func p and c
// both have L PEs; func p is writing into L channels, and func c reads from these L channels.
// After scattering, all the PEs of func p are collapsed into a single PE; this single PE of func p
// writes into 1 channel, and the first PE of func c reads that channel and distributes the data
// to the other PEs.
// If another func p1 is isolated out of func p, dimension l in func p1 should also be "collapsed",
// since our principle of isolation is that producer and consumer have the same loop structure. In
// general, we need collapse dimension l for the producer chain isolated out of func c.
// Note: below we do not check if a Func is on device or not. So loops on host can be serializd as well.
void find_loops_to_serialize_by_scattering(const Function &func, const string &scatter_loop,
                                           const map<string, Function> &env,
                                           vector<string> &all_loops_to_serialize) {
    // Find the scatter loop in the producer chain of the func
    for (auto &e : env) {
        const Function &p = e.second;
        if (/*p.place() == Place::Device &&*/ p.isolated_from_as_producer() == func.name()) {
            // p is on the device and isolated from func. Make the scatter_loop as serial in p.
            all_loops_to_serialize.push_back(p.name() + ".s0." + scatter_loop);
            find_loops_to_serialize_by_scattering(p, scatter_loop, env, all_loops_to_serialize);
            return;
        }
    }
}

}

void find_loops_to_serialize(const std::map<std::string, Function> &env,
                             std::vector<std::string> &all_loops_to_serialize) {
    for (auto e : env) {
        auto &func = e.second;
        const std::vector<ScatterItem>& scatter_params = func.definition().schedule().scatter_params();
        internal_assert(scatter_params.size() < 2); // Currently a func can only have zero or one time of scatter
        if (scatter_params.size() > 0) {
            string scatter_loop = scatter_params[0].loop_name;
            ScatterStrategy ss = scatter_params[0].strategy;
            // ForwardVector does not collapse the producer's PEs into a single one, because it
            // requires each PE to send out one element of the vector at the same time.
            if (ss != ScatterStrategy::ForwardVector) {
                // Find the scatter loop in the entire producer chain of func
                find_loops_to_serialize_by_scattering(func, scatter_loop, env, all_loops_to_serialize);
            }
        }
    }
}

Stmt insert_addressable_buffer(Stmt s, const std::map<std::string, Function> &env) {
    AddressableBufferArgs args;
    get_AddressableBufferArgs(env, args);
    if (args.empty()) {
        return s;
    }

    // Check for assumptions, as well as getting the info of all the loops
    // in the entire IR (Full names, mins, extents and types), read nodes and
    // path conditions to the read nodes.
    vector<tuple<string, Expr, Expr, ForType>> all_loops;
    AddressableBufferChecker ibc(args, env, all_loops);
    s.accept(&ibc);

    // Implement buffering
    AddressableBufferInserter ibi(env, args, all_loops);
    s = ibi.mutate(s);

    // For double buffering, a producer needs send one more period of trash data to the consumer
    OneMorePeriod omp(args, env);
    s = omp.mutate(s);

    // Finalize scatter loops.
    // First, find all the loops to be serialized in the entire IR.
    vector<string> all_loops_to_serialize;
    find_loops_to_serialize(env, all_loops_to_serialize);
    ModifyScatterLoop msl(all_loops_to_serialize);
    s = msl.mutate(s);

    return s;
}

} // namespace Internal
} // namespace Halide