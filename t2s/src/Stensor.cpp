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
#include "../../Halide/src/Simplify.h"
#include "../../Halide/src/Substitute.h"
#include "./DebugPrint.h"
#include "./Utilities.h"
#include "./PreprocessBeforeLower.h"
#include "./Stensor.h"

namespace Halide {

using namespace Internal;
using std::vector;
using std::map;
using std::string;

struct Schain {
    bool is_output;                 // Output chain needs different primitives
    Func outf;                      // The output chain starts from a function
    Stensor fork_from;              // The input chain starts from another stensor
    vector<ImageParam> imp;         // The input chain starts from external inputs
    vector<Stensor> stensors;
    vector<Func> funcs;
};
vector<Schain> schains;
Starget target = None;

Stensor &Stensor::scope(Var v) {
    v_scope = v;
    return *this;
}

Stensor &Stensor::banks(const vector<Var> &v) {
    if (v.empty()) {
        // By default, this stensor will output a scalar each time.
        return *this;
    }
    v_banks = v;
    return *this;
}

Stensor &Stensor::out(const vector<Var> &v) {
    if (v.empty()) {
        // By default, this stensor will output a scalar each time.
        return *this;
    }
    v_outs = v;
    return *this;
}

Stensor &Stensor::transpose() {
    transposed = true;
    return *this;
}

Stensor &Stensor::operator()(const vector<Expr> &d) {
    if (d.empty()) {
        // By default, this stensor will use the original layout.
        return *this;
    }
    dims = d;
    return *this;
}

int find_stensor(string name, Stensor &ret) {
    for (auto &sc : schains) {
        for (size_t j = 0; j < sc.stensors.size(); j++) {
            if (sc.stensors[j].name == name) {
                ret = sc.stensors[j];
                return j;
            }
        }
    }
    return -1;
}

Stensor &Stensor::operator>>(Stensor &s) {
    int c = this->schain_idx;
    if (c >= 0) {
        s.schain_idx = c;
        schains[c].stensors.push_back(s);
        return schains[c].stensors.back();
    } else {
        // If the current stensor does not belong to any chain,
        // we create a new chain and add it into the new chain.
        internal_assert(!s.producer.empty());
        Stensor producer;
        internal_assert(find_stensor(s.producer, producer) >= 0);
        Schain tmp;
        s.schain_idx = schains.size();
        tmp.is_output = schains[producer.schain_idx].is_output;
        tmp.fork_from = producer;
        tmp.stensors.push_back(*this);
        tmp.stensors.push_back(s);
        schains.push_back(std::move(tmp));
        c = schains.size();
    }
    return schains[c].stensors.back();
}

Stensor &operator>>(const vector<ImageParam> &im, Stensor &s) {
    Schain tmp;
    s.schain_idx = schains.size();
    tmp.is_output = false;
    tmp.imp = im;
    tmp.stensors.push_back(s);
    schains.push_back(std::move(tmp));
    return schains.back().stensors.back();
}

void operator>>(const vector<ImageParam> &im, const vector<FuncOrStensor> &fs) {
    // Create a new chain for every stensor in fs
    for (size_t i = 0; i < fs.size(); i++) {
        Schain tmp;
        internal_assert(fs[i].stensor);
        fs[i].stensor->schain_idx = schains.size();
        tmp.is_output = false;
        tmp.imp = im;
        tmp.stensors.push_back(*(fs[i].stensor));
        schains.push_back(std::move(tmp));
    }
}

void operator>>(Stensor &s, Func &f) {
    if (s.schain_idx >= 0) {
        auto &sc = schains[s.schain_idx];
        sc.outf = f;
    } else {
        internal_assert(!s.producer.empty());
        Stensor producer;
        internal_assert(find_stensor(s.producer, producer) >= 0);
        Schain tmp;
        s.schain_idx = schains.size();
        tmp.is_output = schains[producer.schain_idx].is_output;
        tmp.fork_from = producer;
        tmp.stensors.push_back(s);
        tmp.outf = f;
        schains.push_back(std::move(tmp));
    }
}

void operator>>(Stensor &s, const vector<FuncOrStensor> &fs) {
    vector<Stensor*> isolated_stensors;
    bool has_func = false;
    for (size_t i = 0; i < fs.size(); i++) {
        if (fs[i].stensor) {
            isolated_stensors.push_back(fs[i].stensor);
        } else {
            // We asssume there is only one Func
            user_assert(!has_func);
            has_func = true;
            s >> (*fs[i].func);
        }
    }
    if (!has_func) {
        // Append the first stensor to the current stensor chain
        s >> (*isolated_stensors[0]);
        isolated_stensors.erase(isolated_stensors.begin());
    }
    for (size_t i = 0; i < isolated_stensors.size(); i++) {
        isolated_stensors[i]->producer = s.name;
    }
}

Stensor &operator>>(const ImageParam &im, Stensor &s) {
    return vector<ImageParam>{im} >> s;
}

void operator>>(const ImageParam &im, const vector<FuncOrStensor> &fs) {
    vector<ImageParam>{im} >> fs;
}

Stensor &operator>>(Func &f, Stensor &s) {
    Schain tmp;
    s.schain_idx = schains.size();
    tmp.is_output = true;
    tmp.outf = f;
    tmp.stensors.push_back(s);
    schains.push_back(std::move(tmp));
    return schains.back().stensors.back();
}

struct FindVars
{
    vector<Func> ures;
    map<string, vector<Var>> all_free_vars;      // Variables appeared in the function definition

    int var_index(Var v, Func func = Func()) {
        func = find_main_ure(func);
        auto free_vars = all_free_vars[func.name()];
        for (size_t i = 0; i < free_vars.size(); i++) {
            if (v.same_as(free_vars[i]))
                return i;
        }
        return -1;
    }

    bool exists(const vector<Var> &vars) {
        for (auto &v : vars) {
            if (var_index(v) == -1) {
                return false;
            }
        }
        return true;
    }

    bool exists(Var v) {
        return this->exists(vector<Var>{v});
    }

    vector<VarOrRVar> find_reuse_vars(string imp, Var scope, Func func = Func()) {
        vector<VarOrRVar> reuse_vars;
        auto &used_vars = fuv.used_vars;
        internal_assert(used_vars.count(imp) > 0);
        func = find_main_ure(func);
        for (Var v : all_free_vars[func.name()]) {
            if (v.same_as(scope)) {
                break;
            }
            if (used_vars[imp].count(v.name()) == 0) {
                reuse_vars.push_back(Var(v));
            }
        }
        return reuse_vars;
    }

    // Find the first var below/above the given var v whose extent is not 1
    Var find_non_unit_var(Var v, bool above = true, Func func = Func()) {
        func = find_main_ure(func);
        auto free_vars = all_free_vars[func.name()];
        size_t j = var_index(v, func);
        while (j > 0 && j < free_vars.size()) {
            auto bound = func.function().get_bounds(free_vars[j].name());
            if (!is_one(bound.second)) break;
            j = above ? j + 1 : j - 1;
        }
        return free_vars[j];
    }

    vector<Expr> get_access_index(string name, Var scope, Func func = Func()) {
        func = find_main_ure(func);
        auto free_vars = all_free_vars[func.name()];
        auto args = fuv.access_indexes[name];
        map<string, Expr> loop_to_0;
        for (int i = free_vars.size()-1; i >= 0; i--) {
            loop_to_0[free_vars[i].name()] = 0;
            if (free_vars[i].name() == scope.name()) {
                break;
            }
        }
        vector<Expr> new_args;
        for (size_t i = 0; i < args.size(); i++) {
            Expr tmp = substitute(loop_to_0, args[i]);
            new_args.push_back(simplify(tmp));
        }
        return new_args;
    }

    Func find_main_ure(Func func = Func()) {
        if (!func.defined()) {
            return ures[0];
        }
        for (auto u : ures) {
            auto merged_funcs = u.function().definition().schedule().merged_funcs();
            auto it = std::find_if(merged_funcs.begin(), merged_funcs.end(),
                                   [&](Function &f){ return f.name() == func.name(); });
            if (it != merged_funcs.end()) {
                return u;
            }
        }
        return func;
    }

    // Find variables appeared in the arguments of inputs
    class FindUsedVars : public IRVisitor
    {
        string image_param;
    public:
        using IRVisitor::visit;
        map<string, vector<Expr>> access_indexes;
        map<string, std::set<string>> used_vars;

        void visit(const Variable *op) override {
            if (!image_param.empty()) {
                used_vars[image_param].insert(op->name);
            }
        }

        void visit(const Call *op) override {
            if (ends_with(op->name, "_im")) {
                image_param = remove_postfix(op->name, "_im");
                access_indexes[image_param] = op->args;
                for (size_t i = 0; i < op->args.size(); i++) {
                    op->args[i].accept(this);
                }
                image_param.clear();
            }
        }
    } fuv;

    FindVars(const map<string, Func> &env) {
        for (auto &p : env) {
            const Func &f = p.second;
            f.value().accept(&fuv);
            // UREs have the same iteration space, so we just check the one applied merge_ures
            if (!f.function().definition().schedule().is_merged()) {
                if (f.function().has_merged_defs()) {
                    ures.push_back(f);
                }
                for (auto &d : f.function().definition().schedule().dims()) {
                    all_free_vars[f.name()].push_back(d.var);
                }
            }
        }
        user_assert(!ures.empty())
            << "Cannot find merged UREs. Do you forget to apply merge_ures?";
    }
};

// We assume an output function always follows such a pattern:
// Out(...) = select(cond, Z(...)),
// in this case, we find and set func_name as Z
class FindProducerForOutput : public IRVisitor {
    const map<string, Func> env;
public:
    using IRVisitor::visit;
    Func producer;

    void visit(const Select *op) override {
        if (!op->false_value.defined()) {
            auto call = op->true_value.as<Call>();
            internal_assert(call);
            if (call->call_type == Call::CallType::Halide) {
                producer = env.at(call->name);
            }
        }
    }

    FindProducerForOutput(const map<string, Func> &_e)
        : env(_e) {}
};


class RealizeOnFPGA
{
    vector<Var> output_array_dims;
    FindVars &fv;
    FindProducerForOutput &fpo;

    void isolate_producer(Schain &c) {
        if (!c.imp.empty() && c.stensors[0].position != HOST) {
            // The device stensors needs serialized inputs
            // If the host stensor is not specified, we automatically generate it
            string host_name = c.imp[0].name() + (c.outf.defined() ? "_" + c.outf.name() : "") + "_serializer";
            Stensor s_host(host_name);
            s_host.schain_idx = c.stensors[0].schain_idx;
            c.stensors.insert(c.stensors.begin(), s_host);
        }
        vector<Func> producers;
        for (auto &s : c.stensors) {
            Place place = s.position == SMemType::HOST ? Place::Host : Place::Device;
            Func isolated_func(s.name, place);
            producers.push_back(std::move(isolated_func));
            debug(1) << "T2X emits: " << "Func " << s.name << "(\"" << s.name << "\", "
                     << (place == Place::Host ? "Place::Host" : "Place::Device") << ");\n";
        }

        if (c.imp.empty()) {
            int ori_chain = c.fork_from.schain_idx;
            auto imp = schains[ori_chain].imp;
            internal_assert(!imp.empty());

            auto &funcs = schains[ori_chain].funcs;
            Func fork_func;
            for (size_t i = 0; i < funcs.size(); i++) {
                if (funcs[i].name() == c.fork_from.name) {
                    fork_func = funcs[i];
                    break;
                }
            }
            internal_assert(fork_func.defined());
            vector<FuncOrExpr> imp_expr;
            std::copy(imp.begin(), imp.end(), std::back_inserter(imp_expr));
            auto func = fv.find_main_ure(c.outf);
            auto tmp_producers = producers;
            tmp_producers.insert(tmp_producers.begin(), fork_func);
            func.isolate_producer_chain(imp_expr, tmp_producers);
            debug(1) << "T2X emits: " << func.name() << ".isolate_producer_chain("
                     << names_to_string(imp) << ", " << names_to_string(tmp_producers) << ");\n";
        } else {
            vector<FuncOrExpr> imp_expr;
            std::copy(c.imp.begin(), c.imp.end(), std::back_inserter(imp_expr));
            auto func = fv.find_main_ure(c.outf);
            func.isolate_producer_chain(imp_expr, producers);
            debug(1) << "T2X emits: " << func.name() << ".isolate_producer_chain("
                     << names_to_string(c.imp) << ", " << names_to_string(producers) << ");\n";
        }
        c.funcs = producers;
    }

#if 0
    void generate_output_array(Func out, Func drainer) {
        // TODO: check non-output-stationary dataflow
        auto src_vars = fv.ure.function().definition().schedule().transform_params()[0].src_vars;
        vector<string> pe_array_dims(src_vars.begin(), src_vars.end()-1);
        auto func_dims = out.function().args();

        for (auto u : pe_array_dims) {
            for (auto o : func_dims) {
                if (o == u)
                    output_array_dims.push_back(Var(o));
            }
        }
        drainer.space_time_transform(output_array_dims);
        debug(1) << "T2X emits: " << drainer.name() << ".space_time_transform("
                 << names_to_string(output_array_dims) << ");\n";
    }
#endif

    void isolate_consumer(Schain &c) {
        vector<Func> consumers;
        // If the host stensor is not specified, we automatically generate it
        // if (c.stensors.back().position == DRAM) {
        //     string host_name = c.outf.name() + "_deserializer";
        //     Stensor s_host(host_name);
        //     s_host.schain_idx = c.stensors[0].schain_idx;
        //     c.stensors.push_back(s_host);
        // }

        // Isolate subsequent consumers
        for (auto &s : c.stensors) {
            Place place = s.position == SMemType::HOST ? Place::Host : Place::Device;
            Func new_func(s.name, place);
            consumers.push_back(std::move(new_func));
            debug(1) << "T2X emits: " << "Func " << s.name << "(\"" << s.name << "\", "
                     << (place == Place::Host ? "Place::Host" : "Place::Device") << ");\n";
        }
        if (c.stensors[0].v_banks.size() == 1 && c.stensors[0].position == REG) {
            // This is a special case where the single dimension banks are inside systolic array
            Var bank = c.stensors[0].v_banks[0];
            c.outf.value().accept(&fpo);
            c.outf.relay(fpo.producer, bank);
            debug(1) << "T2X emits: " << c.outf.name() << ".relay("
                     << fpo.producer.name() << ", " << bank.name() << ");\n";
            // The channel is inside the systolic array
            if (c.stensors[0].fifo_depth != 0) {
                c.outf.min_depth(c.stensors[0].fifo_depth);
            }
            // Remove the first stensor as it is inside systolic array
            c.stensors.erase(c.stensors.begin());
            consumers.erase(consumers.begin());
            // Vectorize all the subsequent stensors
            if (!consumers.empty()) {
                c.outf.isolate_consumer_chain(consumers);
                debug(1) << "T2X emits: " << c.outf.name() << ".isolate_consumer_chain("
                        << names_to_string(consumers) << ");\n";
            }
            for (auto &f : consumers) {
                f.vectorize(bank);
                debug(1) << "T2X emits: " << f.name() << ".vectorize("
                         << bank.name() << ");\n";
            }
        } else if (c.stensors[0].v_banks.size() == 2 && c.stensors[0].position == REG) {
            // The output stensor inherits loops of the output URE, generally less than that of systolic array
            // So we isolate the first consumer alone and apply space-time transform to regenerate loop structure,
            // then the subsequent stensors could be isolated based on that
            Func first_func = consumers[0];
            c.outf.isolate_consumer(first_func);
            debug(1) << "T2X emits: " << c.outf.name() << ".isolate_consumer("
                     << first_func.name() << ");\n";
            // generate_output_array(outf, f_dev);
            first_func.space_time_transform(c.stensors[0].v_banks);
            debug(1) << "T2X emits: " << first_func.name() << ".space_time_transform("
                     << names_to_string(c.stensors[0].v_banks) << ");\n";
            vector<Func> other_cons(consumers.begin()+1, consumers.end());
            first_func.isolate_consumer_chain(other_cons);
            debug(1) << "T2X emits: " << first_func.name() << ".isolate_consumer_chain("
                     << names_to_string(other_cons) << ");\n";
        } else {
            // Inherit the PE array
            internal_assert(consumers.size() == c.stensors.size());
            c.outf.isolate_consumer_chain(consumers);
            debug(1) << "T2X emits: " << c.outf.name() << ".isolate_consumer_chain("
                     << names_to_string(consumers) << ");\n";
        }
        c.funcs = consumers;
    }

    // Remove reuse variables from stensors as per their scope
    void remove(Schain &c) {
        auto &producers = c.funcs;
        vector<VarOrRVar> loops;
        Var scope = c.stensors.back().v_scope;

        for (int i = producers.size()-2; i >= 0; i--) {
            ImageParam im;
            if (!c.imp.empty()) {
                im = c.imp[0];
            } else {
                int ori_chain = c.fork_from.schain_idx;
                im = schains[ori_chain].imp[0];
            }
            loops = fv.find_reuse_vars(im.name(), scope, c.outf);
            producers[i].remove(loops);
            debug(1) << "T2X emits: " << producers[i].name() << ".remove("
                     << names_to_string(loops) << ");\n";
            scope = (i > 0) ? c.stensors[i].v_scope : scope;
        }
    }

    Var find_differences(vector<Var> set_a, vector<Var> set_b) {
        for (auto &a : set_a) {
            bool found = false;
            for (auto &b : set_b)
                if (a.name() == b.name()) found = true;
            if (!found) return a;
        }
        return Var("");
    }

    // The scatter primitive only applies to the stensors with increasing dimensional banks (0->1, 1->2)
    void scatter(Schain &c) {
        auto &producers = c.funcs;
        internal_assert(c.stensors.size() == producers.size());

        for (size_t i = 0; i < c.stensors.size(); i++) {
            auto v_banks = c.stensors[i].v_banks;
            auto position = c.stensors[i].position;
            Func prev;
            vector<Var> prev_dims;
            if (i != 0) {
                prev = producers[i-1];
                prev_dims = c.stensors[i-1].v_banks;
            } else {
                Stensor tmp;
                int j = find_stensor(c.stensors[i].producer, tmp);
                if (j >= 0) {
                    prev = schains[c.fork_from.schain_idx].funcs[j];
                    prev_dims = tmp.v_banks;
                } else {
                    internal_assert(position != SRAM);
                    prev = producers[0];
                    prev_dims = c.stensors[0].v_banks;
                }
            }
            if (position == SRAM) {
                if (v_banks.size() == prev_dims.size()+1) {
                    Var v_scatter = find_differences(v_banks, prev_dims);
                    producers[i].scatter(prev, v_scatter);
                    debug(1) << "T2X emits: " << producers[i].name() << ".scatter("
                             << prev.name() << ", " << v_scatter << ");\n";
                }
                if (c.stensors[i].transposed) {
                    producers[i].scatter(prev, v_banks[0], ScatterStrategy::ForwardVector);
                    debug(1) << "T2X emits: " << producers[i].name() << ".scatter("
                             << prev.name() << ", " << v_banks[0] << ", ScatterStrategy::ForwardVector);\n";
                }
            }
            prev_dims = v_banks;
        }
    }

    // The gather primitive only applies to the stensors with decreasing dimensional banks (2->1, 1->0)
    void gather(Schain &c) {
        auto &consumers = c.funcs;
        internal_assert(c.stensors.size() == consumers.size());
        auto &prev_dims = c.stensors[0].v_banks;

        for (size_t i = 1; i < c.stensors.size(); i++) {
            auto v_banks = c.stensors[i].v_banks;
            auto position = c.stensors[i].position;
            if (position == REG && v_banks.size() == prev_dims.size()-1) {
                Func prev_1 = consumers[i-1];
                Func prev_2 = (i == 1) ? c.outf : consumers[i-2];
                Var v_gather = find_differences(prev_dims, v_banks);
                prev_1.gather(prev_2, v_gather);
                debug(1) << "T2X emits: " << prev_1.name() << ".gather("
                         << prev_2.name() << ", " << v_gather << ");\n";
                // Trick: The behavior of gather depends on bank dimensions
                // 2->1: Values transferred one by one via shift registers
                // 1->0: Values are gathered across banks and sent as a vector,
                //       to simplify vectorize phase, we perform it here
                if (v_banks.size() == 0) {
                    // producer
                    prev_1.vectorize(v_gather);
                    debug(1) << "T2X emits: " << prev_1.name() << ".vectorize("
                             << v_gather << ");\n";
                    // consumer
                    consumers[i].vectorize(v_gather);
                    debug(1) << "T2X emits: " << consumers[i].name() << ".vectorize("
                             << v_gather << ");\n";
                }
            }
            prev_dims = v_banks;
        }
    }

    void buffer(Schain &c) {
        auto &producers = c.funcs;
        internal_assert(c.stensors.size() == producers.size());
        for (size_t i = 0; i < c.stensors.size(); i++) {
            Var v_scope = c.stensors[i].v_scope;
            if (fv.exists(v_scope) && c.stensors[i].position == SRAM) {
                Func prev;
                if (i != 0) {
                    prev = producers[i-1];
                } else {
                    Stensor tmp;
                    int j = find_stensor(c.stensors[i].producer, tmp);
                    internal_assert(j >= 0);
                    prev = schains[c.fork_from.schain_idx].funcs[j];
                }
                if (c.stensors[i].transposed) {
                    // Insert an addressable buffer. Get the access index first
                    auto imp = c.imp;
                    if (imp.empty()) {
                        int ori_chain = c.fork_from.schain_idx;
                        imp = schains[ori_chain].imp;
                    }
                    internal_assert(imp.size() == 1);
                    auto args = fv.get_access_index(imp[0].name(), v_scope, c.outf);
                    internal_assert(args.size() == 2);
                    vector<Expr> transposed_args = { args[1], args[0] };

                    producers[i].addressable_buffer(prev, v_scope, transposed_args, args);
                    debug(1) << "T2X emits: " << producers[i].name() << ".addressable_buffer("
                             << prev.name() << ", " << v_scope << ", {"
                             << to_string(transposed_args) << "}, {" << to_string(args) << "});\n";
                } else {
                    internal_assert(i > 0);
                    auto &remove_params = c.funcs[i-1].function().definition().schedule().remove_params();
                    if (remove_params.empty()) continue;
                    producers[i].buffer(prev, v_scope);
                    debug(1) << "T2X emits: " << producers[i].name() << ".buffer("
                             << prev.name() << ", " << v_scope << ");\n";
                }
            }
        }
    }

    void vectorize(Schain &c) {
        vector<Func> &funcs = c.funcs;
        internal_assert(c.stensors.size() == funcs.size());
        // In general, each stensor could independently specify bankwidth,
        // so we do not check the consistency between the producer and consumer,
        // NOTE: Currently the producer and consumer must be consistent with manual work,
        // and we leave the sophisticated vectorization to future work
        for (size_t i = 0; i < c.stensors.size(); i++) {
            if (c.stensors[i].v_width.size() == 0)
                continue;
            user_assert(c.stensors[i].v_width.size() == 1)
                << "Currently we only support packing one dimension as a vector\n";
            Var v_width = c.stensors[i].v_width[0];
            if (fv.exists(v_width)) {
                const auto &dims = funcs[i].function().definition().schedule().dims();
                const auto iter = std::find_if(dims.begin(), dims.end(), [](const Dim &d){
                    return d.for_type == ForType::Vectorized;
                });
                if (iter != dims.end() && iter->var != v_width.name() && !ends_with(iter->var, "." + v_width.name())) {
                    user_warning << "Func " << funcs[i].name() << " can't vectorize across "
                                 << v_width <<", because Func is already vectorized across " << iter->var << "\n";
                } else {
                    debug(1) << "T2X emits: " << funcs[i].name() << ".vectorize("
                            << v_width << ");\n";
                    funcs[i].vectorize(v_width);
                }
            }
        }

        // To make UREs be consistent with its producer, we vectorize UREs as well
        auto &last_stensor = c.stensors.back();
        if (!c.is_output && last_stensor.v_width.size() > 0) {
            Var last_width = last_stensor.v_width[0];
            Func main_func = fv.find_main_ure(c.outf);
            const auto &dims = main_func.function().definition().schedule().dims();
            const auto iter = std::find_if(dims.begin(), dims.end(), [](const Dim &d){
                return d.for_type == ForType::Vectorized;
            });
            if (iter != dims.end() && iter->var != last_width.name() && !ends_with(iter->var, "." + last_width.name())) {
                user_warning << "Func " << main_func.name() << " can't vectorize across "
                             << last_width <<", because Func is already vectorized across " << iter->var << "\n";
            } else {
                debug(1) << "T2X emits: " << main_func.name() << ".vectorize("
                        << last_width << ");\n";
                main_func.vectorize(last_width);
            }
        }
    }

    void partition(Schain &c) {
        vector<Func> &funcs = c.funcs;
        internal_assert(c.stensors.size() == funcs.size());
        for (size_t i = 0; i < c.stensors.size(); i++) {
            if (c.stensors[i].v_outs.size() == 0)
                continue;
            if (c.stensors[i].position == DRAM) {
                user_assert(c.stensors[i].v_outs.size() == 1)
                    << "Currently we only support packing one dimension as a vector\n";
                auto v_width = c.stensors[i].v_outs[0];
                if (fv.exists(v_width)) {
                    Func main_ure = fv.find_main_ure(c.outf);
                    Type type = main_ure.output_types()[0];
                    auto bound = main_ure.function().get_bounds(v_width.name());
                    auto ext_node = bound.second.as<IntImm>();
                    internal_assert(ext_node);
                    int num_banks = (ext_node->value * type.bits()) / 512;
                    if (num_banks > 1 && v_width.name() == main_ure.function().definition().schedule().dims()[0].var) {
                        internal_assert(c.imp.size() == 1);
                        funcs[i-1].partition(c.imp[0], funcs[i], v_width, num_banks);
                        debug(1) << "T2X emits: " << funcs[i-1].name() << ".partition("
                                << c.imp[0].name() << ", " << funcs[i].name() << ", "
                                << v_width.name() << ", " << num_banks << ");\n";
                    }
                }
            }
        }
    }

    void min_depth(Schain &c) {
        vector<Func> &funcs = c.funcs;
        internal_assert(c.stensors.size() == funcs.size());
        for (size_t i = 0; i < c.stensors.size(); i++) {
            size_t d = c.stensors[i].fifo_depth;
            if (d > 0) {
                funcs[i].min_depth(d);
                debug(1) << "T2X emits: " << funcs[i].name() << ".min_depth("
                         << d << ");\n";
            }
        }
    }

    // Check if the stensors are inclusive cache
    // Namely, for input chain the scope of consumer cannot be beyond its predecessor,
    // for output chain the scope of consumer cannot below its predecessor
    // If not specified, the scope is inherited
    void check_inclusiveness(Schain &c) {
        Func func = fv.find_main_ure(c.outf);
        if (!c.is_output) {
            // start from the outermost loop
            int i = fv.all_free_vars[func.name()].size()-1;
            for (auto &s: c.stensors) {
                if (!fv.exists(s.v_scope)) {
                    s.v_scope = fv.all_free_vars[func.name()][i];
                    continue;
                }
                int j = fv.var_index(s.v_scope, func);
                user_assert(j > 0 && j <= i)
                    << "The scope of " << s.name << " is beyond its predecessor\n";
                // Find a loop whose extent is not 1, otherwise this loop would be removed in lowering
                s.v_scope = fv.find_non_unit_var(s.v_scope, true, func);
                i = j;
            }
        }
    }

    // Triangular loops often look like this:
    // for (i, 0, I)
    //   for (k, i, K-i)
    // We set the storage bound of loop k as K/2+1
    void set_triangular_bound(Schain &c) {
        // For producer, the buffer is allocated in the first host stensor (serializer), otherwise the last DRAM stensor
        Func f;
        vector<Func> &funcs = c.funcs;
        if (!c.is_output) {
            if (c.stensors[0].position != HOST) return;
            f = funcs[0];
        } else {
            for (size_t i = 0; i < c.stensors.size(); i++) {
                if (c.stensors[i].position == DRAM) {
                    f = funcs[i];
                    break;
                }
            }
            if (!f.defined()) return;
        }
        auto dims = f.function().definition().schedule().dims();
        for (size_t i = 0; i < dims.size()-1; i++) {
            auto bound = f.function().get_bounds(dims[i].var);
            if (!is_const(bound.first)) {
                // We assume the outer triangular loop is at the outermost level
                auto outermost_var = dims[dims.size()-2].var;
                auto min_var = bound.first.as<Variable>();
                internal_assert(min_var && min_var->name == outermost_var);
                auto ori_ext = simplify(bound.second + bound.first);
                auto remove_dims = f.function().definition().schedule().remove_params();
                Var inner(dims[i].var);
                if (std::find(remove_dims.begin(), remove_dims.end(), inner.name()) == remove_dims.end()) {
                    if (std::find(remove_dims.begin(), remove_dims.end(), min_var->name) == remove_dims.end()) {
                        // If the outer triangular loop is not removed, we set the storage bound as the half
                        f.bound_storage(inner, 0, ori_ext/2+1);
                        debug(1) << "T2X emits: " << f.name() << ".bound_storage("
                                 << inner.name() << ", 0, " << ori_ext << "/2+1);\n";
                    } else {
                        // If the outer triangular loop is removed, we set the storage bound as its original bound
                        f.bound_storage(inner, 0, ori_ext);
                        debug(1) << "T2X emits: " << f.name() << ".bound_storage("
                                 << inner.name() << ", 0, " << ori_ext << ");\n";
                    }
                }
            }
        }
    }

    void find_banks(Schain &c) {
        // The dst_vars includes space loops plus one time loop
        Func func = fv.find_main_ure(c.outf);
        auto stt_param = func.function().definition().schedule().transform_params();
        if (stt_param.empty()) {
            return;
        }
        auto dst_vars = stt_param[0].dst_vars;
        for (auto &s : c.stensors) {
            for (auto &v : s.v_outs) {
                auto p = std::find_if(dst_vars.begin(), dst_vars.end()-1,
                                    [&](string &n){ return v.name() == n; });
                if (p != dst_vars.end()-1) {
                    // Find it is in space loops, so we view it as a bank
                    s.v_banks.push_back(Var(*p));
                } else {
                    // Otherwise view it as bankwidth
                    s.v_width.push_back(Var(*p));
                }
            }
        }
    }

public:
    RealizeOnFPGA(FindVars &_v, FindProducerForOutput &_p)
        : fv(_v), fpo(_p) {}

    void realize() {
        for (auto &c: schains) {
            check_inclusiveness(c);
            find_banks(c);
            if (!c.is_output) {
                isolate_producer(c);
                remove(c);
                set_triangular_bound(c);
                scatter(c);
                buffer(c);
                vectorize(c);
                partition(c);
                min_depth(c);
            } else {
                isolate_consumer(c);
                set_triangular_bound(c);
                gather(c);
                vectorize(c);
                partition(c);
                min_depth(c);
            }
        }
    }
};

class RealizeOnGPU
{
    FindVars &fv;
    int num_gpu_vars;

    // Check if the stensors are inclusive cache
    // Namely, for input chain the scope of consumer cannot be beyond its predecessor,
    // for output chain the scope of consumer cannot below its predecessor
    // If not specified, the scope is inherited
    void check_inclusiveness(Schain &c) {
        if (!c.is_output) {
            // start from the outermost loop
            Func f = fv.ures[0];
            auto free_vars = fv.all_free_vars[f.name()];
            int i = free_vars.size()-1;
            for (auto &s: c.stensors) {
                if (!fv.exists(s.v_scope)) {
                    s.v_scope = free_vars[i];
                    continue;
                }
                int j = fv.var_index(s.v_scope);
                user_assert(j > 0 && j <= i)
                    << "The scope of " << s.name << " is beyond its predecessor\n";
                // Find a loop whose extent is not 1, otherwise this loop would be removed in lowering
                s.v_scope = fv.find_non_unit_var(s.v_scope);
                i = j;
            }
        }
    }

    void gpu_fetch(Schain &c) {
        for (auto &s : c.stensors) {
            // Currently, we separately allocate registers in each thread, and view registers
            // throughout threads as an unified SRAM storage, to realize stensors on GPUs.
            if (s.position == SRAM) {
                for (auto &p : c.imp) {
                    Func f = fv.ures[0];
                    auto free_vars = fv.all_free_vars[f.name()];
                    int gpu_var_index = free_vars.size() - num_gpu_vars -1;
                    Var loop = fv.var_index(s.v_scope) < gpu_var_index ? s.v_scope : free_vars[gpu_var_index];
                    p.gpu_fetch(loop, MemoryType::Register, s.v_outs);
                    debug(1) << "T2X emits: " << p.name() << ".gpu_fetch("
                             << loop.name() << ", {" << names_to_string(s.v_outs) << "});\n";
                }
            }
        }
    }

    void gpu_store(Schain &c) {
        auto &s = c.stensors.back();
        vector<Expr> args;
        if (!s.dims.empty()) {
            args = s.dims;
        } else {
            vector<Var> vars = c.outf.args();
            for (auto v : vars) {
                args.push_back(v);
            }
        }
        c.outf.gpu_store(args, s.name);
        debug(1) << "T2X emits: " << c.outf.name() << ".gpu_store({"
                 << to_string(args) << "}, " << s.name << ");\n";
    }

public:
    RealizeOnGPU(FindVars &_f, int _n)
        : fv(_f), num_gpu_vars(_n) {}

    Func realize() {
        Func out;
        for (auto &c : schains) {
            check_inclusiveness(c);
            if (!c.is_output) {
                gpu_fetch(c);
            } else {
                gpu_store(c);
                out = c.outf;
            }
        }
        return out;
    }
};

Func &operator>>(Func &func, const FIFO &fifo) {
    func.min_depth(fifo.depth);
    debug(1) << "T2X emits: " << func.name() << ".min_depth("
             << fifo.depth << ");\n";
    return func;
}

Stensor &operator>>(Stensor &s, const FIFO &fifo) {
    s.fifo_depth = fifo.depth;
    return s;
}

void Stensor::realize(Starget t) {
    map<string, Func> env;
    user_assert(schains.back().is_output)
        << "Please specify an output path as the last stensor chain\n";
    target = t;
    Func outf = schains.back().outf;
    env = outf.pipeline().compute_environment();

    Func f;
    if (t == Starget::IntelFPGA) {
        FindVars fv(env);
        FindProducerForOutput fpo(env);
        RealizeOnFPGA fpga(fv, fpo);
        fpga.realize();
    } else {
        int num_gpu_vars = 0;
        for (auto &p : env) {
            if (p.second.function().place() == Place::Device) {
                // Placing on device is only valid for FPGAs
                p.second.function().place(Place::Host);
            }
            reorder_gpu_loops(p.second, num_gpu_vars);
        }
        FindVars fv(env);
        RealizeOnGPU gpu(fv, num_gpu_vars);
        gpu.realize();
    }
}

Func Stensor::get_wrapper_func() {
    user_assert(target != None)
        << "Please apply Stensor::realize(target) before getting its wrapper function.\n";
    int c = this->schain_idx;
    auto &sc = schains[c];
    for (size_t i = 0; i < sc.stensors.size(); ++i) {
        if (sc.stensors[i].name == this->name) {
            if (target == IntelGPU) return sc.outf;
            return sc.funcs[i];
        }
    }
    return Func();
}

Func Stensor::stensor_realize_wrapper(Starget t) {
    Func f;
    realize(t);
    for (auto &sc : schains) {
        if (sc.is_output) {
            internal_assert(!f.defined());
            if (t == IntelFPGA) {
                f = sc.funcs.back();
            } else {
                f = sc.outf;
            }
            internal_assert(f.function().place() == Place::Host);
        }
    }
    return f;
}

void Stensor::realize(Buffer<> dst, Starget t) {
    Func f = stensor_realize_wrapper(t);
    if (t == Starget::IntelFPGA) {
        Target acc = get_host_target();
        acc.set_feature(Target::IntelFPGA);
        acc.set_feature(Target::EnableSynthesis);
        f.realize(dst, acc);
    }
    if (t == Starget::IntelGPU) {
        user_error << "Currently the GPU runtime is under developement\n";
    }
}

void Stensor::compile_jit(Starget t) {
    Func f = stensor_realize_wrapper(t);
    if (t == Starget::IntelFPGA) {
        Target acc = get_host_target();
        acc.set_feature(Target::IntelFPGA);
        acc.set_feature(Target::EnableSynthesis);
        f.compile_jit(acc);
    }
}

void Stensor::compile_to_host(string file_name, const vector<Argument> &args,
                              const std::string fn_name, Starget t) {
    Func f = stensor_realize_wrapper(t);
    if (t == Starget::IntelFPGA) {
        Target acc = get_host_target();
        acc.set_feature(Target::IntelFPGA);
        acc.set_feature(Target::EnableSynthesis);
        f.compile_to_host(file_name, args, fn_name, acc);
    }
    if (t == Starget::IntelGPU) {
        user_warning << "Currently the GPU runtime is under developement, "
                        "so we just emit out the source code in " << fn_name << "_genx.cpp\n";
        Target acc = get_host_target();
        acc.set_feature(Target::IntelGPU);
        f.compile_to_cm(fn_name, std::move(args), acc);
    }
}


void Stensor::compile_to_oneapi(const vector<Argument> &args,
                              const std::string fn_name, Starget t) {
    Func f = stensor_realize_wrapper(t);
    Target acc = get_host_target();
    acc.set_feature(Target::OneAPI);
    if (t == Starget::IntelFPGA) {
        acc.set_feature(Target::IntelFPGA);
        acc.set_feature(Target::EnableSynthesis);
        f.compile_to_oneapi(args, fn_name, acc);
    }
}


}
