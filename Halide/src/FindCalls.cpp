#include "FindCalls.h"

#include "ExternFuncArgument.h"
#include "Function.h"
#include "IRVisitor.h"
#include "../../t2s/src/Overlay.h"
#include <utility>

namespace Halide {
namespace Internal {

namespace {

/* Find all the internal halide calls in an expr */
class FindCalls : public IRVisitor {
public:
    std::map<std::string, Function> calls;
    std::vector<Function> order;

    using IRVisitor::visit;

    void include_function(const Function &f) {
        auto [it, inserted] = calls.emplace(f.name(), f);
        if (inserted) {
            order.push_back(f);
        } else {
            user_assert(it->second.same_as(f))
                << "Can't compile a pipeline using multiple functions with same name: "
                << f.name() << "\n";
        }
    }

    void visit(const Call *call) override {
        IRVisitor::visit(call);

        if (call->call_type == Call::Halide && call->func.defined()) {
            Function f(call->func);
            include_function(f);
        }
    }
};

void find_merge_funcs(FindCalls& calls, const Function& f) {
    if (f.has_merged_defs()) { 
        for (auto g : f.definition().schedule().merged_funcs()) {
            if (calls.calls.find(g.name()) == calls.calls.end()) {
                g.accept(&calls);
                calls.calls[g.name()] = g;
                find_merge_funcs(calls, g);
            }
        }
    } 
}

void populate_environment_helper(const Function &f,
                                 std::map<std::string, Function> *env,
                                 std::vector<Function> *order,
                                 bool recursive = true,
                                 bool include_wrappers = false,
                                 bool include_merge_funcs = false) {
    std::map<std::string, Function>::const_iterator iter = env->find(f.name());
    if (iter != env->end()) {
        user_assert(iter->second.same_as(f))
            << "Can't compile a pipeline using multiple functions with same name: "
            << f.name() << "\n";
        return;
    }

    auto insert_func = [](const Function &f,
                          std::map<std::string, Function> *env,
                          std::vector<Function> *order) {
        bool inserted = env->emplace(f.name(), f).second;
        if (inserted) {
            order->push_back(f);
        }
    };

    FindCalls calls;
    f.accept(&calls);
    if (f.has_extern_definition()) {
        for (const ExternFuncArgument &arg : f.extern_arguments()) {
            if (arg.is_func()) {
                insert_func(Function{arg.func}, &calls.calls, &calls.order);
            }
        }
    }

    if (include_merge_funcs) {
        find_merge_funcs(calls, f);
    }

    if (include_wrappers) {
        for (const auto &it : f.schedule().wrappers()) {
            insert_func(Function{it.second}, &calls.calls, &calls.order);
        }
    }

    if (!recursive) {
        for (const Function &g : calls.order) {
            insert_func(g, env, order);
        }
    } else {
        insert_func(f, env, order);
        for (const Function &g : calls.order) {
            populate_environment_helper(g, env, order, recursive, include_wrappers);
        }
    }

    // find dependent tasks in overlay and add them to env
    auto &task_funcs = f.overlay().definition().taskItems();
    auto &task_deps = f.definition().schedule().task_deps();
    for (auto &kv : task_deps) {
        auto task = task_funcs[kv.first];
        // task not in env
        if (env->find(task.name()) == env->end()) {
            populate_environment_helper(task, env, order, recursive, include_wrappers);
        }
    }
}

}  // namespace

std::map<std::string, Function> build_environment(const std::vector<Function> &funcs) {
    std::map<std::string, Function> env;
    std::vector<Function> order;
    for (const Function &f : funcs) {
        populate_environment_helper(f, &env, &order, true, true);
    }
    return env;
}

std::vector<Function> called_funcs_in_order_found(const std::vector<Function> &funcs) {
    std::map<std::string, Function> env;
    std::vector<Function> order;
    for (const Function &f : funcs) {
        populate_environment_helper(f, &env, &order, true, true);
    }
    return order;
}

std::map<std::string, Function> find_transitive_calls(const Function &f) {
    std::map<std::string, Function> res;
    std::vector<Function> order;
    populate_environment_helper(f, &res, &order, true, false);
    return res;
}

std::map<std::string, Function> find_direct_calls(const Function &f) {
    std::map<std::string, Function> res;
    std::vector<Function> order;
    populate_environment_helper(f, &res, &order, false, false);
    return res;
}

}  // namespace Internal
}  // namespace Halide
