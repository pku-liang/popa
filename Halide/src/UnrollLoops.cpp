#include "UnrollLoops.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Substitute.h"
#include "UniquifyVariableNames.h"

#include "../../t2s/src/Utilities.h"

namespace Halide {
namespace Internal {

namespace {

class UnrollLoops : public IRMutator {
    using IRMutator::visit;

    const std::map<std::string, Function> &env;
    bool in_device{false};

    Stmt visit(const ProducerConsumer* op) override {
        Function func;
        if (op->is_producer && function_is_in_environment(op->name, env, func) && func.place() == Place::Device) {
            in_device = true;
        } else {
            in_device = false;
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const For *for_loop) override {
        // Device functions keep loop unrolling annotations instead of physically unrolling.
        if (!in_device && for_loop->for_type == ForType::Unrolled) {
            Stmt body = for_loop->body;
            const IntImm *e = for_loop->extent.as<IntImm>();

            internal_assert(e)
                << "Loop over " << for_loop->name << " should have had a constant extent\n";
            body = mutate(body);

            if (e->value == 1) {
                user_warning << "Warning: Unrolling a for loop of extent 1: " << for_loop->name << "\n";
            }

            Stmt iters;
            for (int i = e->value - 1; i >= 0; i--) {
                Stmt iter = substitute(for_loop->name, for_loop->min + i, body);
                // It's necessary to eagerly simplify this iteration
                // here to resolve things like muxes down to a single
                // item before we go and make N copies of something of
                // size N.
                iter = simplify(iter);
                if (!iters.defined()) {
                    iters = iter;
                } else {
                    iters = Block::make(iter, iters);
                }
            }

            return iters;

        } else {
            return IRMutator::visit(for_loop);
        }
    }
    bool permit_failed_unroll = false;

public:
    UnrollLoops(const std::map<std::string, Function> &env) : env(env) {
        // Experimental autoschedulers may want to unroll without
        // being totally confident the loop will indeed turn out
        // to be constant-sized. If this feature continues to be
        // important, we need to expose it in the scheduling
        // language somewhere, but how? For now we do something
        // ugly and expedient.

        // For the tracking issue to fix this, see
        // https://github.com/halide/Halide/issues/3479
        permit_failed_unroll = get_env_variable("HL_PERMIT_FAILED_UNROLL") == "1";
    }
};

}  // namespace

namespace {
class LoopReplacer : public IRMutator {
  public:
    LoopReplacer(const std::map<std::string, Function> &env, ForType type) : env(env), for_type(type) {}

  private:
    const std::map<std::string, Function> &env;
    ForType for_type;
    bool in_device{false};

    using IRMutator::visit;

    Stmt visit(const ProducerConsumer* op) override {
        Function func;
        if (op->is_producer && function_is_in_environment(op->name, env, func) && func.place() == Place::Device) {
            in_device = true;
        } else {
            in_device = false;
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const For* op) override {
        if (in_device && op->for_type == ForType::Unrolled) {
            Stmt body = mutate(op->body);
            return For::make(op->name, op->min, op->extent, for_type, op->partition_policy, op->device_api, body);
        } else {
            return IRMutator::visit(op);
        }
    }
};
}

Stmt unroll_loops(Stmt s, const std::map<std::string, Function> &env) {
    char *unroll = getenv("PRAGMAUNROLL");
    if (unroll != NULL) {
        LoopReplacer loop_replacer(env, ForType::PragmaUnrolled);
        s = loop_replacer.mutate(s);
    } else {
        /* If the extent of a space loop is too large, segfault might be caused during unrolling. 
        To avoid stack overflow, we define a env variable DELAYUNROLL and does not physically unroll
        loops in this phase. Loops are actually unrolled in codegen. */
        unroll = getenv("DELAYUNROLL");
        if (unroll != NULL) {
            LoopReplacer loop_replacer(env, ForType::DelayUnroll);
            s = loop_replacer.mutate(s);
        }
    }
    
    return UnrollLoops(env).mutate(s);
}

}  // namespace Internal
}  // namespace Halide
