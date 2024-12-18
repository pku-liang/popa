#include "DataPlacement.h"
#include "Func.h"
#include "IR.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {

Func &Func::place_across_banks(FuncOrExpr in, Func device_func, int num_banks) {
    return *this;
}

namespace Internal {

Stmt place_data_accrss_banks(Stmt s, const std::map<std::string, Function> &env) {
    return s;
}

} // Internal
} // Halide