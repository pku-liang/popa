#include "CodeGen_MLIR.h"
#include "Module.h"
#include "Target.h"

namespace Halide {
namespace Internal {


CodeGen_MLIR::CodeGen_MLIR(const Target& t) {
}

void CodeGen_MLIR::compile(const Module& M) {
    internal_error << "This is a blank implementation. Please integrate an external code generator to hook into this function.";
}

} // namespace Internal
} // namespace Halide