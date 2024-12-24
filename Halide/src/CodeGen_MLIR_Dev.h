#ifndef HALIDE_CODEGEN_MLIR_H
#define HALIDE_CODEGEN_MLIR_H

/** \file
 * Defines the code-generator for producing MLIR code
 */

#include "IRVisitor.h"
#include "Scope.h"

namespace Halide {
namespace Internal {

struct CodeGen_GPU_Dev;

std::unique_ptr<CodeGen_GPU_Dev> new_CodeGen_MLIR_Dev(const Target &target);

}  // namespace Internal
}  // namespace Halide

#endif
