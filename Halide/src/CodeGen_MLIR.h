#ifndef CODEGEN_MLIR_H
#define CODEGEN_MLIR_H

#include "IRVisitor.h"
namespace Halide {

namespace Internal {

class CodeGen_MLIR : public IRVisitor {
public:
  CodeGen_MLIR(const Target&);
  ~CodeGen_MLIR() override = default;

  void compile(const Module& M);

};

} // namespace Internal
} // namespace Halide

#endif
