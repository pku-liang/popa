#include "Halide.h"

using namespace Halide;

int main(void) {
    ImageParam A(Float(32), 2, "A"), B(Float(32), 2, "B");
    Func C("C");
    Var i("i"), j("j"), ii("ii"), jj("jj");
    C(i, j) = 0.0f;
    RDom k(0, A.dim(1).extent(), "k");
    C(i, j) += A(i, k) * B(k, j);

    C.bound(i, 0, 16);
    C.bound(j, 0, 16);
    C.tile(i, j, ii, jj, 4, 4);

    std::map<Halide::OutputFileType, std::string> output = {{Halide::OutputFileType::mlir, "mlir"}};
    C.compile_to(output, {A, B}, "C", get_host_target());
}