#include <string>
#include "Halide.h"

// Matrix size
#define M       16
#define N       16
#define S       16
// Tiling factor
#define JJ      2
#define II      2
#define KK      2
#define I       (M/II)
#define J       (S/JJ)
#define K       (N/KK)

using namespace Halide;

Func vanilla(Buffer<int> &A, Buffer<int> &B)
{
    Var i("i"), j("j");
    RDom k(0, N, "k");
    Func C("C", Place::Device);
    C(j, i) = sum(A(k, i) * B(j, k));

    C.set_bounds(i, 0, M,
                 j, 0, S);

    C.output_buffer().dim(0).set_bounds(0, S).set_stride(1);
    C.output_buffer().dim(1).set_bounds(0, M).set_stride(S);
    return C;
}

Func optimized(Buffer<int> &A, Buffer<int> &B)
{
    #define P           kk, jj, ii, k, j, i
    Var i("i"), j("j"), k("k"), ii("ii"), jj("jj"), kk("kk");
    URE X("X", Int(32), {P}), Y("Y", Int(32), {P}), Z("Z", Int(32), {P}), C("C");
    X(P) = select(jj == 0, A(kk + KK*k, ii + II*i), X(kk, jj-1, ii, k, j, i));
    Y(P) = select(ii == 0, B(jj + JJ*j, kk + KK*k), Y(kk, jj, ii-1, k, j, i));
    Z(P) = select(k == 0 && kk == 0, 0,
                  select(kk == 0, Z(kk+KK-1, jj, ii, k-1, j, i), Z(kk-1, jj, ii, k, j, i))
                 ) + X(P) * Y(P);
    C(jj, ii, j, i) = select(kk == KK-1 && k == K-1, Z(P));
    #undef P

    X.merge_ures(Y, Z, C);
    X.set_bounds(i, 0, I, ii, 0, II)
     .set_bounds(j, 0, J, jj, 0, JJ)
     .set_bounds(k, 0, K, kk, 0, KK);
    X.space_time_transform(jj, ii);
    X.vectorize(kk);

    C.output_buffer().dim(0).set_bounds(0, JJ).set_stride(1);
    C.output_buffer().dim(1).set_bounds(0, II).set_stride(JJ);
    C.output_buffer().dim(2).set_bounds(0, J).set_stride(II*JJ);
    C.output_buffer().dim(3).set_bounds(0, I).set_stride(II*JJ*J);
    return C;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [vanilla/optimized]\n";
        return 1;
    }
    std::string version(argv[1]);
    Buffer<int> A(N, M, "A"), B(S, N, "B");
    Func C = (version == "vanilla") ? vanilla(A, B) : optimized(A, B);

    Target target = get_host_target();
    target.set_feature(Target::MLIR);
    string device_file = "SCF_" + version + ".mlir";
    string ir_file = "TensorIR_" + version;
    C.compile_to_device(device_file, {}, target);
    C.compile_to_lowered_stmt(ir_file, {}, Text, target);

    std::cout << "Generated file: " << device_file << ", " << ir_file << "\n";
    return 0;
}
