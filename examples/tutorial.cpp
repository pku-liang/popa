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

Func exp_0(Buffer<int> &A, Buffer<int> &B)
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

Func exp_1(Buffer<int> &A, Buffer<int> &B)
{
    Var i("i"), j("j"), k("k");
    URE X("X", Int(32), {k, j, i}), Y("Y", Int(32), {k, j, i}), Z("Z", Float(32), {k, j, i}), C("C");
    X(k, j, i) = select(j == 0, A(k, i), X(k, j-1, i));
    Y(k, j, i) = select(i == 0, B(j, k), Y(k, j, i-1));
    Z(k, j, i) = select(k == 0, 0, Z(k-1, j, i)) + X(k, j, i) * Y(k, j, i);
    C(j, i) = select(k == N-1, Z(k, j, i));

    X.merge_ures(Y, Z, C);
    X.set_bounds(i, 0, M,
                 j, 0, S,
                 k, 0, N);
    X.space_time_transform(j, i);

    C.output_buffer().dim(0).set_bounds(0, S).set_stride(1);
    C.output_buffer().dim(1).set_bounds(0, M).set_stride(S);
    return C;
}

Func exp_2(Buffer<int> &A, Buffer<int> &B)
{
    #define P           jj, ii, k, j, i
    Var i("i"), j("j"), k("k"), ii("ii"), jj("jj");
    URE X("X", Int(32), {P}), Y("Y", Int(32), {P}), Z("Z", Int(32), {P}), C("C");
    X(P) = select(jj == 0, A(k, ii + II*i), X(jj-1, ii, k, j, i));
    Y(P) = select(ii == 0, B(jj + JJ*j, k), Y(jj, ii-1, k, j, i));
    Z(P) = select(k == 0, 0, Z(jj, ii, k-1, j, i)) + X(P) * Y(P);
    C(jj, ii, j, i) = select(k == N-1, Z(P));
    #undef P

    X.merge_ures(Y, Z, C);
    X.set_bounds(i, 0, I, ii, 0, II)
     .set_bounds(j, 0, J, jj, 0, JJ,
                 k, 0, N);
    X.space_time_transform(jj, ii);

    C.output_buffer().dim(0).set_bounds(0, JJ).set_stride(1);
    C.output_buffer().dim(1).set_bounds(0, II).set_stride(JJ);
    C.output_buffer().dim(2).set_bounds(0, J).set_stride(II*JJ);
    C.output_buffer().dim(3).set_bounds(0, I).set_stride(II*JJ*J);
    return C;
}

Func exp_3(Buffer<int> &A, Buffer<int> &B)
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
        std::cerr << "Usage: " << argv[0] << " <0-3>\n";
        return 1;
    }
    int number = std::stoi(argv[1]);
    Buffer<int> A(N, M, "A"), B(S, N, "B");
    Func C;
    switch (number) {
        case 0: C = exp_0(A, B); break;
        case 1: C = exp_1(A, B); break;
        case 2: C = exp_2(A, B); break;
        case 3: C = exp_3(A, B); break;
        default:
            std::cerr << "Usage: " << argv[0] << " <0-3>\n";
            return 1;
    }
    string device_file = "mm_" + to_string(number) + ".mlir";
    string ir_file = "exp_" + to_string(number) + ".html";
    Target target = get_host_target();
    target.set_feature(Target::MLIR);
    C.compile_to_device(device_file, {}, target);
    C.compile_to_lowered_stmt(ir_file, {}, HTML, target);

    std::cout << "Generated file: " << device_file << ", " << ir_file << ".html\n";
    return 0;
}
