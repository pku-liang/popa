#include <string>
#include "Halide.h"

// Matrix size
#define M       16
#define N       16
#define S       16

using namespace Halide;

Func mm_basic(Buffer<int> &A, Buffer<int> &B)
{
    Var i("i"), j("j");
    RDom k(0, N, "k");
    Func C("C", Place::Device);
    C(j, i) = sum(A(k, i) * B(j, k));

    C.set_bounds(i, 0, M,
                 j, 0, S);
    return C;
}

Func mm_SA(Buffer<int> &A, Buffer<int> &B)
{
    Var i("i"), j("j"), k("k");
    URE X("X", Int(32), {k, j, i}), Y("Y", Int(32), {k, j, i}), Z("Z", Int(32), {k, j, i}), C("C");
    X(k, j, i) = select(j == 0, A(k, i), X(k, j-1, i));
    Y(k, j, i) = select(i == 0, B(j, k), Y(k, j, i-1));
    Z(k, j, i) = select(k == 0, 0, Z(k-1, j, i)) + X(k, j, i) * Y(k, j, i);
    C(j, i) = select(k == N-1, Z(k, j, i));

    X.merge_ures(Y, Z, C);
    X.set_bounds(i, 0, M,
                 j, 0, S,
                 k, 0, N);

    Var ii("ii"), jj("jj");
    X.tile(j, i, jj, ii, 2, 2);
    X.space_time_transform(jj, ii);

    Var kk("kk");
    X.split(k, k, kk, 4);
    X.reorder(kk, jj, ii, k);
    X.vectorize(kk);

    return C;
}

Func mm_IO(Buffer<int> &A, Buffer<int> &B)
{
    Var i("i"), j("j"), k("k");
    URE X("X", Int(32), {k, j, i}), Y("Y", Int(32), {k, j, i}), Z("Z", Int(32), {k, j, i}), C("C");
    X(k, j, i) = select(j == 0, A(k, i), X(k, j-1, i));
    Y(k, j, i) = select(i == 0, B(j, k), Y(k, j, i-1));
    Z(k, j, i) = select(k == 0, 0, Z(k-1, j, i)) + X(k, j, i) * Y(k, j, i);
    C(j, i) = select(k == N-1, Z(k, j, i));

    X.merge_ures(Y, Z, C);
    X.set_bounds(i, 0, M,
                 j, 0, S,
                 k, 0, N);

    Var ii("ii"), jj("jj"), iii("iii"), jjj("jjj");
    X.tile(j, i, jj, ii, 4, 4);
    X.tile(jj, ii, jjj, iii, 2, 2);
    X.space_time_transform(jjj, iii);

    Var kk("kk");
    X.split(k, k, kk, 4);
    X.reorder(kk, jjj, iii, k);
    X.vectorize(kk);

    Stensor DA("DA", DRAM), DB("DB", DRAM), DC("DC", DRAM);
    Stensor SA("SA", SRAM), SB("SB", SRAM), RC2("RC2", REG), RC1("RC1", REG);
    A(k, i) >> DA >> SA.scope(j).out(iii) >> X;
    B(j, k) >> DB >> SB.scope(j).out(jjj) >> Y;
    C >> RC2.out(jjj, iii) >> RC1.out(jjj) >> DC;

    return DC.get_wrapper_func();
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [basic/SA/IO]\n";
        return 1;
    }
    std::string version(argv[1]);
    Buffer<int> A(N, M, "A"), B(S, N, "B");
    Func C;
    if (version == "basic") {
        C = mm_basic(A, B);
    } else if (version == "SA") {
        C = mm_SA(A, B);
    } else if (version == "IO") {
        C = mm_IO(A, B);
    } else {
        std::cerr << "Usage: " << argv[0] << " [basic/SA/IO]\n";
        return 1;
    }
    C.output_buffer().dim(0).set_bounds(0, S).set_stride(1);
    C.output_buffer().dim(1).set_bounds(0, M).set_stride(S);

    string device_file = "SCF_" + version + ".mlir";
    string ir_file = "SchedIR_" + version;
    Target target = get_host_target();
    target.set_feature(Target::MLIR);
    C.compile_to_device(device_file, {}, target);
    C.compile_to_lowered_stmt(ir_file, {}, Text, target);

    std::cout << "Generated file: " << device_file << ", " << ir_file << "\n";
    return 0;
}
