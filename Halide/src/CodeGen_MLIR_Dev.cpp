#include <llvm/Support/raw_os_ostream.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "../../t2s/src/StandardizeIR.h"
#include "../../t2s/src/Utilities.h"
#include "CodeGen_GPU_Dev.h"
#include "CodeGen_MLIR_Dev.h"
#include "ExprUsesVar.h"
#include "IROperator.h"
#include "IRMutator.h"
#include "Module.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

namespace {

class CodeGen_MLIR_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_MLIR_Dev(const Target &t);

    /** Compile a GPU kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args) override;

    void gather_shift_regs_allocates(const Stmt &s);

    Stmt standardize_ir_for_fpga_offloading(const Stmt &s) override;

    /** (Re)initialize the GPU kernel module. This is separate from compile,
     * since a GPU device module will often have many kernels compiled into it
     * for a single pipeline. */
    void init_module() override;

    std::vector<char> compile_to_src() override;

    std::string get_current_kernel_name() override {
        return cur_kernel_name;
    }

    void dump() override {
        std::cerr << stream.str() << std::endl;
    }

    /** This routine returns the GPU API name that is combined into
     *  runtime routine names to ensure each GPU API has a unique
     *  name.
     */
    std::string api_unique_name() override {
        return "mlir";
    }

    /** Returns the specified name transformed by the variable naming rules
     * for the GPU language backend. Used to determine the name of a parameter
     * during host codegen. */
    std::string print_gpu_name(const std::string &name) override {
        return name;
    }

protected:
    void compile_func(mlir::ImplicitLocOpBuilder &builder, const LoweredFunc &func);

    static mlir::Type mlir_type_of(mlir::ImplicitLocOpBuilder &builder, Halide::Type t);

    class GatherShiftRegsAllocates : public IRVisitor {
        using IRVisitor::visit;
        std::map<std::string, int> space_loops;

    public:
        struct RegAlloc {
            Type type;
            std::vector<int> shapes;
            std::vector<int> factors;
            std::vector<int> space_dims;
        };
        std::map<std::string, RegAlloc> func_to_regalloc;

        void visit(const For *op) override;
        void visit(const Call *op) override;
        void visit(const Realize *op) override;
    };

    class MLIRBuilder : public IRVisitor {
    public:
        MLIRBuilder(mlir::ImplicitLocOpBuilder &builder,
                    const std::vector<DeviceArgument> &args,
                    const GatherShiftRegsAllocates &gather_reg_allocs);

    protected:
        mlir::Value codegen(const Expr &);
        mlir::Value index_codegen(const Expr &);
        mlir::AffineExpr affine_codegen(const Expr &);
        mlir::Value get_affine_index(const Expr &);
        void codegen(const Stmt &);

        void visit(const IntImm *) override;
        void visit(const UIntImm *) override;
        void visit(const FloatImm *) override;
        void visit(const StringImm *) override;
        void visit(const Cast *) override;
        void visit(const Reinterpret *) override;
        void visit(const Variable *) override;
        void visit(const Add *) override;
        void visit(const Sub *) override;
        void visit(const Mul *) override;
        void visit(const Div *) override;
        void visit(const Mod *) override;
        void visit(const Min *) override;
        void visit(const Max *) override;
        void visit(const EQ *) override;
        void visit(const NE *) override;
        void visit(const LT *) override;
        void visit(const LE *) override;
        void visit(const GT *) override;
        void visit(const GE *) override;
        void visit(const And *) override;
        void visit(const Or *) override;
        void visit(const Not *) override;
        void visit(const Select *) override;
        void visit(const Load *) override;
        void visit(const Ramp *) override;
        void visit(const Broadcast *) override;
        void visit(const Call *) override;
        void visit(const Let *) override;
        void visit(const LetStmt *) override;
        void visit(const AssertStmt *) override;
        void visit(const ProducerConsumer *) override;
        void visit(const For *) override;
        void visit(const Store *) override;
        void visit(const Provide *) override;
        void visit(const Allocate *) override;
        void visit(const Free *) override;
        void visit(const Realize *) override;
        void visit(const Block *) override;
        void visit(const IfThenElse *) override;
        void visit(const Evaluate *) override;
        void visit(const Shuffle *) override;
        void visit(const VectorReduce *) override;
        void visit(const Prefetch *) override;
        void visit(const Fork *) override;
        void visit(const Acquire *) override;
        void visit(const Atomic *) override;
        void visit(const HoistedStorage *) override;

        mlir::Type mlir_type_of(Halide::Type t) const;

        void sym_push(const std::string &name, mlir::Value value);
        void sym_pop(const std::string &name);
        mlir::Value sym_get(const std::string &name, bool must_succeed = true) const;

    private:
        mlir::ImplicitLocOpBuilder &builder;
        mlir::Value value;
        Scope<mlir::Value> symbol_table;
        // For those symbols added during processing
        std::vector<std::string> symbol_recorder;
        const GatherShiftRegsAllocates &gather_reg_allocs;
        bool need_index_type = false;
        bool generate_affine = false;
        std::map<std::string, int> var_to_affine_dims;
        mlir::AffineExpr affine_expr;
    };

    const Target &target;
    mlir::MLIRContext mlir_context;
    mlir::ModuleOp mlir_module;
    std::ostringstream stream;
    std::string cur_kernel_name;
    GatherShiftRegsAllocates gather_reg_allocs;
};

CodeGen_MLIR_Dev::CodeGen_MLIR_Dev(const Target &t)
    : target(t) {
    mlir_context.loadDialect<mlir::AffineDialect>();
    mlir_context.loadDialect<mlir::arith::ArithDialect>();
    mlir_context.loadDialect<mlir::func::FuncDialect>();
    mlir_context.loadDialect<mlir::memref::MemRefDialect>();
    mlir_context.loadDialect<mlir::scf::SCFDialect>();
    mlir_context.loadDialect<mlir::vector::VectorDialect>();
}

void CodeGen_MLIR_Dev::init_module() {
    mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
    mlir_module = mlir::ModuleOp::create(loc, llvm::StringRef("kernels"));
}

void CodeGen_MLIR_Dev::add_kernel(Stmt s,
                                  const std::string &name,
                                  const std::vector<DeviceArgument> &args) {
    debug(2) << "CodeGen_MLIR_Dev::compile " << name << "\n";

    cur_kernel_name = name;
    mlir::SmallVector<mlir::Type> inputs;
    mlir::SmallVector<mlir::Type> results;
    mlir::SmallVector<mlir::NamedAttribute> funcAttrs;
    mlir::SmallVector<mlir::DictionaryAttr> funcArgAttrs;

    auto builder = mlir::ImplicitLocOpBuilder::atBlockEnd(mlir_module.getLoc(), mlir_module.getBody());
    for (const auto &arg : args) {
        if (!arg.is_buffer) {
            inputs.push_back(mlir_type_of(builder, arg.type));
        }
    }
    mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
    mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(name),
                                                                       functionType, funcAttrs, funcArgAttrs);
    builder.setInsertionPointToStart(functionOp.addEntryBlock());

    CodeGen_MLIR_Dev::MLIRBuilder visitor(builder, args, gather_reg_allocs);
    s.accept(&visitor);
    builder.create<mlir::func::ReturnOp>();
}

std::vector<char> CodeGen_MLIR_Dev::compile_to_src() {
    mlir::PassManager pm(&mlir_context);
    pm.addPass(mlir::createCanonicalizerPass());
    internal_assert(mlir::succeeded(pm.run(mlir_module)));

    llvm::raw_os_ostream output(stream);
    mlir_module.print(output);
    output.flush();

    std::string str = stream.str();
    std::vector<char> buffer(str.begin(), str.end());
    buffer.push_back(0);
    return buffer;
}

mlir::Type CodeGen_MLIR_Dev::mlir_type_of(mlir::ImplicitLocOpBuilder &builder, Halide::Type t) {
    if (t.lanes() == 1) {
        if (t.is_int_or_uint()) {
            return builder.getIntegerType(t.bits());
        } else if (t.is_bfloat()) {
            return builder.getBF16Type();
        } else if (t.is_float()) {
            switch (t.bits()) {
            case 16:
                return builder.getF16Type();
            case 32:
                return builder.getF32Type();
            case 64:
                return builder.getF64Type();
            default:
                internal_error << "There is no MLIR type matching this floating-point bit width: " << t << "\n";
                return nullptr;
            }
        } else {
            internal_error << "Type not supported: " << t << "\n";
        }
    } else {
        return mlir::VectorType::get(t.lanes(), mlir_type_of(builder, t.element_of()));
    }

    return mlir::Type();
}

CodeGen_MLIR_Dev::MLIRBuilder::MLIRBuilder(mlir::ImplicitLocOpBuilder &builder,
                                           const std::vector<DeviceArgument> &args,
                                           const GatherShiftRegsAllocates &ga)
    : builder(builder), gather_reg_allocs(ga) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());
    for (auto [index, arg] : llvm::enumerate(args)) {
        if (arg.is_buffer) {
            std::string func = arg.name + ".buffer";
            const auto &regalloc = ga.func_to_regalloc.at(func);
            mlir::SmallVector<int64_t> shapes(regalloc.shapes.begin(), regalloc.shapes.end());
            mlir::MemRefType type = mlir::MemRefType::get(shapes, mlir_type_of(arg.type));
            mlir::memref::AllocOp alloc = builder.create<mlir::memref::AllocOp>(type);

            mlir::SmallVector<mlir::Attribute> cyclic, dims, factors;
            for (size_t i = 0; i < regalloc.space_dims.size(); i++) {
                cyclic.push_back(builder.getIntegerAttr(builder.getI32Type(), 1));
                dims.push_back(builder.getIntegerAttr(builder.getI32Type(), regalloc.space_dims[i]));
                factors.push_back(builder.getIntegerAttr(builder.getI32Type(), regalloc.factors[i]));
            }
            alloc->setAttr("var_name", builder.getStringAttr(func));
            if (!cyclic.empty()) {
                alloc->setAttr("partition_cyclic_array", builder.getArrayAttr(cyclic));
                alloc->setAttr("partition_dim_array", builder.getArrayAttr(dims));
                alloc->setAttr("partition_factor_array", builder.getArrayAttr(factors));
            }
            sym_push(func, alloc);
        } else {
            sym_push(arg.name, funcOp.getArgument(index));
        }
    }
}

mlir::AffineExpr CodeGen_MLIR_Dev::MLIRBuilder::affine_codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    affine_expr = mlir::AffineExpr();
    generate_affine = true;
    e.accept(this);
    generate_affine = false;
    return affine_expr;
}

mlir::Value CodeGen_MLIR_Dev::MLIRBuilder::index_codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    need_index_type = true;
    e.accept(this);
    need_index_type = false;
    return value;
}

mlir::Value CodeGen_MLIR_Dev::MLIRBuilder::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    return value;
}

void CodeGen_MLIR_Dev::MLIRBuilder::codegen(const Stmt &s) {
    internal_assert(s.defined());
    debug(4) << "Codegen (S): " << s << "\n";
    value = mlir::Value();
    s.accept(this);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const IntImm *op) {
    if (generate_affine) {
        affine_expr = mlir::getAffineConstantExpr(op->value, builder.getContext());
        return;
    }
    std::string symbol_name = "c" + std::to_string(op->value) + (need_index_type ? "" : "_i32");
    if (!(value = sym_get(symbol_name, false))) {
        mlir::Type type = mlir_type_of(op->type);
        value = need_index_type ? builder.create<mlir::arith::ConstantIndexOp>(op->value)
                                : builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
        sym_push(symbol_name, value);
        symbol_recorder.push_back(symbol_name);
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const UIntImm *op) {
    if (generate_affine) {
        affine_expr = mlir::getAffineConstantExpr(op->value, builder.getContext());
        return;
    }
    std::string symbol_name = "c" + std::to_string(op->value) + (need_index_type ? "" : "_i32");
    if (!(value = sym_get(symbol_name, false))) {
        mlir::Type type = mlir_type_of(op->type);
        value = need_index_type ? builder.create<mlir::arith::ConstantIndexOp>(op->value)
                                : builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
        sym_push(symbol_name, value);
        symbol_recorder.push_back(symbol_name);
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const FloatImm *op) {
    mlir::Type type = mlir_type_of(op->type);
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getFloatAttr(type, op->value));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const StringImm *op) {
    internal_error << "String immediates are not supported\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Cast *op) {
    Halide::Type src = op->value.type();
    Halide::Type dst = op->type;
    mlir::Type mlir_type = mlir_type_of(dst);

    value = codegen(op->value);

    if (src.is_int_or_uint() && dst.is_int_or_uint()) {
        if (dst.bits() > src.bits()) {
            if (src.is_int())
                value = builder.create<mlir::arith::ExtSIOp>(mlir_type, value);
            else
                value = builder.create<mlir::arith::ExtUIOp>(mlir_type, value);
        } else {
            value = builder.create<mlir::arith::TruncIOp>(mlir_type, value);
        }
    } else if (src.is_float() && dst.is_int()) {
        value = builder.create<mlir::arith::FPToSIOp>(mlir_type, value);
    } else if (src.is_float() && dst.is_uint()) {
        value = builder.create<mlir::arith::FPToUIOp>(mlir_type, value);
    } else if (src.is_int() && dst.is_float()) {
        value = builder.create<mlir::arith::SIToFPOp>(mlir_type, value);
    } else if (src.is_uint() && dst.is_float()) {
        value = builder.create<mlir::arith::UIToFPOp>(mlir_type, value);
    } else if (src.is_float() && dst.is_float()) {
        if (dst.bits() > src.bits()) {
            value = builder.create<mlir::arith::ExtFOp>(mlir_type, value);
        } else {
            value = builder.create<mlir::arith::TruncFOp>(mlir_type, value);
        }
    } else {
        internal_error << "Cast of " << src << " to " << dst << " is not implemented\n";
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Reinterpret *op) {
    value = builder.create<mlir::arith::BitcastOp>(mlir_type_of(op->type), codegen(op->value));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Variable *op) {
    if (generate_affine) {
        if (var_to_affine_dims.find(op->name) == var_to_affine_dims.end()) {
            var_to_affine_dims[op->name] = var_to_affine_dims.size();
        }
        affine_expr = mlir::getAffineDimExpr(var_to_affine_dims[op->name], builder.getContext());
        return;
    }
    value = sym_get(op->name, true);
    if (!need_index_type && value.getType().isa<mlir::IndexType>()) {
        value = builder.create<mlir::arith::IndexCastOp>(builder.getIntegerType(32), value);
    }
    if (need_index_type && !value.getType().isa<mlir::IndexType>()) {
        value = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), value);
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Add *op) {
    if (generate_affine) {
        affine_expr = affine_codegen(op->a) + affine_codegen(op->b);
        return;
    }
    if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::AddIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::AddFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Sub *op) {
    if (generate_affine) {
        affine_expr = affine_codegen(op->a) - affine_codegen(op->b);
        return;
    }
    if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::SubIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::SubFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Mul *op) {
    if (generate_affine) {
        affine_expr = affine_codegen(op->a) * affine_codegen(op->b);
        return;
    }
    if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::MulIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MulFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Div *op) {
    internal_assert(!generate_affine);
    if (op->type.is_int())
        value = builder.create<mlir::arith::DivSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::DivUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::DivFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Mod *op) {
    internal_assert(!generate_affine);
    if (op->type.is_int())
        value = builder.create<mlir::arith::RemSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::RemUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::RemFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Min *op) {
    if (op->type.is_int())
        value = builder.create<mlir::arith::MinSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::MinUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MinFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Max *op) {
    if (op->type.is_int())
        value = builder.create<mlir::arith::MaxSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::MaxUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MaxFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const EQ *op) {
    if (op->a.type().is_int_or_uint())
        value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, codegen(op->a), codegen(op->b));
    else if (op->a.type().is_float())
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::OEQ, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const NE *op) {
    if (op->a.type().is_int_or_uint())
        value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, codegen(op->a), codegen(op->b));
    else if (op->a.type().is_float())
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::ONE, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const LT *op) {
    if (op->a.type().is_int_or_uint()) {
        mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::slt :
                                                                   mlir::arith::CmpIPredicate::ult;
        value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
    } else if (op->a.type().is_float()) {
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::OLT, codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const LE *op) {
    if (op->a.type().is_int_or_uint()) {
        mlir::arith::CmpIPredicate predicate = op->a.type().is_int() ? mlir::arith::CmpIPredicate::sle :
                                                                       mlir::arith::CmpIPredicate::ule;
        value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
    } else if (op->a.type().is_float()) {
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::OLE, codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const GT *op) {
    if (op->a.type().is_int_or_uint()) {
        mlir::arith::CmpIPredicate predicate = op->a.type().is_int() ? mlir::arith::CmpIPredicate::sgt :
                                                                       mlir::arith::CmpIPredicate::ugt;
        value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
    } else if (op->a.type().is_float()) {
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::OGT, codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const GE *op) {
    if (op->a.type().is_int_or_uint()) {
        mlir::arith::CmpIPredicate predicate = op->a.type().is_int() ? mlir::arith::CmpIPredicate::sge :
                                                                       mlir::arith::CmpIPredicate::uge;
        value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
    } else if (op->a.type().is_float()) {
        value = builder.create<mlir::arith::CmpFOp>(mlir::arith::CmpFPredicate::OGE, codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const And *op) {
    // value = builder.create<mlir::arith::AndIOp>(codegen(NE::make(op->a, make_zero(op->a.type()))),
    //                                             codegen(NE::make(op->b, make_zero(op->b.type()))));
    value = builder.create<mlir::arith::AndIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Or *op) {
    // value = builder.create<mlir::arith::OrIOp>(codegen(NE::make(op->a, make_zero(op->a.type()))),
    //                                            codegen(NE::make(op->b, make_zero(op->b.type()))));
    value = builder.create<mlir::arith::OrIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Not *op) {
    value = codegen(EQ::make(op->a, make_zero(op->a.type())));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Select *op) {
    value = builder.create<mlir::arith::SelectOp>(codegen(op->condition),
                                                  codegen(op->true_value),
                                                  codegen(op->false_value));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Load *op) {
    mlir::Value buffer = sym_get(op->name + ".buffer");
    mlir::Type type = mlir_type_of(op->type);
    mlir::Value index;
    if (op->type.is_scalar()) {
        index = index_codegen(op->index);
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = index_codegen(ramp_base);
    } else {
        internal_error << "Unsupported load\n";
    }

    // index = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), index);
    if (op->type.is_scalar()) {
        value = builder.create<mlir::memref::LoadOp>(type, buffer, mlir::ValueRange{index});
    } else {
        value = builder.create<mlir::vector::LoadOp>(type, buffer, mlir::ValueRange{index});
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Ramp *op) {
    mlir::Value base = codegen(op->base);
    mlir::Value stride = codegen(op->stride);
    mlir::Type elementType = mlir_type_of(op->base.type());
    mlir::VectorType vectorType = mlir::VectorType::get(op->lanes, elementType);

    mlir::SmallVector<mlir::Attribute> indicesAttrs(op->lanes);
    for (int i = 0; i < op->lanes; i++)
        indicesAttrs[i] = mlir::IntegerAttr::get(elementType, i);

    mlir::DenseElementsAttr indicesDenseAttr = mlir::DenseElementsAttr::get(vectorType, indicesAttrs);
    mlir::Value indicesConst = builder.create<mlir::arith::ConstantOp>(indicesDenseAttr);
    mlir::Value splatStride = builder.create<mlir::vector::SplatOp>(vectorType, stride);
    mlir::Value offsets = builder.create<mlir::arith::MulIOp>(splatStride, indicesConst);
    mlir::Value splatBase = builder.create<mlir::vector::SplatOp>(vectorType, base);
    value = builder.create<mlir::arith::AddIOp>(splatBase, offsets);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Broadcast *op) {
    value = builder.create<mlir::vector::SplatOp>(mlir_type_of(op->type), codegen(op->value));
}

mlir::Value CodeGen_MLIR_Dev::MLIRBuilder::get_affine_index(const Expr &e) {
    var_to_affine_dims.clear();
    mlir::AffineExpr affine_expr = affine_codegen(e);
    auto num_dims = var_to_affine_dims.size();
    auto map = mlir::AffineMap::get(num_dims, 0, affine_expr, builder.getContext());
    mlir::SmallVector<mlir::Value> vars(num_dims);
    for (auto kv : var_to_affine_dims) {
        vars[kv.second] = sym_get(kv.first);
    }
    return builder.create<mlir::AffineApplyOp>(builder.getUnknownLoc(), map, vars);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Call *op) {
    if (op->is_intrinsic(Call::bitwise_and)) {
        value = builder.create<mlir::arith::AndIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::shift_left)) {
        value = builder.create<mlir::arith::ShLIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::shift_right)) {
        if (op->type.is_int())
            value = builder.create<mlir::arith::ShRSIOp>(codegen(op->args[0]), codegen(op->args[1]));
        else
            value = builder.create<mlir::arith::ShRUIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::widen_right_mul)) {
        mlir::Value a = codegen(op->args[0]);
        mlir::Value b = codegen(op->args[1]);
        if (op->type.is_int())
            b = builder.create<mlir::arith::ExtSIOp>(a.getType(), b);
        else
            b = builder.create<mlir::arith::ExtUIOp>(a.getType(), b);
        value = builder.create<mlir::arith::MulIOp>(a, b);
    } else if (op->name == Call::buffer_get_host) {
        value = codegen(op->args[0]);
    } else if (op->name == Call::buffer_get_min) {
        mlir::Type type = mlir_type_of(op->type);
        value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, 0));
    } else if (op->name == Call::buffer_get_extent) {
        mlir::Type type = mlir_type_of(op->type);
        mlir::Value buffer = codegen(op->args[0]);
        mlir::Value index = codegen(op->args[1]);
        index = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), index);
        mlir::Value dim = builder.create<mlir::memref::DimOp>(buffer, index);
        value = builder.create<mlir::arith::IndexCastOp>(type, dim);
    } else if (op->is_intrinsic(Call::read_shift_reg)) {
        auto name = op->args[0].as<StringImm>();
        internal_assert(name);
        mlir::Value buffer = sym_get(name->value);
        mlir::SmallVector<mlir::Value> args;
        for (size_t i = 1; i < op->args.size(); i++) {
            args.push_back(get_affine_index(op->args[i]));
        }
        value = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), buffer, args);
    } else if (op->is_intrinsic(Call::write_shift_reg)) {
        auto name = op->args[0].as<StringImm>();
        internal_assert(name);
        mlir::Value buffer = sym_get(name->value);
        mlir::SmallVector<mlir::Value> args;
        for (size_t i = 1; i < op->args.size()-1; i++) {
            args.push_back(get_affine_index(op->args[i]));
        }
        mlir::Value value = codegen(op->args.back());
        builder.create<mlir::AffineStoreOp>(value, buffer, args);
    } else if (op->is_intrinsic(Call::image_load)) {
        auto name = op->args[0].as<StringImm>();
        internal_assert(name);
        mlir::Value buffer = sym_get(name->value + ".buffer");
        mlir::SmallVector<mlir::Value> args;
        for (size_t i = 2; i < op->args.size(); i += 2) {
            args.push_back(get_affine_index(op->args[i]));
        }
        value = builder.create<mlir::AffineLoadOp>(builder.getUnknownLoc(), buffer, args);
    } else if (op->is_intrinsic(Call::image_store)) {
        auto name = op->args[0].as<StringImm>();
        internal_assert(name);
        mlir::Value buffer = sym_get(name->value + ".buffer");
        mlir::SmallVector<mlir::Value> args;
        for (size_t i = 2; i < op->args.size()-1; i += 2) {
            args.push_back(get_affine_index(op->args[i]));
        }
        mlir::Value value = codegen(op->args.back());
        builder.create<mlir::AffineStoreOp>(value, buffer, args);
    } else {
        internal_error << "Call to " << op->name << " not implemented\n";
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Let *op) {
    sym_push(op->name, codegen(op->value));
    value = codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const LetStmt *op) {
    sym_push(op->name, codegen(op->value));
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const AssertStmt *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const ProducerConsumer *op) {
    codegen(op->body);
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const For *op) {
    if (ends_with(op->name, ".run_on_device")) {
        std::string func_name = extract_first_token(op->name);
        const auto &func_to_regalloc = gather_reg_allocs.func_to_regalloc;
        if (func_to_regalloc.find(func_name) != func_to_regalloc.end()) {
            const auto &regalloc = func_to_regalloc.at(func_name);
            mlir::SmallVector<int64_t> shapes(regalloc.shapes.begin(), regalloc.shapes.end());
            mlir::MemRefType type = mlir::MemRefType::get(shapes, mlir_type_of(regalloc.type));
            mlir::memref::AllocOp alloc = builder.create<mlir::memref::AllocOp>(type);

            mlir::SmallVector<mlir::Attribute> cyclic, dims, factors;
            for (size_t i = 0; i < regalloc.space_dims.size(); i++) {
                cyclic.push_back(builder.getIntegerAttr(builder.getI32Type(), 1));
                dims.push_back(builder.getIntegerAttr(builder.getI32Type(), regalloc.space_dims[i]));
                factors.push_back(builder.getIntegerAttr(builder.getI32Type(), regalloc.factors[i]));
            }
            std::string var_name = func_name + ".shreg";
            alloc->setAttr("var_name", builder.getStringAttr(var_name));
            if (!cyclic.empty()) {
                alloc->setAttr("partition_cyclic_array", builder.getArrayAttr(cyclic));
                alloc->setAttr("partition_dim_array", builder.getArrayAttr(dims));
                alloc->setAttr("partition_factor_array", builder.getArrayAttr(factors));
            }
            sym_push(var_name, alloc);
            codegen(op->body);
            sym_pop(var_name);
        } else {
            codegen(op->body);
        }
        return;
    }
    mlir::Value i;
    mlir::Block *body;
    if (is_const(op->min) && is_const(op->extent)) {
        auto lb = *as_const_int(op->min);
        auto ub = lb + *as_const_int(op->extent);
        mlir::AffineForOp forOp = builder.create<mlir::AffineForOp>(lb, ub, 1);
        i = forOp.getInductionVar();
        body = forOp.getBody();

        if (op->for_type == ForType::Pipelined) {
            forOp->setAttr("pipeline", builder.getIntegerAttr(builder.getIntegerType(32), 1));
        }
        if (op->for_type == ForType::Unrolled) {
            forOp->setAttr("unroll", builder.getIntegerAttr(builder.getIntegerType(32), 0));
        }
    } else {
        mlir::Value lb = index_codegen(op->min);
        mlir::Value ub = index_codegen(simplify(op->min + op->extent));
        mlir::Value step = index_codegen(1);
        mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(lb, ub, step);
        i = forOp.getInductionVar();
        body = forOp.getBody();
    }
    int prev_syms = symbol_recorder.size();
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body);
        sym_push(op->name, i);
        codegen(op->body);
        sym_pop(op->name);
        for (int i = symbol_recorder.size()-1; i >= prev_syms; i--) {
            sym_pop(symbol_recorder.back());
            symbol_recorder.pop_back();
        }
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Store *op) {
    mlir::Value buffer = sym_get(op->name + ".buffer");
    mlir::Value value = codegen(op->value);
    mlir::Value index;
    if (op->value.type().is_scalar()) {
        index = index_codegen(op->index);
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = index_codegen(ramp_base);
    } else {
        internal_error << "Unsupported store\n";
    }

    // index = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), index);
    if (op->value.type().is_scalar()) {
        builder.create<mlir::memref::StoreOp>(value, buffer, mlir::ValueRange{index});
    } else {
        builder.create<mlir::vector::StoreOp>(value, buffer, mlir::ValueRange{index});
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Provide *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Allocate *op) {
    int32_t size = op->constant_allocation_size();
    mlir::MemRefType type = mlir::MemRefType::get({size}, mlir_type_of(op->type));
    mlir::memref::AllocOp alloc = builder.create<mlir::memref::AllocOp>(type);

    sym_push(op->name + ".buffer", alloc);
    codegen(op->body);
    sym_pop(op->name + ".buffer");
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Free *op) {
    builder.create<mlir::memref::DeallocOp>(sym_get(op->name));
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Realize *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Block *op) {
    // Peel blocks of assertions with pure conditions
    const AssertStmt *a = op->first.as<AssertStmt>();
    if (a && is_pure(a->condition)) {
        std::vector<const AssertStmt *> asserts;
        asserts.push_back(a);
        Stmt s = op->rest;
        while ((op = s.as<Block>()) && (a = op->first.as<AssertStmt>()) && is_pure(a->condition) && asserts.size() < 63) {
            asserts.push_back(a);
            s = op->rest;
        }
        // TODO
        // codegen_asserts(asserts);
        codegen(s);
    } else {
        codegen(op->first);
        codegen(op->rest);
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const IfThenElse *op) {
    auto then_builder = [&](mlir::OpBuilder &b, mlir::Location l) {
        codegen(op->then_case);
        b.create<mlir::scf::YieldOp>(l);
    };
    auto else_builder = [&](mlir::OpBuilder &b, mlir::Location l) {
        codegen(op->else_case);
        b.create<mlir::scf::YieldOp>(l);
    };
    if (!op->else_case.defined()) {
        builder.create<mlir::scf::IfOp>(codegen(op->condition), then_builder);
    } else {
        builder.create<mlir::scf::IfOp>(codegen(op->condition), then_builder, else_builder);
    }
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Evaluate *op) {
    codegen(op->value);
    // Discard result
    value = mlir::Value();
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Shuffle *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const VectorReduce *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Prefetch *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Fork *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Acquire *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const Atomic *op) {
    internal_error << "Unimplemented\n";
}

void CodeGen_MLIR_Dev::MLIRBuilder::visit(const HoistedStorage *op) {
    internal_error << "Unimplemented\n";
}

mlir::Type CodeGen_MLIR_Dev::MLIRBuilder::mlir_type_of(Halide::Type t) const {
    return CodeGen_MLIR_Dev::mlir_type_of(builder, t);
}

void CodeGen_MLIR_Dev::MLIRBuilder::sym_push(const std::string &name, mlir::Value value) {
    symbol_table.push(name, value);
}

void CodeGen_MLIR_Dev::MLIRBuilder::sym_pop(const std::string &name) {
    symbol_table.pop(name);
}

mlir::Value CodeGen_MLIR_Dev::MLIRBuilder::sym_get(const std::string &name, bool must_succeed) const {
    // look in the symbol table
    if (!symbol_table.contains(name)) {
        if (must_succeed) {
            std::ostringstream err;
            err << "Symbol not found: " << name << "\n";

            if (debug::debug_level() > 0) {
                err << "The following names are in scope:\n"
                    << symbol_table << "\n";
            }

            internal_error << err.str();
        } else {
            return nullptr;
        }
    }
    return symbol_table.get(name);
}

void CodeGen_MLIR_Dev::GatherShiftRegsAllocates::visit(const Realize *op) {
    if (ends_with(op->name, ".shreg")) {
        std::string func = remove_postfix(op->name, ".shreg");
        internal_assert(op->types.size() == 1);
        func_to_regalloc[func].type = op->types[0];
        auto &shapes = func_to_regalloc[func].shapes;
        for (auto b : op->bounds) {
            internal_assert(is_const(b.extent));
            shapes.push_back(*as_const_int(b.extent));
        }
    }
    op->body.accept(this);
}

void CodeGen_MLIR_Dev::GatherShiftRegsAllocates::visit(const For *op) {
    if (op->for_type == ForType::Unrolled) {
        user_assert(is_const(op->extent))
            << "Unrolled loop " << op->name << " must have a constant bound.\n";
        space_loops[op->name] = *as_const_int(op->extent);
    }
    op->body.accept(this);
}

void CodeGen_MLIR_Dev::GatherShiftRegsAllocates::visit(const Call *op) {
    for (size_t i = 0; i < op->args.size(); i++) {
        op->args[i].accept(this);
    }
    if (op->is_intrinsic(Call::write_shift_reg)) {
        internal_assert(op->args[0].as<StringImm>());
        std::string var_name = op->args[0].as<StringImm>()->value;
        std::string func = remove_postfix(var_name, ".shreg");
        // This alloc is collected when visiting Realize node
        internal_assert(func_to_regalloc.find(func) != func_to_regalloc.end());
        auto &alloc = func_to_regalloc[func];
        for (size_t i = 1; i < op->args.size()-1; i++) {
            auto var = op->args[i].as<Variable>();
            if (var && space_loops.find(var->name) != space_loops.end()) {
                alloc.space_dims.push_back(i-1);
                alloc.factors.push_back(space_loops[var->name]);
            }
        }
    }
    if (op->is_intrinsic(Call::image_load) || op->is_intrinsic(Call::image_store)) {
        internal_assert(op->args[0].as<StringImm>());
        std::string func = op->args[0].as<StringImm>()->value + ".buffer";
        // Load from or store into an external buffer only once
        internal_assert(func_to_regalloc.find(func) == func_to_regalloc.end());
        auto &alloc = func_to_regalloc[func];
        alloc.type = op->type;
        int num_dims = (op->args.size() - 2) / 2;
        for (int i = 0; i < num_dims; i++) {
            internal_assert(is_const(op->args[i*2 + 3]));
            alloc.shapes.push_back(*as_const_int(op->args[i*2 + 3]));
            // If any space loop appears in this dimension, it must be partitioned
            auto space_it = std::find_if(space_loops.begin(), space_loops.end(),
                                        [&](const auto &kv){ return expr_uses_var(op->args[i*2 + 2], kv.first); });
            if (space_it != space_loops.end()) {
                alloc.space_dims.push_back(i);
                alloc.factors.push_back(space_it->second);
            }
        }
    }
}

Stmt CodeGen_MLIR_Dev::standardize_ir_for_fpga_offloading(const Stmt &s) {
    s.accept(&gather_reg_allocs);
    Stmt result = RemoveDeviceDeclaration().mutate(s);
    result = RemoveIfStmt().mutate(s);
    return simplify(result);
}

}  // namespace

std::unique_ptr<CodeGen_GPU_Dev> new_CodeGen_MLIR_Dev(const Target &target) {
    return std::make_unique<CodeGen_MLIR_Dev>(target);
}

}  // namespace Internal
}  // namespace Halide
