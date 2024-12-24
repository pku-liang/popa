#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <sstream>

#include "Lower.h"

#include "AddAtomicMutex.h"
#include "AddImageChecks.h"
#include "AddParameterChecks.h"
#include "AddSplitFactorChecks.h"
#include "AllocationBoundsInference.h"
#include "AsyncProducers.h"
#include "BoundConstantExtentLoops.h"
#include "BoundSmallAllocations.h"
#include "Bounds.h"
#include "BoundsInference.h"
#include "CSE.h"
#include "CanonicalizeGPUVars.h"
#include "ClampUnsafeAccesses.h"
#include "CompilerLogger.h"
#include "Debug.h"
#include "DebugArguments.h"
#include "DebugToFile.h"
#include "Deinterleave.h"
#include "EarlyFree.h"
#include "ExtractTileOperations.h"
#include "FindCalls.h"
#include "FindIntrinsics.h"
#include "FlattenNestedRamps.h"
#include "Func.h"
#include "Function.h"
#include "FuseGPUThreadLoops.h"
#include "FuzzFloatStores.h"
#include "HexagonOffload.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "InferArguments.h"
#include "InjectHostDevBufferCopies.h"
#include "Inline.h"
#include "LICM.h"
#include "LoopCarry.h"
#include "LowerParallelTasks.h"
#include "LowerWarpShuffles.h"
#include "Memoization.h"
#include "OffloadGPULoops.h"
#include "PartitionLoops.h"
#include "Prefetch.h"
#include "Profiling.h"
#include "PurifyIndexMath.h"
#include "Qualify.h"
#include "RealizationOrder.h"
#include "RebaseLoopsToZero.h"
#include "RemoveDeadAllocations.h"
#include "RemoveExternLoops.h"
#include "RemoveUndef.h"
#include "ScheduleFunctions.h"
#include "SelectGPUAPI.h"
#include "Simplify.h"
#include "SimplifyCorrelatedDifferences.h"
#include "SimplifySpecializations.h"
#include "SkipStages.h"
#include "SlidingWindow.h"
#include "SplitTuples.h"
#include "StageStridedLoads.h"
#include "StorageFlattening.h"
#include "StorageFolding.h"
#include "StrictifyFloat.h"
#include "StripAsserts.h"
#include "Substitute.h"
#include "Tracing.h"
#include "TrimNoOps.h"
#include "UnifyDuplicateLets.h"
#include "UniquifyVariableNames.h"
#include "UnpackBuffers.h"
#include "UnrollLoops.h"
#include "UnsafePromises.h"
#include "VectorizeLoops.h"
#include "WrapCalls.h"

// T2S related
#include "../../t2s/src/AddressableBuffer.h"
#include "../../t2s/src/AutorunKernels.h"
#include "../../t2s/src/ChannelPromotion.h"
#include "../../t2s/src/CheckRecursiveCalls.h"
#include "../../t2s/src/ComputeLoopBounds.h"
#include "../../t2s/src/CombineChannels.h"
#include "../../t2s/src/DebugPrint.h"
#include "../../t2s/src/Devectorize.h"
#include "../../t2s/src/FlattenLoops.h"
#include "../../t2s/src/Gather.h"
#include "../../t2s/src/LateFuse.h"
#include "../../t2s/src/LoopRemoval.h"
#include "../../t2s/src/MemorySchedule.h"
#include "../../t2s/src/MinimizeShregs.h"
#include "../../t2s/src/NoIfSimplify.h"
#include "../../t2s/src/Overlay.h"
#include "../../t2s/src/PatternMatcher.h"
#include "../../t2s/src/Place.h"
#include "../../t2s/src/Relay.h"
#include "../../t2s/src/RemoveDeadDimensions.h"
#include "../../t2s/src/ScatterAndBuffer.h"
#include "../../t2s/src/SpaceTimeTransform.h"
#include "../../t2s/src/StandardizeIRForOpenCL.h"
#include "../../t2s/src/TriangularLoopOptimize.h"

namespace Halide {
namespace Internal {

using std::ostringstream;
using std::string;
using std::vector;

namespace {

class LoweringLogger {
    Stmt last_written;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time;
    std::vector<std::pair<double, std::string>> timings;
    bool time_lowering_passes = false;

public:
    LoweringLogger() {
        last_time = std::chrono::high_resolution_clock::now();
        static bool should_time = !get_env_variable("HL_TIME_LOWERING_PASSES").empty();
        time_lowering_passes = should_time;
    }

    void operator()(const string &message, const Stmt &s) {
        auto t = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t - last_time;
        if (!s.same_as(last_written)) {
            debug(2) << message << "\n"
                     << s << "\n";
            last_written = s;
            last_time = t;
        } else {
            debug(2) << message << " (unchanged)\n\n";
            last_time = t;
        }
        timings.emplace_back(diff.count() * 1000, message);
    }

    ~LoweringLogger() {
        if (time_lowering_passes) {
            debug(0) << "Lowering pass runtimes:\n";
            std::sort(timings.begin(), timings.end());
            for (const auto &p : timings) {
                debug(0) << " " << p.first << " ms : " << p.second << "\n";
            }
        }
    }
};

void lower_impl(const vector<Function> &output_funcs,
                const string &pipeline_name,
                const Target &t,
                const vector<Argument> &args,
                const LinkageType linkage_type,
                const vector<Stmt> &requirements,
                bool trace_pipeline,
                const vector<IRMutator *> &custom_passes,
                Module &result_module) {
    auto time_start = std::chrono::high_resolution_clock::now();

    size_t initial_lowered_function_count = result_module.functions().size();

    // Create a deep-copy of the entire graph of Funcs.
    auto [outputs, env] = deep_copy(output_funcs, build_environment(output_funcs));

    bool any_strict_float = strictify_float(env, t);
    result_module.set_any_strict_float(any_strict_float);

    // Output functions should all be computed and stored at root.
    for (const Function &f : outputs) {
        Func(f).compute_root().store_root();
    }

    // Finalize all the LoopLevels
    for (auto &iter : env) {
        iter.second.lock_loop_levels();
    }

    // Substitute in wrapper Funcs
    env = wrap_func_calls(env);

    // Compute a realization order and determine group of functions which loops
    // are to be fused together
    auto [order, fused_groups] = realization_order(outputs, env);

    // Try to simplify the RHS/LHS of a function definition by propagating its
    // specializations' conditions
    simplify_specializations(env);

    LoweringLogger log;

    debug(1) << "Creating initial loop nests...\n";
    bool any_memoized = false;
    Stmt s = schedule_functions(outputs, fused_groups, env, t, any_memoized);
    log("Lowering after creating initial loop nests:", s);

    // Get the global min, max value of loop bounds
    LoopBounds global_bounds = compute_global_loop_bounds(s);

    debug(1) << "Applying space time transformation...\n";
    std::map<std::string, RegBound > reg_size_map;
    s = apply_space_time_transform(s, env, t, reg_size_map);
    log("Lowering after applying space time transformation:", s);

    debug(1) << "Fixing calls' args that correspond to loops marked as removed ...\n";
    s = fix_call_args_for_removed_loops(s, env);
    log("Lowering after fixing calls' args that correspond to loops marked as removed:", s);

    if (any_memoized) {
        debug(1) << "Injecting memoization...\n";
        s = inject_memoization(s, env, pipeline_name, outputs);
        log("Lowering after injecting memoization:", s);
    } else {
        debug(1) << "Skipping injecting memoization...\n";
    }
    debug(1) << "Injecting tracing...\n";
    s = inject_tracing(s, pipeline_name, trace_pipeline, env, outputs, t);
    log("Lowering after injecting tracing:", s);

    debug(1) << "Adding checks for recursice calls\n";
    check_recursive_calls(env);

    debug(1) << "Adding checks for parameters\n";
    s = add_parameter_checks(requirements, s, t);
    log("Lowering after injecting parameter checks:", s);

    // Compute the maximum and minimum possible value of each
    // function. Used in later bounds inference passes.
    debug(1) << "Computing bounds of each function's value\n";
    FuncValueBounds func_bounds = compute_function_value_bounds(order, env);

    // Clamp unsafe instances where a Func f accesses a Func g using
    // an index which depends on a third Func h.
    debug(1) << "Clamping unsafe data-dependent accesses\n";
    s = clamp_unsafe_accesses(s, env, func_bounds);
    log("Lowering after clamping unsafe data-dependent accesses", s);

    // This pass injects nested definitions of variable names, so we
    // can't simplify statements from here until we fix them up. (We
    // can still simplify Exprs).
    debug(1) << "Performing computation bounds inference...\n";
    s = bounds_inference(s, outputs, order, fused_groups, env, func_bounds, t);
    log("Lowering after computation bounds inference:", s);

    // This uniquifies the variable names, so we're good to simplify
    // after this point. This lets later passes assume syntactic
    // equivalence means semantic equivalence.
    debug(1) << "Uniquifying variable names...\n";
    s = uniquify_variable_names(s);
    log("Lowering after uniquifying variable names:", s);

    debug(1) << "Partitioning loops to simplify boundary conditions...\n";
    s = partition_loops(s);
    log("Lowering after partitioning loops :", s);

    debug(1) << "Simplifying IfThenElse but keeping unit loops...\n";
    s = no_if_simplify(s, true);
    log("Lowering after simplifying IfThenElse but keeping unit loops:\n", s);

    debug(1) << "Asserting that all split factors are positive...\n";
    s = add_split_factor_checks(s, env);
    log("Lowering after asserting that all split factors are positive:", s);

    debug(1) << "Removing extern loops...\n";
    s = remove_extern_loops(s);
    log("Lowering after removing extern loops:", s);

    debug(1) << "Performing sliding window optimization...\n";
    s = sliding_window(s, env);
    log("Lowering after sliding window:", s);

    debug(1) << "Simplifying correlated differences...\n";
    s = simplify_correlated_differences(s);
    log("Lowering after simplifying correlated differences:", s);

    debug(1) << "Performing allocation bounds inference...\n";
    s = allocation_bounds_inference(s, env, func_bounds);
    log("Lowering after allocation bounds inference:", s);

    bool will_inject_host_copies =
        (t.has_gpu_feature() ||
         t.has_feature(Target::HexagonDma) ||
         (t.arch != Target::Hexagon && (t.has_feature(Target::HVX))));

    debug(1) << "Adding checks for images\n";
    s = add_image_checks(s, outputs, t, order, env, func_bounds, will_inject_host_copies);
    log("Lowering after injecting image checks:", s);

    debug(1) << "Removing code that depends on undef values...\n";
    s = remove_undef(s);
    log("Lowering after removing code that depends on undef values:", s);

    if (t.has_fpga_feature()) {
        debug(1) << "Placing device functions...\n";
        s = place_device_functions(s, env, t);
        log("Lowering after placing device functions:", s);

        debug(1) << "Replacing references with channels and shift registers...\n";
        add_reference_names(s, env, reg_size_map);
        s = replace_references_with_channels(s, env, global_bounds);
        s = replace_references_with_shift_registers(s, env, reg_size_map);
        log("Lowering after replacing references with channels and shift registers:", s);
    }

    debug(1) << "Simplifying IfThenElse without keeping unit loops...\n";
    s = no_if_simplify(s, false);
    log("Lowering after simplifying IfThenElse without keeping unit loops:", s);

    if (t.has_fpga_feature()) {
        debug(1) << "Minimizing shift registers...\n";
        map<string, ShiftRegAlloc> func_to_regalloc;
        s = minimize_shift_registers(s, env, func_to_regalloc);
        log("Lowering after minimizing shift registers:", s);

        debug(1) << "Relaying...\n";
        s = relay_data(s, env, func_to_regalloc);
        log("Lowering after relaying:", s);
    }

    debug(1) << "Performing storage folding optimization...\n";
    s = storage_folding(s, env);
    log("Lowering after storage folding:", s);

    debug(1) << "Injecting debug_to_file calls...\n";
    s = debug_to_file(s, outputs, env);
    log("Lowering after injecting debug_to_file calls:", s);

    debug(1) << "Injecting prefetches...\n";
    s = inject_prefetch(s, env);
    log("Lowering after injecting prefetches:", s);

    debug(1) << "Discarding safe promises...\n";
    s = lower_safe_promises(s);
    log("Lowering after discarding safe promises:", s);

    debug(1) << "Dynamically skipping stages...\n";
    s = skip_stages(s, outputs, fused_groups, env);
    log("Lowering after dynamically skipping stages:", s);

    if (!t.has_fpga_feature()) {
        debug(1) << "Forking asynchronous producers...\n";
        s = fork_async_producers(s, env);
        log("Lowering after forking asynchronous producers:", s);
    } else {
        // We will generate code so that a producer communicates with a consumer in a channel.
    }

    debug(1) << "Destructuring tuple-valued realizations...\n";
    s = split_tuples(s, env);
    log("Lowering after destructuring tuple-valued realizations:", s);

    if (t.has_gpu_feature()) {
        debug(1) << "Canonicalizing GPU var names...\n";
        s = canonicalize_gpu_vars(s);
        log("Lowering after canonicalizing GPU var names:", s);
    }

    debug(1) << "Bounding small realizations...\n";
    s = simplify_correlated_differences(s);
    s = bound_small_allocations(s);
    log("Lowering after bounding small realizations:", s);

    debug(1) << "Performing storage flattening...\n";
    s = storage_flattening(s, outputs, env, t);
    log("Lowering after storage flattening:", s);

    if (t.has_feature(Target::IntelGPU)) {
        debug(1) << "Applying memory schedule...\n";
        s = do_memory_schedule(s, env);
        log("Lowering after memory schedule:\n", s);
    }

    debug(1) << "Adding atomic mutex allocation...\n";
    s = add_atomic_mutex(s, outputs);
    log("Lowering after adding atomic mutex allocation:", s);

    debug(1) << "Unpacking buffer arguments...\n";
    s = unpack_buffers(s);
    log("Lowering after unpacking buffer arguments:", s);

    if (any_memoized) {
        debug(1) << "Rewriting memoized allocations...\n";
        s = rewrite_memoized_allocations(s, env);
        log("Lowering after rewriting memoized allocations:", s);
    } else {
        debug(1) << "Skipping rewriting memoized allocations...\n";
    }

    map<string, Place> funcs_using_mem_channels;
    vector<std::pair<string, Expr>> letstmts_backup;
    if (t.has_fpga_feature()) {
        debug(1) << "Replacing references with mem channels...\n";
        s = replace_references_with_mem_channels(s, env, funcs_using_mem_channels, letstmts_backup);
        log("Lowering after replacing references with mem channels:", s);
    }

    if (will_inject_host_copies) {
        debug(1) << "Selecting a GPU API for GPU loops...\n";
        s = select_gpu_api(s, t);
        log("Lowering after selecting a GPU API:", s);

        debug(1) << "Injecting host <-> dev buffer copies...\n";
        s = inject_host_dev_buffer_copies(s, t, env, order);
        log("Lowering after injecting host <-> dev buffer copies:", s);

        debug(1) << "Selecting a GPU API for extern stages...\n";
        s = select_gpu_api(s, t);
        log("Lowering after selecting a GPU API for extern stages:", s);
    }

    debug(1) << "Second simplification...\n";
    s = simplify(s);
    s = unify_duplicate_lets(s);
    log("Lowering after second simplification:", s);

    debug(1) << "Reduce prefetch dimension...\n";
    s = reduce_prefetch_dimension(s, t);
    log("Lowering after reduce prefetch dimension:", s);

    debug(1) << "Simplifying correlated differences...\n";
    s = simplify_correlated_differences(s);
    log("Lowering after simplifying correlated differences:", s);

    if (t.has_fpga_feature()) {
        debug(1) << "Devectorize unsuitable loops...\n";
        s = devectorize(s);
        log("Lowering after devectorizing unsuitable loops:\n", s);
    }

    debug(1) << "Vectorizing...\n";
    s = vectorize_loops(s, t, env);
    log("Lowering after vectorizing:", s);
    s = simplify(s);
    log("Lowering after simplify after vectorizing:", s);

    debug(1) << "Combining channels ...\n";
    s = combine_channels(s);
    log("Lowering after combining channels:", s);

    debug(1) << "Remove Lets and LetStmts in funcs with buffering or scattering...\n";
    {
        std::set<string> funcs;
        for(auto entry : env){
            bool with_buffer = entry.second.definition().schedule().buffer_params().size() > 0;
            bool with_scatter = entry.second.definition().schedule().scatter_params().size() > 0;
            if (with_buffer || with_scatter) {
                funcs.insert(entry.first);
            }
        }
        s = simplify(remove_lets(s, true, true, true, false, funcs));
    }
    log("Lowering after removing Lets and LetStmts in funcs with buffering or scattering:", s);

    debug(1) << "Scattering and buffering...\n";
    s = simplify(scatter_buffer(s,env));
    log("Lowering after Scattering and buffering:", s);

    debug(1) << "Inserting addressable buffer...\n";
    s = simplify(insert_addressable_buffer(s, env));
    log("Lowering after inserting addressable buffer:", s);

    debug(1) << "Gathering...\n";
    s = simplify(gather_data(s, env));
    log("Lowering after Gathering:\n", s);

    debug(1) << "Bounding constant extent loops...\n";
    s = bound_constant_extent_loops(s);
    log("Lowering after bounding constant extent loops:", s);

    debug(1) << "Unrolling...\n";
    s = unroll_loops(s, env);
    log("Lowering after unrolling:", s);

    if (t.has_gpu_feature() ||
        t.has_feature(Target::Vulkan)) {
        debug(1) << "Injecting per-block gpu synchronization...\n";
        s = fuse_gpu_thread_loops(s);
        log("Lowering after injecting per-block gpu synchronization:", s);
    }

    // debug(1) << "Detecting vector interleavings...\n";
    // s = rewrite_interleavings(s);
    // s = simplify(s);
    // log("Lowering after rewriting vector interleavings:", s);

    debug(1) << "Partitioning loops to simplify boundary conditions...\n";
    s = partition_loops(s);
    s = simplify(s);
    log("Lowering after partitioning loops:", s);

    debug(1) << "Staging strided loads...\n";
    s = stage_strided_loads(s);
    log("Lowering after staging strided loads:", s);

    debug(1) << "Trimming loops to the region over which they do something...\n";
    s = trim_no_ops(s);
    log("Lowering after loop trimming:", s);

    debug(1) << "Rebasing loops to zero...\n";
    s = rebase_loops_to_zero(s);
    debug(2) << "Lowering after rebasing loops to zero:\n"
             << s << "\n\n";

    debug(1) << "Hoisting loop invariant if statements...\n";
    s = hoist_loop_invariant_if_statements(s);
    log("Lowering after hoisting loop invariant if statements:", s);

    if (!t.has_fpga_feature()) {
        debug(1) << "Injecting early frees...\n";
        s = inject_early_frees(s);
        log("Lowering after injecting early frees:", s);
    } else {
        // We issue kernels and immediately return without waiting for their completion.
        // So the host memory might still be needed to transfer data. Do not free.
    }

    if (t.has_feature(Target::FuzzFloatStores)) {
        debug(1) << "Fuzzing floating point stores...\n";
        s = fuzz_float_stores(s);
        log("Lowering after fuzzing floating point stores:", s);
    }

    debug(1) << "Simplifying correlated differences...\n";
    s = simplify_correlated_differences(s);
    log("Lowering after simplifying correlated differences:", s);

    if (t.has_fpga_feature()) {
        debug(1) << "Replace memory channel with references...\n";
        s = replace_mem_channels(s, env, letstmts_backup);
        log("Lowering after replacing memory channels:", s);
    }

    debug(1) << "Bounding small allocations...\n";
    s = bound_small_allocations(s);
    log("Lowering after bounding small allocations:", s);

    if (t.has_feature(Target::Profile) || t.has_feature(Target::ProfileByTimer)) {
        debug(1) << "Injecting profiling...\n";
        s = inject_profiling(s, pipeline_name, env);
        log("Lowering after injecting profiling:", s);
    }

    if (t.has_feature(Target::CUDA)) {
        debug(1) << "Injecting warp shuffles...\n";
        s = lower_warp_shuffles(s, t);
        log("Lowering after injecting warp shuffles:", s);
    }

    debug(1) << "Simplifying...\n";
    s = common_subexpression_elimination(s);

    debug(1) << "Matching compute patterns...\n";
    s = match_patterns(s);
    log("Lowering after matching patterns:", s);

    if (t.has_fpga_feature()) {
        debug(1) << "Inserting FPGA register calls\n";
        s = insert_fpga_reg(s, env);
        log("Lowering after inserting FPGA register calls:", s);
    }

    debug(1) << "Lowering unsafe promises...\n";
    s = lower_unsafe_promises(s, t);
    log("Lowering after lowering unsafe promises:", s);

    if (t.has_feature(Target::AVX512_SapphireRapids)) {
        debug(1) << "Extracting tile operations...\n";
        s = extract_tile_operations(s);
        log("Lowering after extracting tile operations:", s);
    }

    debug(1) << "Flattening nested ramps...\n";
    s = flatten_nested_ramps(s);
    log("Lowering after flattening nested ramps:", s);

    debug(1) << "Removing dead allocations and moving loop invariant code...\n";
    s = remove_dead_allocations(s);
    s = simplify(s);
    s = hoist_loop_invariant_values(s);
    s = hoist_loop_invariant_if_statements(s);
    log("Lowering after removing dead allocations and hoisting loop invariants:", s);

    debug(1) << "Finding intrinsics...\n";
    // Must be run after the last simplification, because it turns
    // divisions into shifts, which the simplifier reverses.
    s = find_intrinsics(s);
    log("Lowering after finding intrinsics:", s);

    debug(1) << "Hoisting prefetches...\n";
    s = hoist_prefetches(s);
    log("Lowering after hoisting prefetches:", s);

    if (t.has_feature(Target::NoAsserts)) {
        debug(1) << "Stripping asserts...\n";
        s = strip_asserts(s);
        log("Lowering after stripping asserts:", s);
    }

    debug(1) << "Late fuse...\n";
    s = do_late_fuse(s, env);
    log("Lowering after late fuse:\n", s);

    debug(1) << "Promoting channels...\n";
    s = channel_promotion(s);
    log("Lowering after channel promotion:", s);

    // For overlay, we don't need to flatten task loops.
    char *overlay_num = getenv("HL_OVERLAY_NUM");
    if (t.has_fpga_feature() && overlay_num == NULL) {
        debug(1) << "Flatten the loops...\n";
        s = simplify(flatten_loops(s, env));
        log("Lowering after loop flattening:", s);
    }

    debug(1) << "Flatten triangular loop...\n";
    s = flatten_tirangualr_loop_nest(s, env);
    log("Lowering after triangular loop optimizing:", s);

    if (getenv("DISABLE_AUTORUN") == NULL) {
        if (t.has_fpga_feature()) {
            debug(1) << "Making device funcs as autorun ...\n";
            s = autorun_kernels(s, env);
            log("Lowering after making device funcs as autorun:", s);
        }
    }

    debug(1) << "Creating overlay scheduler...\n";
    s = simplify(create_overlay_schedule(s, env));
    log("Lowering after creating overlay scheduler:\n", s);

    debug(1) << "Remove lets...\n";
    s = remove_lets(s, true, false, false, false, {});
    log("Lowering after removing lets:", s);

    // The code generator should blindly generate code according to the IR, without tricks if possible.
    // So here standardize the IR to make it have the same abstraction level as the target language to generate.
    // Although below it is done only for OpenCL and clear code gen only, ideally it should be done for any target
    // HW and language, and any code generator.
    if (t.features_any_of({Target::OpenCL}) && (getenv("CLEARCODE") != NULL)) {
        debug(1) << "Standardize IR for generating OpenCL code...\n";
        s = standardize_ir_for_opencl_code_gen(s);
        log("Lowering after standardizing IR for generating OpenCL code:", s);
    }

    if (!custom_passes.empty()) {
        for (size_t i = 0; i < custom_passes.size(); i++) {
            debug(1) << "Running custom lowering pass " << i << "...\n";
            s = custom_passes[i]->mutate(s);
            debug(1) << "Lowering after custom pass " << i << ":\n"
                     << s << "\n\n";
        }
    }

    // Make a copy of the Stmt code, before we lower anything to less human-readable code.
    result_module.set_conceptual_code_stmt(s);

    if (t.arch != Target::Hexagon && t.has_feature(Target::HVX)) {
        debug(1) << "Splitting off Hexagon offload...\n";
        s = inject_hexagon_rpc(s, t, result_module);
        debug(2) << "Lowering after splitting off Hexagon offload:\n"
                 << s << "\n";
    } else {
        debug(1) << "Skipping Hexagon offload...\n";
    }

    if (t.has_gpu_feature() || t.has_fpga_feature()) {
        debug(1) << "Offloading GPU loops...\n";
        s = inject_gpu_offload(s, t);
        debug(2) << "Lowering after splitting off GPU loops:\n"
                 << s << "\n\n";
    } else {
        debug(1) << "Skipping GPU offload...\n";
    }

    // TODO: This needs to happen before lowering parallel tasks, because global
    // images used inside parallel loops are rewritten from loads from images to
    // loads from closure parameters. Closure parameters are missing the Buffer<>
    // object, which needs to be found by infer_arguments here. Running
    // infer_arguments prior to lower_parallel_tasks is a hacky solution to this
    // problem. It would be better if closures could directly reference globals
    // so they don't add overhead to the closure.
    vector<InferredArgument> inferred_args = infer_arguments(s, outputs);

    std::vector<LoweredFunc> closure_implementations;
    debug(1) << "Lowering Parallel Tasks...\n";
    s = lower_parallel_tasks(s, closure_implementations, pipeline_name, t);
    // Process any LoweredFunctions added by other passes. In practice, this
    // will likely not work well enough due to ordering issues with
    // closure generating passes and instead all such passes will need to
    // be done at once.
    for (size_t i = initial_lowered_function_count; i < result_module.functions().size(); i++) {
        // Note that lower_parallel_tasks() appends to the end of closure_implementations
        result_module.functions()[i].body =
            lower_parallel_tasks(result_module.functions()[i].body, closure_implementations,
                                 result_module.functions()[i].name, t);
    }
    for (auto &lowered_func : closure_implementations) {
        result_module.append(lowered_func);
    }
    debug(2) << "Lowering after generating parallel tasks and closures:\n"
             << s << "\n\n";

    vector<Argument> public_args = args;
    for (const auto &out : outputs) {
        for (const Parameter &buf : out.output_buffers()) {
            public_args.emplace_back(buf.name(),
                                     Argument::OutputBuffer,
                                     buf.type(), buf.dimensions(), buf.get_argument_estimates());
        }
    }

    for (const InferredArgument &arg : inferred_args) {
        if (arg.param.defined() && arg.param.name() == "__user_context") {
            // The user context is always in the inferred args, but is
            // not required to be in the args list.
            continue;
        }

        internal_assert(arg.arg.is_input()) << "Expected only input Arguments here";

        bool found = false;
        for (const Argument &a : args) {
            found |= (a.name == arg.arg.name);
        }

        if (arg.buffer.defined() && !found) {
            // It's a raw Buffer used that isn't in the args
            // list. Embed it in the output instead.
            debug(1) << "Embedding image " << arg.buffer.name() << "\n";
            result_module.append(arg.buffer);
        } else if (!found) {
            std::ostringstream err;
            err << "Generated code refers to ";
            if (arg.arg.is_buffer()) {
                err << "image ";
            }
            err << "parameter " << arg.arg.name
                << ", which was not found in the argument list.\n";

            err << "\nArgument list specified: ";
            for (const auto &arg : args) {
                err << arg.name << " ";
            }
            err << "\n\nParameters referenced in generated code: ";
            for (const InferredArgument &ia : inferred_args) {
                if (ia.arg.name != "__user_context") {
                    err << ia.arg.name << " ";
                }
            }
            err << "\n\n";
            user_error << err.str();
        }
    }

    // We're about to drop the environment and outputs vector, which
    // contain the only strong refs to Functions that may still be
    // pointed to by the IR. So make those refs strong.
    class StrengthenRefs : public IRMutator {
        using IRMutator::visit;
        Expr visit(const Call *c) override {
            Expr expr = IRMutator::visit(c);
            c = expr.as<Call>();
            internal_assert(c);
            if (c->func.defined()) {
                FunctionPtr ptr = c->func;
                ptr.strengthen();
                expr = Call::make(c->type, c->name, c->args, c->call_type,
                                  ptr, c->value_index,
                                  c->image, c->param);
            }
            return expr;
        }
    };
    s = StrengthenRefs().mutate(s);

    LoweredFunc main_func(pipeline_name, public_args, s, linkage_type);

    // If we're in debug mode, add code that prints the args.
    if (t.has_feature(Target::Debug)) {
        debug_arguments(&main_func, t);
    }

    result_module.append(main_func);

    auto *logger = get_compiler_logger();
    if (logger) {
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;
        logger->record_compilation_time(CompilerLogger::Phase::HalideLowering, diff.count());
    }
}

}  // namespace

Module lower(const vector<Function> &output_funcs,
             const string &pipeline_name,
             const Target &t,
             const vector<Argument> &args,
             const LinkageType linkage_type,
             const vector<Stmt> &requirements,
             bool trace_pipeline,
             const vector<IRMutator *> &custom_passes) {
    Module result_module{strip_namespaces(pipeline_name), t};
    run_with_large_stack([&]() {
        lower_impl(output_funcs, pipeline_name, t, args, linkage_type, requirements, trace_pipeline, custom_passes, result_module);
    });
    return result_module;
}

Stmt lower_main_stmt(const std::vector<Function> &output_funcs,
                     const std::string &pipeline_name,
                     const Target &t,
                     const std::vector<Stmt> &requirements,
                     bool trace_pipeline,
                     const std::vector<IRMutator *> &custom_passes) {
    // We really ought to start applying for appellation d'origine contrôlée
    // status on types representing arguments in the Halide compiler.
    vector<InferredArgument> inferred_args = infer_arguments(Stmt(), output_funcs);
    vector<Argument> args;
    for (const auto &ia : inferred_args) {
        if (!ia.arg.name.empty() && ia.arg.is_input()) {
            args.push_back(ia.arg);
        }
    }

    Module module = lower(output_funcs, pipeline_name, t, args, LinkageType::External, requirements, trace_pipeline, custom_passes);

    return module.functions().front().body;
}

}  // namespace Internal
}  // namespace Halide
