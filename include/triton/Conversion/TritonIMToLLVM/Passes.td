#ifndef TRITONIM_CONVERSION_PASSES
#define TRITONIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// Conversion pass: lowers TritonGPU IR → LLVM IR for In-Memory (IM)
// processing targets such as HBM-PIM.  No GPU-specific constructs
// (shared memory, warps, barriers) are emitted; the resulting LLVM IR
// is suitable for instrumentation (e.g. MemTracePass) and PIM
// simulation in Ramulator 2.
def ConvertTritonIMToLLVM
    : Pass<"convert-triton-im-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonGPU IR to LLVM for In-Memory processing";
  let description = [{
    Lowers Triton GPU dialect operations to the LLVM dialect targeting
    In-Memory processing architectures such as HBM-PIM.  The pass
    reuses the shared TritonGPUToLLVM lowering patterns (elementwise,
    memory, control-flow, SPMD, etc.) with an IM-specific TargetInfo
    that models bank-level SIMD parallelism: each PIM bank is one
    Triton "thread" (threads_per_warp = numBanks).  The bank index is
    obtained at runtime via the extern function __pim_get_bank_id().
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect", "mlir::scf::SCFDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Layout rewrite pass: changes the BlockedEncodingAttr so that each PIM
// bank owns a *consecutive* chunk of elements (register-innermost
// layout) instead of the default GPU-style interleaved layout
// (lane-innermost / coalesced).
//
// GPU default (sizePerThread=1):
//   bank 0 → {0, 16, 32, 48}  (stride = threadsPerWarp)
// IM layout (sizePerThread = shape / threadsPerWarp):
//   bank 0 → {0, 1, 2, 3}     (consecutive)
//
// This pass replaces tritongpu-coalesce in the IM pipeline.
def RewriteIMLayout : Pass<"rewrite-im-layout", "mlir::ModuleOp"> {
  let summary = "Rewrite tensor layout for bank-consecutive PIM access";
  let description = [{
    Rewrites BlockedEncodingAttr on tensor-typed memory operations so
    that each PIM bank (= one TritonGPU "lane") owns a consecutive
    slice of elements rather than an interleaved slice.  The pass
    reads im.num-banks and ttg.num-warps from the module and computes
    sizePerThread = tensorShape / (threadsPerWarp × warpsPerCTA) for
    each dimension.  ConvertLayoutOps are inserted around load/store
    operations; the downstream tritongpu-remove-layout-conversions
    pass propagates the new encoding through the rest of the IR.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Operand-residency layout pass: the explicit, compiler-side analog of
// OptiPIM's DataLayout/DetailLayout MILP (heuristic, not a solver).  Decides a
// per-operand physical layout + residency so that operand reuse is a property
// of the chosen layout (operand laid out resident / replicated across the
// spatial banks; reduction axis on the column-low bits) rather than something
// recovered post-hoc by trace dedup.  Run over fused regions it can co-optimize
// layout across operators — a capability per-operator MILP compilers lack.
// See docs/reuse-as-layout-plan.md.
def IMOperandResidencyLayout
    : Pass<"im-operand-residency-layout", "mlir::ModuleOp"> {
  let summary = "Assign per-operand physical layout/residency for PIM reuse";
  let description = [{
    Classifies each tensor operand's reuse axes and assigns a physical layout
    (bank/bankgroup placement, reduction-axis column locality, residency,
    per-BG replication for broadcasts) that realizes operand reuse in the
    layout itself.  This is the explicit alternative to recovering reuse via
    trace dedup, making the per-operator cost comparison fair by construction
    and enabling cross-operator layout co-optimization over fused regions.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Marks tile boundaries in a persistent kernel so the trace runtime can scope its
// per-dispatch collapse correctly.
def IMTileBoundary : Pass<"im-tile-boundary", "mlir::ModuleOp"> {
  let summary = "Call __pim_tile_boundary() at each tile-loop back edge";
  let description = [{
    Only runs when the module carries `im.persistent`. Inserts a call to the
    runtime's `__pim_tile_boundary()` at the end of the OUTERMOST `scf.for` body
    in every function.

    Why it is needed. The trace runtime folds the N bank replays of one all-bank
    command into one event, and scopes that fold to a dispatch. With one tile per
    program instance the host's program-id setter marks the dispatch. A persistent
    kernel puts the tile loop inside the kernel and the bank loop outside, so bank 0
    walks every tile before bank 1 starts; without a signal from inside the loop the
    whole run looks like one dispatch and every cross-tile re-read of the same
    address collapses to nothing. That reads as a large speedup and nothing warns.

    Why the outermost loop. In a persistent kernel the tile loop is outermost by
    construction and the reduction loop is nested inside it. Marking the specific
    loop in kernel source instead would put the decision back in the kernel.

    Ordering. Must run BEFORE scf-to-cf: afterwards the back edge is a branch and
    identifying the tile loop means recovering the structure just discarded.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
}

// Marks reductions that fold the lanes of one DRAM column, so the trace runtime can
// price a cost no memory access represents.
def IMLaneFold : Pass<"im-lane-fold", "mlir::ModuleOp"> {
  let summary = "Call __pim_trace_fold(steps, outputs) after each in-column reduce";
  let description = [{
    Only runs when the module carries `im.price_folds`. A reduce whose axis is the
    layout's contiguous axis within a lane adds values that shared a DRAM column,
    which the lane-parallel PIM MAC cannot do: it is a tree of log2(values) steps.
    The call after the reduce reports the steps and the outputs per lane. A reduce
    along any other axis is MAC accumulation across columns and is not marked.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect"];
}

// Marks loads a ReLU consumes, so the dcc-parity runtime issues them as PIM_RELU.
def IMReluOpcode : Pass<"im-relu-opcode", "mlir::ModuleOp"> {
  let summary = "Bracket each ReLU-only load with __pim_trace_op calls";
  let description = [{
    Only runs when the module carries `im.dcc_relu_opcode`. A load whose only use,
    through float casts, is a max against zero is bracketed by
    `__pim_trace_op(1)` and `__pim_trace_op(0)`, so the runtime issues its reads as
    DCC's PIM_RELU. This adopts DCC's pricing of ReLU, which their simulator does not
    space on the command bus.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect"];
}

// Groups a loop's loads by tensor, n unrolled iterations at a time.
def IMLoadCluster : Pass<"im-load-cluster", "mlir::ModuleOp"> {
  let summary = "Unroll a two-tensor loop and group each strip's loads by argument";
  let description = [{
    Only runs when the module carries `im.cluster_loads = n`, the register-file entries
    per lane. An innermost scf.for with a constant trip count whose body loads two or
    more kernel arguments, stores none of them and touches memory no other way is
    unrolled by G, the largest proper divisor of its trip count whose G iterations of
    any one argument fit n entries. Each strip's loads are grouped by argument, so a
    DRAM row stays open for G loads. With `im.cluster_serpentine` the body holds two
    strips and the second takes the arguments in reverse. Reads the lane layout.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::triton::TritonDialect"];
}

def IMReductionDistribute : Pass<"im-reduction-distribute", "mlir::ModuleOp"> {
  let summary = "Interchange a reduction loop with the sum of loads it accumulates";
  let description = [{
    Only runs when the module carries `im.distribute_reductions`. A loop whose only
    carried value is acc and whose body yields acc + (x + z), with x and z each built
    from loads and independent of acc, becomes two loops, one adding x and one adding z.
    Reassociates the floating-point sum.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::arith::ArithDialect"];
}

def IMOperandHoist : Pass<"im-operand-hoist", "mlir::ModuleOp"> {
  let summary = "Fully unroll a reduction loop so its tile-invariant operand can be hoisted";
  let description = [{
    Only runs when the module carries `im.hoist_operand_regs = n`. An innermost scf.for
    with a constant trip count T, inside another loop, whose loads depend on nothing in
    the outer loop but its own induction variable, is fully unrolled when T copies of
    those loads fit n register-file entries per lane. triton-licm then hoists them above
    the outer loop. Runs after rewrite-im-layout, because the count reads the layout.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::triton::TritonDialect"];
}

#endif // TRITONIM_CONVERSION_PASSES
