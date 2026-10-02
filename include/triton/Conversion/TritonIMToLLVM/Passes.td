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
    reads ttg.threads-per-warp and ttg.num-warps from the module and computes
    sizePerThread = tensorShape / (threadsPerWarp × warpsPerCTA) for
    each dimension.  ConvertLayoutOps are inserted around load/store
    operations; the downstream tritongpu-remove-layout-conversions
    pass propagates the new encoding through the rest of the IR.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Stamps what the runtimes need to know about each operand, read back through
// emitPimLayoutTable.
def IMOperandResidencyLayout
    : Pass<"im-operand-residency-layout", "mlir::ModuleOp"> {
  let summary = "Stamp each memory access with its replication and footprint";
  let description = [{
    Stamps each tensor-of-pointer load or store, and each scalar load splatted
    across the lanes, with an im.residency dictionary. Its fields include the reuse
    class, how many banks receive a copy (bank_replicated), the kernel argument read
    and the footprint of one program instance. im.schedule and
    im.residency.force can override bank_replicated.
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
    DRAM row stays open for G loads. Reads the lane layout.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::triton::TritonDialect"];
}

def IMDccTileMac : Pass<"im-dcc-tile-mac", "mlir::ModuleOp"> {
  let summary = "Select DCC's tile MAC for a contraction whose per-lane tile is one";
  let description = [{
    Only runs when the module carries `im.dcc_tile_mac`. A reduce over load * broadcast
    whose load keeps one column of outputs lane-fastest and whole tiles of the reduce axis
    next is stamped with the column reads one DCC MAC_OP1 covers. Every load of that
    argument must qualify. Adopts DCC's ISA pricing, a convention of the DCC path.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::triton::TritonDialect"];
}

#endif // TRITONIM_CONVERSION_PASSES
