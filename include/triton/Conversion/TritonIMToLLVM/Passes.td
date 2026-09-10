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

#endif // TRITONIM_CONVERSION_PASSES
