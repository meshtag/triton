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

#endif // TRITONIM_CONVERSION_PASSES
