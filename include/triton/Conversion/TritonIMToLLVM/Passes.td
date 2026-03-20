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
    generic In-Memory processing architectures.  The pass reuses the
    shared TritonGPUToLLVM lowering patterns (elementwise, memory,
    control-flow, SPMD, etc.) with an IM-specific TargetInfo that
    maps the execution model to a single-threaded, flat-memory PIM
    unit.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect", "mlir::scf::SCFDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif // TRITONIM_CONVERSION_PASSES
