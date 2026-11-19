// What a kernel states to the trace runtimes, and the runtime functions it calls.
#ifndef TRITONIM_CONVERSION_IMARTIFACT_H
#define TRITONIM_CONVERSION_IMARTIFACT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace triton {
namespace im {

/// Bits one column command moves: the `im.dq_bits` the harness states for its machine (256 on
/// HBM-PIM, 128 on SIMDRAM), or 128 when the module states none. The runtime's cfg_dq_bits
/// has to agree, and @__pim_dq_bits lets it check.
inline int64_t moduleDqBits(ModuleOp mod) {
  if (mod)
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"); a && a.getInt() > 0)
      return a.getInt();
  return 128;
}

/// The external function `name` of type `ty`, declared at the top of the module on first use.
/// `b` keeps its insertion point, so a conversion pattern passes its rewriter.
inline LLVM::LLVMFuncOp getOrDeclareRuntimeFn(OpBuilder &b, ModuleOp mod, Location loc,
                                              StringRef name, LLVM::LLVMFunctionType ty) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(mod.getBody());
  return LLVM::LLVMFuncOp::create(b, loc, name, ty);
}

/// The same for a pass, which declares it at the module's location.
inline LLVM::LLVMFuncOp getOrDeclareRuntimeFn(ModuleOp mod, StringRef name,
                                              LLVM::LLVMFunctionType ty) {
  OpBuilder b(mod.getContext());
  return getOrDeclareRuntimeFn(b, mod, mod.getLoc(), name, ty);
}

/// Refuses an op that would move data between banks, then states the tables the runtimes read
/// as module globals. The lowering calls it before its conversion drops the im.residency
/// stamps most tables come from.
LogicalResult emitArtifact(ModuleOp mod);

} // namespace im
} // namespace triton
} // namespace mlir

#endif // TRITONIM_CONVERSION_IMARTIFACT_H
