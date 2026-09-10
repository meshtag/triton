#ifndef TRITONIM_CONVERSION_TRITOIMTOLLVM_PASSES_H
#define TRITONIM_CONVERSION_TRITOIMTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace im {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonIMToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createRewriteIMLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createIMOperandResidencyLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createIMTileBoundaryPass();

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"

} // namespace im
} // namespace triton
} // namespace mlir

#endif // TRITONIM_CONVERSION_TRITOIMTOLLVM_PASSES_H
