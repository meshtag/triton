//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {

class TritonGPUTypeConverter : public TypeConverter {
public:
  /// `target` is the value of the `--target` pass option (e.g.
  /// "cuda:80", "im:hbm-pim"). When the target starts with "im:" the
  /// converter produces an IM-optimal BlockedEncodingAttr with
  /// `sizePerThread = shape / (threadsPerWarp × warpsPerCTA)` so each
  /// PIM bank owns a consecutive chunk of the tensor — this matches
  /// what `rewrite-im-layout` would produce as a follow-up pass, and
  /// crucially avoids the cross-lane convert_layout that the default
  /// (`sizePerThread = 1`) encoding induces. Without this, wider
  /// tile shapes (BLOCK_M > 16 on the current Triton fork) trip a
  /// shared-memory-mediated layout transform that the IM backend
  /// has no lowering for.
  TritonGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                         int numCTAs, bool enableSourceRemat,
                         llvm::StringRef target = {});
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }
  llvm::StringRef getTarget() const { return target; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
  std::string target;
};

class TritonGPUConversionTarget : public ConversionTarget {
public:
  explicit TritonGPUConversionTarget(MLIRContext &ctx,
                                     TritonGPUTypeConverter &typeConverter);

  // Determine whether the operation is currently legal. I.e. it has layouts
  // assigned to its tensor operands and results.
  static bool isDynamicallyLegal(Operation *op,
                                 const TypeConverter &typeConverter);
};

namespace impl {
LogicalResult convertGatherScatterOp(Operation *op, ValueRange operands,
                                     OpOperand &xOffsetsMutable,
                                     const TypeConverter &typeConverter,
                                     ConversionPatternRewriter &rewriter);
} // namespace impl

// Generic pattern for converting a TMA gather or scatter operation.
template <typename OpT>
struct GatherScatterOpPattern : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return impl::convertGatherScatterOp(op, adaptor.getOperands(),
                                        op.getXOffsetsMutable(),
                                        *this->getTypeConverter(), rewriter);
  }
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
