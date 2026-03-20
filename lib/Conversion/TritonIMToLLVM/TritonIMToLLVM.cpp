/// TritonIMToLLVM.cpp — Lower TritonGPU IR → LLVM for In-Memory targets.
///
/// This pass reuses the shared TritonGPUToLLVM lowering patterns
/// (elementwise, memory, SPMD, control-flow, …) with an IM-specific
/// TargetInfo that models a single-threaded, flat-memory PIM unit.

#include "TargetInfo.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// --------------------------------------------------------------------------
// TableGen pass base class
// --------------------------------------------------------------------------
namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_CONVERTTRITONIMTOLLVM
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

namespace ttg = mlir::triton::gpu;

// --------------------------------------------------------------------------
// gpu::ThreadIdOp → constant 0  (IM is single-threaded)
// --------------------------------------------------------------------------

/// Replace `gpu.thread_id` with a constant zero index.  The PIM execution
/// model has exactly one thread, so the thread ID is always 0.
struct IMThreadIdOpConversion : public RewritePattern {
  IMThreadIdOpConversion(MLIRContext *ctx, PatternBenefit benefit)
      : RewritePattern(::mlir::gpu::ThreadIdOp::getOperationName(), benefit,
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

// --------------------------------------------------------------------------
// tt.load → scalar LLVM loads  (IM single-threaded flat memory)
// --------------------------------------------------------------------------

struct IMLoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Value ptr = op.getPtr();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = ttg::getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numElems; ++i) {
      if (!mask) {
        // Unconditional scalar load.
        auto loaded =
            LLVM::LoadOp::create(rewriter, loc, valueElemTy, ptrElems[i]);
        loadedVals.push_back(loaded);
      } else {
        // Masked load: branch on mask[i].
        Value falseVal =
            i < otherElems.size() ? otherElems[i] : b.undef(valueElemTy);

        Block *currentBlock = rewriter.getInsertionBlock();
        auto ip = rewriter.getInsertionPoint();
        Block *afterBlock = rewriter.splitBlock(currentBlock, ip);
        afterBlock->addArgument(valueElemTy, loc);
        Block *loadBlock = rewriter.createBlock(afterBlock);

        // currentBlock: cond_br mask → loadBlock, afterBlock(falseVal)
        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[i], loadBlock,
                               ValueRange{}, afterBlock, ValueRange{falseVal});

        // loadBlock: %v = load ptr; br afterBlock(%v)
        rewriter.setInsertionPointToStart(loadBlock);
        Value loaded =
            LLVM::LoadOp::create(rewriter, loc, valueElemTy, ptrElems[i]);
        LLVM::BrOp::create(rewriter, loc, ValueRange{loaded}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        loadedVals.push_back(afterBlock->getArgument(0));
      }
    }

    Type llvmResultTy = getTypeConverter()->convertType(valueTy);
    Value result = packLLElements(loc, getTypeConverter(), loadedVals, rewriter,
                                  llvmResultTy);
    rewriter.replaceOp(op, {result});
    return success();
  }
};

// --------------------------------------------------------------------------
// tt.store → scalar LLVM stores  (IM single-threaded flat memory)
// --------------------------------------------------------------------------

struct IMStoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    unsigned numElems = ttg::getTotalElemsPerThread(op.getPtr().getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    for (unsigned i = 0; i < numElems; ++i) {
      if (!op.getMask()) {
        // Unconditional store.
        LLVM::StoreOp::create(rewriter, loc, valueElems[i], ptrElems[i]);
      } else {
        // Masked store: branch on mask[i].
        Block *currentBlock = rewriter.getInsertionBlock();
        auto ip = rewriter.getInsertionPoint();
        Block *afterBlock = rewriter.splitBlock(currentBlock, ip);
        Block *storeBlock = rewriter.createBlock(afterBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[i], storeBlock,
                               afterBlock);

        rewriter.setInsertionPointToStart(storeBlock);
        LLVM::StoreOp::create(rewriter, loc, valueElems[i], ptrElems[i]);
        LLVM::BrOp::create(rewriter, loc, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// --------------------------------------------------------------------------
// Conversion targets
// --------------------------------------------------------------------------

/// Target for the function-signature rewriting phase.
class IMLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit IMLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<UnrealizedConversionCastOp>();
  }
};

/// Target for the bulk TritonGPU → LLVM lowering phase.
class IMLLVMConversionTarget : public ConversionTarget {
public:
  explicit IMLLVMConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<::mlir::gpu::GPUDialect>();
    addLegalOp<UnrealizedConversionCastOp>();
  }
};

// --------------------------------------------------------------------------
// Pass implementation
// --------------------------------------------------------------------------

struct ConvertTritonIMToLLVM
    : public triton::im::impl::ConvertTritonIMToLLVMBase<
          ConvertTritonIMToLLVM> {
  using ConvertTritonIMToLLVMBase::ConvertTritonIMToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    triton::im::TargetInfo targetInfo;

    // -- Allocation & membar (kept for compatibility with shared patterns,
    //    but no shared memory is actually used on IM targets) --
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // -- Type converter --
    LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // ---- Phase 1: lower function signatures ----
    {
      IMLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      triton::populateFuncOpConversionPattern(typeConverter, funcPatterns,
                                              targetInfo,
                                              triton::patternBenefitDefault);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // ---- Phase 2: lower remaining TritonGPU ops ----
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = triton::patternBenefitPrioritizeOverLLVMConversions;

    triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           benefit);
    triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                  patterns, benefit);
    triton::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                        benefit);
    triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    triton::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);
    triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                             patterns, benefit);
    triton::populateReduceOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                           benefit);
    triton::populateScanOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                         benefit);
    triton::populateGatherOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                           benefit);
    triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                              targetInfo, benefit);
    triton::populatePrintOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                         benefit);
    triton::populateAssertOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                          benefit);

    // IM-specific: lower gpu::ThreadIdOp → constant 0.
    patterns.insert<IMThreadIdOpConversion>(context, PatternBenefit(benefit));
    // IM-specific: lower tt.load / tt.store → scalar LLVM ops.
    patterns.add<IMLoadOpConversion>(typeConverter, benefit);
    patterns.add<IMStoreOpConversion>(typeConverter, benefit);

    // Standard MLIR conversion patterns (arith, math, cf, ub).
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    IMLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // ---- Phase 3: lower residual CF ops ----
    {
      IMLLVMFunctionConversionTarget cfTarget(*context);
      cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return op->getDialect() !=
               context->getLoadedDialect<cf::ControlFlowDialect>();
      });
      RewritePatternSet cfPatterns(context);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            cfPatterns);
      if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
        return signalPassFailure();
    }
  }
};

} // anonymous namespace

// --------------------------------------------------------------------------
// Public entry point
// --------------------------------------------------------------------------

namespace mlir {
namespace triton {
namespace im {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonIMToLLVMPass() {
  return std::make_unique<ConvertTritonIMToLLVM>();
}

} // namespace im
} // namespace triton
} // namespace mlir
