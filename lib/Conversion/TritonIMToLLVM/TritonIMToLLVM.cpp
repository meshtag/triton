/// TritonIMToLLVM.cpp — Lower TritonGPU IR → LLVM for In-Memory targets.
///
/// This pass reuses the shared TritonGPUToLLVM lowering patterns
/// (elementwise, memory, SPMD, control-flow, …) with an IM-specific
/// TargetInfo that models an HBM-PIM architecture with bank-level
/// SIMD parallelism.  Each PIM bank is mapped to one Triton "thread"
/// (threads_per_warp = numBanks).  The bank index is provided at
/// runtime via the extern function __pim_get_bank_id().

#include "TargetInfo.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
// gpu::ThreadIdOp → PIM bank index
// --------------------------------------------------------------------------

/// Replace `gpu.thread_id` with a call to the extern runtime function
/// `__pim_get_bank_id() -> i32`.  On HBM-PIM each bank executes the
/// kernel in lock-step; the bank index is the analogue of a GPU
/// thread ID.  The runtime / trace harness provides the implementation.
struct IMThreadIdOpConversion : public RewritePattern {
  IMThreadIdOpConversion(MLIRContext *ctx, PatternBenefit benefit)
      : RewritePattern(::mlir::gpu::ThreadIdOp::getOperationName(), benefit,
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    Type i32 = IntegerType::get(ctx, 32);

    // Declare __pim_get_bank_id() if not yet present.
    auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__pim_get_bank_id");
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto fnType = LLVM::LLVMFunctionType::get(i32, {});
      funcOp =
          LLVM::LLVMFuncOp::create(rewriter, loc, "__pim_get_bank_id", fnType);
    }

    // Call and wrap in index_cast (gpu.thread_id returns IndexType;
    // the arith→LLVM patterns lower the cast in the same step).
    auto callOp = LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{});
    Value bankIdx = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), callOp.getResult());
    rewriter.replaceOp(op, bankIdx);
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

    // Read the number of PIM banks from the module attribute
    // (threads-per-warp models bank-level parallelism on HBM-PIM).
    unsigned numBanks = 1;
    if (auto attr =
            mod->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumThreadsPerWarp))
      numBanks = attr.getInt();

    triton::im::TargetInfo targetInfo(numBanks);

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

    // IM-specific: lower gpu::ThreadIdOp → __pim_get_bank_id().
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
