/// TritonIMToLLVM.cpp — Lower TritonGPU IR → LLVM for In-Memory targets.
///
/// This pass reuses the shared TritonGPUToLLVM lowering patterns
/// (elementwise, memory, SPMD, control-flow, …) with an IM-specific
/// TargetInfo that models an HBM-PIM architecture with bank-level
/// SIMD parallelism.  Each PIM bank is mapped to one Triton "thread"
/// (threads_per_warp = numBanks).  The bank index is provided at
/// runtime via the extern function __pim_get_bank_id().
///
/// ---------- Vectorization strategy ----------
///
/// Each PIM bank owns `sizePerThread` consecutive elements (set by the
/// RewriteIMLayout pass).  When those elements are contiguous in
/// memory, we can load/store them as a single LLVM vector operation
/// instead of N scalar operations.
///
/// Memory contiguity is determined by Triton's AxisInfo dataflow
/// analysis — the same mechanism used by the NVIDIA and AMD backends.
/// AxisInfo tracks pointer arithmetic (make_range, splat, addptr)
/// and computes per-dimension contiguity, divisibility, and constancy.
/// The vector width formula is:
///
///     contiguity = axisInfoAnalysis.getContiguity(ptr)
///     vec = min(128 / pointeeBitWidth, contiguity)
///
/// For a simple AXPY kernel (`ptr = base + pid*BLOCK + arange(0,BLOCK)`)
/// AxisInfo computes contiguity = sizePerThread, yielding full
/// vectorization.  For indirect patterns like `A + offs * K + k`,
/// AxisInfo correctly reports contiguity = 1, preventing incorrect
/// vector loads.
///
/// Mask handling:
///   vec is further clamped by the mask's alignment (the number of
///   consecutive mask elements that share the same truth value):
///
///       vec = min(vec, maskAlignment)
///
///   - Loads: branch on mask[vecStart] (group predicate), then
///     per-element `llvm.select` to handle partial masks at the
///     boundary.
///   - Stores: AND all mask bits in the group.  If all-true, emit a
///     single vector store.  Otherwise, fall back to per-element
///     scalar stores with individual mask checks.
///
/// When vec==1 (e.g. non-contiguous access or sub-byte elements),
/// this gracefully degrades to scalar behavior.

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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
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
// ttg.barrier → erase (no shared memory on IM)
// --------------------------------------------------------------------------

/// IM has no shared memory and num_warps == 1, so cross-warp
/// synchronization is meaningless. The membar pass still inserts
/// `ttg.barrier` ops at convert_layout boundaries (it does so as a
/// side effect that also updates allocation.offset attrs that
/// downstream patterns require), but those barriers must be erased
/// before final lowering. This pattern erases them.
struct IMBarrierOpErase
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op,
                  typename triton::gpu::BarrierOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// --------------------------------------------------------------------------
// Base class for vectorized load/store conversions
// --------------------------------------------------------------------------

/// Shared helper for IMLoadOpConversion and IMStoreOpConversion.
///
/// Holds a reference to ModuleAxisInfoAnalysis and provides methods to
/// compute the vector width for memory operations.  This follows the
/// same pattern used by the AMD and NVIDIA backends
/// (LoadStoreConversionBase in their LoadStoreOpToLLVM.cpp files).
///
/// The vector width formula is:
///
///     vec = min(128 / pointeeBitWidth, contiguity)
///
/// - 128 bits is the maximum LLVM vector width we emit (matches GPU
///   backends; yields <4 x i32>, <8 x i16>, <16 x i8>, etc.).
/// - `contiguity` is the number of consecutive elements accessed per
///   thread in the fastest-varying dimension, derived from Triton's
///   AxisInfo dataflow analysis.  After RewriteIMLayout sets
///   sizePerThread=[K], the contiguity for `pid*BLOCK + arange(0,BLOCK)`
///   patterns is exactly K.
///
/// When a mask is present, the vector width is further clamped to the
/// mask's alignment (the number of consecutive mask elements that share
/// the same value):
///
///     vec = min(vec, maskAlignment)
///
/// This ensures that within each vector group, either all elements are
/// active or all are inactive, so we can use a single predicate bit
/// for the whole vector load/store.
struct IMLoadStoreConversionBase {
  explicit IMLoadStoreConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  /// Return the number of contiguous elements accessed per thread
  /// along the fastest-varying dimension, as determined by AxisInfo
  /// dataflow analysis on the pointer operand.
  unsigned getContiguity(Value ptr) const {
    return axisAnalysisPass.getContiguity(ptr);
  }

  /// Compute the vector width for a pointer-typed tensor operand.
  ///
  /// vec = min(128 / pointeeBitWidth, contiguity)
  ///
  /// AxisInfo tracks pointer arithmetic (make_range, splat, addptr,
  /// muli, etc.) and correctly reports contiguity=1 for non-contiguous
  /// patterns like `A + offs * K + k`.
  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    if (pointeeBitWidth == 0)
      return 1;
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  /// Return the mask alignment — the number of consecutive mask
  /// elements guaranteed to share the same truth value.  Used to
  /// clamp vec so that vector loads/stores don't straddle mask
  /// boundaries.
  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;

  /// Build a zero-valued LLVM vector constant.  Used as the `other`
  /// (false-value) default when no explicit `other` operand is given
  /// on a masked load.
  static Value createZeroVector(OpBuilder &builder, Location loc,
                                VectorType vecTy) {
    auto zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseVal = DenseElementsAttr::get(cast<ShapedType>(vecTy), zeroAttr);
    return LLVM::ConstantOp::create(builder, loc, vecTy, denseVal);
  }

  /// Pack `elems[start .. start+vec-1]` into an LLVM vector value.
  /// Used to build the `other` vector for masked loads and the value
  /// vector for stores.
  ///
  /// Example for vec=4, start=0, elems=[a, b, c, d, ...]:
  ///   %v = undef : <4 x i32>
  ///   %v = insertelement %v, a, 0
  ///   %v = insertelement %v, b, 1
  ///   %v = insertelement %v, c, 2
  ///   %v = insertelement %v, d, 3
  static Value packElementRange(RewriterBase &rewriter,
                                const TypeConverter *typeConverter,
                                Location loc, VectorType vecTy,
                                ArrayRef<Value> elems, unsigned start) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned vec = vecTy.getNumElements();
    Value v = b.undef(vecTy);
    for (unsigned s = 0; s < vec; ++s) {
      Value idx = LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, elems[start + s], idx);
    }
    return v;
  }
};

// --------------------------------------------------------------------------
// tt.load → vectorized LLVM loads  (IM flat memory)
// --------------------------------------------------------------------------

/// Lower `tt.load` to LLVM vector load operations.
///
/// For each group of `vec` contiguous elements assigned to this
/// bank/thread, we emit a single `LLVM::LoadOp` of type `<vec x elemTy>`.
/// Individual scalar values are then extracted via `llvm.extractelement`.
///
/// Vector width is derived from AxisInfo dataflow analysis on the
/// pointer operand (see `IMLoadStoreConversionBase::getVectorSize`).
/// When a mask is present, vec is further clamped to `getMaskAlignment`
/// so that within each vector group, all mask bits are uniform.
///
/// Mask handling:
///   The group predicate is `mask[vecStart]` — the first element of
///   the vec-group.  After mask-alignment clamping, all elements in
///   the group share the same mask value, so a single branch suffices.
///   Per-element `llvm.select` after the load handles edge cases.
////// Unmasked loads (vec=4, i32):
///
///   %vec = load <4 x i32>, ptr %p       ; single 128-bit load
///   %e0 = extractelement %vec, 0
///   %e1 = extractelement %vec, 1
///   %e2 = extractelement %vec, 2
///   %e3 = extractelement %vec, 3
///
/// Masked loads (vec=4, i32) — branch + per-element select:
///
///   currentBlock:
///     %pred = mask[0]                     ; group predicate
///     cond_br %pred → loadBlock, afterBlock(other[0..3])
///
///   loadBlock:
///     %vec = load <4 x i32>, ptr %p
///     %e0 = extractelement %vec, 0
///     %e1 = extractelement %vec, 1
///     %e2 = extractelement %vec, 2
///     %e3 = extractelement %vec, 3
///     %r0 = select mask[0], %e0, other[0]
///     %r1 = select mask[1], %e1, other[1]
///     %r2 = select mask[2], %e2, other[2]
///     %r3 = select mask[3], %e3, other[3]
///     br afterBlock(%r0, %r1, %r2, %r3)
///
///   afterBlock(%a0, %a1, %a2, %a3 : i32):   ; per-element phi nodes
///     ...
///
struct IMLoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                            public IMLoadStoreConversionBase {
  IMLoadOpConversion(LLVMTypeConverter &converter,
                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        IMLoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // ---- Operands ----
    Value ptr = op.getPtr();   // original Triton ptr (tensor of pointers)
    Value mask = op.getMask(); // original Triton mask (or null)

    Value llPtr = adaptor.getPtr();     // lowered LLVM struct of ptrs
    Value llMask = adaptor.getMask();   // lowered LLVM struct of i1s
    Value llOther = adaptor.getOther(); // lowered LLVM struct of elems

    // ---- Types and element count ----
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = ttg::getTotalElemsPerThread(ptr.getType());

    // ---- Compute vector width from AxisInfo ----
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<unsigned>(vec, getMaskAlignment(mask));

    // ---- Unpack operand elements ----
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // ---- Build the LLVM vector type ----
    Type vecTy =
        (vec > 1) ? LLVM::getVectorType(valueElemTy, vec) : valueElemTy;

    // ---- Emit one (vector) load per group of `vec` elements ----
    SmallVector<Value> loadedVals;
    for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value basePtr = ptrElems[vecStart];

      if (!mask) {
        // ---- Unconditional vector load ----
        Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, basePtr);

        if (vec > 1) {
          for (unsigned j = 0; j < vec; ++j) {
            Value idx = createIndexAttrConstant(
                rewriter, loc, getTypeConverter()->getIndexType(), j);
            loadedVals.push_back(b.extract_element(valueElemTy, loaded, idx));
          }
        } else {
          loadedVals.push_back(loaded);
        }

      } else if (vec == 1) {
        // ---- Scalar masked load (vec==1) ----
        // Simple diamond: branch on mask, load or use other.
        Value falseVal =
            !otherElems.empty() ? otherElems[vecStart] : b.undef(valueElemTy);

        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        afterBlock->addArgument(valueElemTy, loc);
        Block *loadBlock = rewriter.createBlock(afterBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[vecStart], loadBlock,
                               ValueRange{}, afterBlock, ValueRange{falseVal});

        rewriter.setInsertionPointToStart(loadBlock);
        Value loaded =
            LLVM::LoadOp::create(rewriter, loc, valueElemTy, basePtr);
        LLVM::BrOp::create(rewriter, loc, ValueRange{loaded}, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);
        loadedVals.push_back(afterBlock->getArgument(0));

      } else {
        // ---- Masked vector load with per-element select (vec>1) ----
        //
        // Group predicate: mask[vecStart].
        //   - If false, the IM monotonic-mask property guarantees all
        //     mask[vecStart..vecStart+vec-1] are false → use `other`.
        //   - If true, we load the full vector and use per-element
        //     `llvm.select` to handle the (rare) case where trailing
        //     elements in the group are masked out.
        //
        // CFG:
        //   currentBlock: cond_br mask[vecStart] → loadBlock,
        //                                          afterBlock(other[0..vec-1])
        //   loadBlock: load <vec x T>, extract, select per elem, br afterBlock
        //   afterBlock(phis): continue

        // Prepare the per-element `other` values (false-path scalars).
        SmallVector<Value> otherVals;
        for (unsigned j = 0; j < vec; ++j) {
          otherVals.push_back(!otherElems.empty() ? otherElems[vecStart + j]
                                                  : b.undef(valueElemTy));
        }

        // Split: currentBlock → afterBlock (with vec scalar phi args).
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        for (unsigned j = 0; j < vec; ++j)
          afterBlock->addArgument(valueElemTy, loc);
        Block *loadBlock = rewriter.createBlock(afterBlock);

        // currentBlock: branch on group predicate.
        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[vecStart], loadBlock,
                               ValueRange{}, afterBlock, ValueRange(otherVals));

        // loadBlock: vector load → extract → per-element select.
        rewriter.setInsertionPointToStart(loadBlock);
        Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, basePtr);

        SmallVector<Value> selectedVals;
        for (unsigned j = 0; j < vec; ++j) {
          Value idx = createIndexAttrConstant(
              rewriter, loc, getTypeConverter()->getIndexType(), j);
          Value elem = b.extract_element(valueElemTy, loaded, idx);
          // Per-element select: use loaded value if mask is true,
          // else use `other`.  For the common case (all masks true),
          // LLVM will fold these selects away.
          Value sel = LLVM::SelectOp::create(rewriter, loc, valueElemTy,
                                             maskElems[vecStart + j], elem,
                                             otherVals[j]);
          selectedVals.push_back(sel);
        }
        LLVM::BrOp::create(rewriter, loc, ValueRange(selectedVals), afterBlock);

        // afterBlock: collect phi results.
        rewriter.setInsertionPointToStart(afterBlock);
        for (unsigned j = 0; j < vec; ++j)
          loadedVals.push_back(afterBlock->getArgument(j));
      }
    }

    // ---- Pack all scalar values back into the LLVM struct ----
    Type llvmResultTy = getTypeConverter()->convertType(valueTy);
    Value result = packLLElements(loc, getTypeConverter(), loadedVals, rewriter,
                                  llvmResultTy);
    rewriter.replaceOp(op, {result});
    return success();
  }
};

// --------------------------------------------------------------------------
// tt.store → vectorized LLVM stores  (IM flat memory)
// --------------------------------------------------------------------------

/// Lower `tt.store` to LLVM vector store operations.
///
/// For each group of `vec` contiguous elements assigned to this
/// bank/thread, we pack the scalars into an LLVM vector via
/// `llvm.insertelement`, then emit a single `LLVM::StoreOp`.
///
/// Vector width is derived from AxisInfo dataflow analysis on the
/// pointer operand (see `IMLoadStoreConversionBase::getVectorSize`).
/// When a mask is present, vec is further clamped to `getMaskAlignment`.
///
/// Mask handling (vec > 1):
///   A vector store writes ALL elements atomically, so we can only
///   use it when every element in the group should actually be stored.
///   We AND all mask bits in the group; if the result is true, we emit
///   a single vector store.  Otherwise, we fall back to per-element
///   scalar stores guarded by individual mask checks.
///
///   The AND-all check is essentially free for the common case where
///   all masks are true (every program except the boundary one).
///   The scalar fallback only activates for the single boundary
///   program where N falls within the thread's element range.
///
/// Unmasked stores (vec=4, i32):
///
///   %v = insertelement undef, val[0], 0
///   %v = insertelement %v,   val[1], 1
///   %v = insertelement %v,   val[2], 2
///   %v = insertelement %v,   val[3], 3
///   store <4 x i32> %v, ptr %p              ; single 128-bit store
///
/// Masked stores (vec=4, i32) — AND-predicated vector or scalar fallback:
///
///   currentBlock:
///     %all = and mask[0], mask[1]
///     %all = and %all, mask[2]
///     %all = and %all, mask[3]
///     %v = <pack 4 scalars into vector>
///     cond_br %all → vecStoreBlock, scalarBlock
///
///   vecStoreBlock:
///     store <4 x i32> %v, ptr %p
///     br afterBlock
///
///   scalarBlock:                            ; per-element guarded stores
///     cond_br mask[0] → s0, skip0
///   s0: store i32 val[0], ptr[0]; br skip0
///   skip0: cond_br mask[1] → s1, skip1
///   ...
///     br afterBlock
///
///   afterBlock:
///     ...
///
struct IMStoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                             public IMLoadStoreConversionBase {
  IMStoreOpConversion(LLVMTypeConverter &converter,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        IMLoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // ---- Operands ----
    Value ptr = op.getPtr();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    // ---- Types and element count ----
    Type valueElemTy = typeConverter->convertType(
        getElementTypeOrSelf(op.getValue().getType()));
    unsigned numElems = ttg::getTotalElemsPerThread(ptr.getType());

    // ---- Compute vector width from AxisInfo ----
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<unsigned>(vec, getMaskAlignment(mask));

    // ---- Unpack operand elements ----
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    // ---- Emit one (vector) store per group of `vec` elements ----
    for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value basePtr = ptrElems[vecStart];

      if (!mask) {
        // ---- Unconditional vector store ----
        if (vec > 1) {
          auto vecTy = cast<VectorType>(LLVM::getVectorType(valueElemTy, vec));
          Value v = packElementRange(rewriter, getTypeConverter(), loc, vecTy,
                                     valueElems, vecStart);
          LLVM::StoreOp::create(rewriter, loc, v, basePtr);
        } else {
          LLVM::StoreOp::create(rewriter, loc, valueElems[vecStart], basePtr);
        }

      } else if (vec == 1) {
        // ---- Scalar masked store (vec==1) ----
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        Block *storeBlock = rewriter.createBlock(afterBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, maskElems[vecStart], storeBlock,
                               afterBlock);

        rewriter.setInsertionPointToStart(storeBlock);
        LLVM::StoreOp::create(rewriter, loc, valueElems[vecStart], basePtr);
        LLVM::BrOp::create(rewriter, loc, afterBlock);

        rewriter.setInsertionPointToStart(afterBlock);

      } else {
        // ---- Masked vector store with AND-all predicate (vec>1) ----
        //
        // Compute allTrue = AND of all mask elements in the group.
        // If all true → single vector store (common case).
        // Else → per-element scalar stores with individual guards.

        // AND all mask elements together.
        Value allTrue = maskElems[vecStart];
        for (unsigned j = 1; j < vec; ++j) {
          allTrue = LLVM::AndOp::create(rewriter, loc, allTrue,
                                        maskElems[vecStart + j]);
        }

        // Pack the scalars into a vector for the fast path.
        auto vecTy = cast<VectorType>(LLVM::getVectorType(valueElemTy, vec));
        Value vecVal = packElementRange(rewriter, getTypeConverter(), loc,
                                        vecTy, valueElems, vecStart);

        // Split CFG: currentBlock → afterBlock.
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        Block *scalarBlock = rewriter.createBlock(afterBlock);
        Block *vecStoreBlock = rewriter.createBlock(scalarBlock);

        // currentBlock: branch on allTrue.
        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, allTrue, vecStoreBlock,
                               scalarBlock);

        // vecStoreBlock: single vector store → afterBlock.
        rewriter.setInsertionPointToStart(vecStoreBlock);
        LLVM::StoreOp::create(rewriter, loc, vecVal, basePtr);
        LLVM::BrOp::create(rewriter, loc, afterBlock);

        // scalarBlock: per-element conditional stores.
        // Each element gets its own diamond: cond_br → storeJ / skipJ.
        rewriter.setInsertionPointToStart(scalarBlock);
        for (unsigned j = 0; j < vec; ++j) {
          Block *curBlk = rewriter.getInsertionBlock();
          Block *nextBlk =
              (j + 1 < vec) ? rewriter.createBlock(afterBlock) : afterBlock;
          Block *storeJ = rewriter.createBlock(nextBlk);

          rewriter.setInsertionPointToEnd(curBlk);
          LLVM::CondBrOp::create(rewriter, loc, maskElems[vecStart + j], storeJ,
                                 nextBlk);

          rewriter.setInsertionPointToStart(storeJ);
          LLVM::StoreOp::create(rewriter, loc, valueElems[vecStart + j],
                                ptrElems[vecStart + j]);
          LLVM::BrOp::create(rewriter, loc, nextBlk);

          rewriter.setInsertionPointToStart(nextBlk);
        }
        // After the loop, insertion point is at the start of afterBlock.
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

    // Read the number of PIM banks from the IM-specific module attribute.
    // (The compiler also sets ttg.threads-per-warp = num_banks so that
    // TritonGPU's BlockedEncodingAttr distributes elements across banks,
    // but our code reads the canonical "im.num-banks" attribute directly.)
    unsigned numBanks = 1;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("im.num-banks"))
      numBanks = attr.getInt();

    triton::im::TargetInfo targetInfo(numBanks);

    // -- Allocation: analyze shared-memory needs (which IM has none of,
    //    but TritonGPU's shared lowering patterns invoke
    //    `getSharedMemoryBase` for any convert_layout that crosses
    //    warps — the cross-warp path materializes via shared memory,
    //    and the lowering reads `allocation.offset` attrs from the
    //    op). `attachAllocationSizeAndOffsetAttr` walks the module
    //    and sets those attrs from the allocation analysis; without
    //    it, lowering trips an assertion when emitting wider tile
    //    shapes (e.g. BLOCK_M=32, sizePerThread=[32,2]).
    //
    //    Membar pass also runs because it inserts ttg.barrier ops
    //    around shared-memory-reading layout conversions. IM has no
    //    real shared memory so those barriers are no-ops — they're
    //    erased by IMBarrierOpErase below. --
    ModuleAllocation allocation(mod);
    triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
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

    // -- AxisInfo analysis (after Phase 1) --
    // AxisInfo tracks pointer arithmetic (make_range, splat, addptr)
    // to compute per-dimension contiguity, divisibility, and constancy.
    // Used by both the shared elementwise patterns and our IM-specific
    // load/store patterns to determine safe vectorization widths.
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // ---- Phase 2: lower remaining TritonGPU ops ----

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
    // IM-specific: lower tt.load / tt.store → vectorized LLVM ops.
    // Vector width is derived from AxisInfo dataflow analysis on the
    // pointer operand, matching the NVIDIA/AMD backends.
    patterns.add<IMLoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
    patterns.add<IMStoreOpConversion>(typeConverter, axisInfoAnalysis, benefit);
    // IM-specific: erase ttg.barrier (IM has no shared memory and
    // num_warps == 1, so cross-warp barriers are no-ops). Higher
    // priority than the shared-pattern BarrierOpConversion (which
    // would lower to gpu.barrier).
    patterns.add<IMBarrierOpErase>(typeConverter, PatternBenefit(benefit + 1));

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
