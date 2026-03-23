/// RewriteIMLayout.cpp — Rewrite tensor layout for PIM bank-consecutive access.
///
/// After ConvertTritonToTritonGPU, every tensor type carries a
/// BlockedEncodingAttr with sizePerThread=[1] (GPU-coalesced, lane-
/// innermost).  For In-Memory processing each PIM bank should own a
/// *consecutive* chunk, so this pass rewrites the encoding to have
/// sizePerThread = shape / (threadsPerWarp × warpsPerCTA) per
/// dimension.  ConvertLayoutOps are inserted around load/store ops;
/// the downstream tritongpu-remove-layout-conversions pass propagates
/// the new encoding through the full IR.
///
/// This pass replaces tritongpu-coalesce in the IM pipeline.

#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include <bit>

// -----------------------------------------------------------------------
// TableGen pass base class
// -----------------------------------------------------------------------
namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_REWRITEIMLAYOUT
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

namespace ttg = mlir::triton::gpu;

/// Compute the IM-optimal BlockedEncodingAttr for the given tensor type.
///
/// For each dimension d:
///   sizePerThread[d] = shape[d] / (threadsPerWarp[d] × warpsPerCTA[d])
///
/// This makes the "register" (per-thread) dimension contiguous so that
/// each PIM bank accesses a consecutive slice of the tensor.
static Attribute buildIMEncoding(RankedTensorType tensorType,
                                 int threadsPerWarp, int numWarps) {
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  if (!blocked)
    return {};

  auto shape = tensorType.getShape();
  int rank = shape.size();

  SmallVector<unsigned> oldTPW(blocked.getThreadsPerWarp());
  SmallVector<unsigned> oldWPC(blocked.getWarpsPerCTA());
  auto order = blocked.getOrder();
  auto cgaLayout = blocked.getCGALayout();

  // Compute new sizePerThread: each bank (lane) owns a consecutive chunk.
  SmallVector<unsigned> newSizePerThread(rank);
  for (int d = 0; d < rank; ++d) {
    unsigned totalThreads = oldTPW[d] * oldWPC[d];
    unsigned spt = std::max<unsigned>(1, shape[d] / totalThreads);
    // sizePerThread must be a power of two.
    spt = llvm::bit_floor(spt);
    newSizePerThread[d] = spt;
  }

  // If nothing changed, skip.
  if (SmallVector<unsigned>(blocked.getSizePerThread()) == newSizePerThread)
    return {};

  return ttg::BlockedEncodingAttr::get(tensorType.getContext(), shape,
                                       newSizePerThread, order, numWarps,
                                       threadsPerWarp, cgaLayout);
}

// -----------------------------------------------------------------------
// Pass implementation
// -----------------------------------------------------------------------

struct RewriteIMLayoutPass
    : public triton::im::impl::RewriteIMLayoutBase<RewriteIMLayoutPass> {
  using RewriteIMLayoutBase::RewriteIMLayoutBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Read module-level attributes.
    int threadsPerWarp =
        ttg::TritonGPUDialect::getThreadsPerWarp(mod); // = num_banks
    int numWarps = ttg::lookupNumWarps(mod.getOperation());

    // Walk all memory ops and compute the IM-optimal encoding.
    llvm::MapVector<Operation *, Attribute> layoutMap;

    mod.walk([&](Operation *op) {
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorType = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorType || !isa<triton::PointerType>(tensorType.getElementType()))
        return;

      Attribute newEnc = buildIMEncoding(tensorType, threadsPerWarp, numWarps);
      if (newEnc)
        layoutMap[op] = newEnc;
    });

    // Insert ConvertLayoutOps around each memory op.
    for (auto &[op, enc] : layoutMap) {
      convertDistributedOpEncoding(enc, op);
    }
  }
};

} // anonymous namespace

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

namespace mlir {
namespace triton {
namespace im {

std::unique_ptr<OperationPass<ModuleOp>> createRewriteIMLayoutPass() {
  return std::make_unique<RewriteIMLayoutPass>();
}

} // namespace im
} // namespace triton
} // namespace mlir
