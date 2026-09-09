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
/// Which tensor axis carries the banks.
///
/// `order` decides it: the BlockedEncodingAttr::get overload below fills lanes along
/// the fastest dim first, so order=[1,0] puts all threadsPerWarp banks on axis 1.
/// That choice bounds operand amortisation, because every bank re-reads the operand
/// and the emitted traffic measures MACs*banks/BLOCK[bankAxis]. With banks on N, a
/// shape with small N can never amortise however the tile is picked.
///
/// -1 keeps whatever convert-triton-to-tritongpu chose.
static SmallVector<unsigned> orderForBankAxis(ArrayRef<unsigned> curOrder,
                                              int rank, int bankAxis) {
  SmallVector<unsigned> order(curOrder.begin(), curOrder.end());
  if (bankAxis < 0 || bankAxis >= rank || rank < 2)
    return order;
  // Fastest dim is order[0]. Move the requested axis there, keep the rest.
  order.erase(std::find(order.begin(), order.end(), (unsigned)bankAxis));
  order.insert(order.begin(), (unsigned)bankAxis);
  return order;
}

static Attribute buildIMEncoding(RankedTensorType tensorType,
                                 int threadsPerWarp, int numWarps,
                                 int bankAxis) {
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  if (!blocked)
    return {};

  auto shape = tensorType.getShape();
  int rank = shape.size();

  SmallVector<unsigned> oldTPW(blocked.getThreadsPerWarp());
  SmallVector<unsigned> oldWPC(blocked.getWarpsPerCTA());
  SmallVector<unsigned> order =
      orderForBankAxis(blocked.getOrder(), rank, bankAxis);
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

  // If nothing changed, skip. The order counts as a change: a bank-axis flip can
  // leave sizePerThread identical while moving every lane to the other axis.
  if (SmallVector<unsigned>(blocked.getSizePerThread()) == newSizePerThread &&
      SmallVector<unsigned>(blocked.getOrder()) == order)
    return {};

  if (bankAxis < 0 || rank < 2)
    return ttg::BlockedEncodingAttr::get(tensorType.getContext(), shape,
                                         newSizePerThread, order, numWarps,
                                         threadsPerWarp, cgaLayout);

  // Explicit per-dim lane placement. The shape-driven overload above distributes
  // the lane count itself and always lands on the contiguous axis, so `order`
  // alone never moves the banks -- flipping it left threadsPerWarp at [1,32] on
  // every tile tried. This builder states the placement outright.
  SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), spt(rank, 1);
  unsigned lanes = std::min<unsigned>(threadsPerWarp,
                                      llvm::bit_floor((unsigned)shape[bankAxis]));
  tpw[bankAxis] = lanes;
  wpc[order.front() == (unsigned)bankAxis ? (rank - 1) : 0] = numWarps;
  for (int d = 0; d < rank; ++d)
    spt[d] = std::max<unsigned>(1, llvm::bit_floor((unsigned)(shape[d] /
                                                              (tpw[d] * wpc[d]))));
  return ttg::BlockedEncodingAttr::get(tensorType.getContext(), spt, tpw, wpc,
                                       order, cgaLayout);
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

    // Step 4 lever: which axis carries the banks. Absent = keep the upstream
    // choice, so this is a no-op unless a schedule asks for it.
    int bankAxis = -1;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.bank_axis")) {
      bankAxis = (int)a.getInt();
      if (bankAxis < 0) {
        mod.emitError() << "im.bank_axis must be a non-negative tensor axis";
        return signalPassFailure();
      }
    }
    mod->setAttr("im.bank-axis-requested",
                 IntegerAttr::get(IntegerType::get(mod.getContext(), 64),
                                  bankAxis));

    // A bank-axis flip is a WHOLE-KERNEL substitution, not a per-op rewrite.
    // Rewriting only the memory-op pointers leaves the loop-carried accumulator
    // on the old encoding, and reconciling the two needs a lane transpose. On a
    // GPU that materialises through shared memory; on PIM it is cross-bank data
    // movement, which the hardware cannot do and the lowering asserts on.
    if (bankAxis >= 0) {
      ttg::BlockedEncodingAttr oldEnc, newEnc;
      mod.walk([&](Operation *op) {
        if (oldEnc)
          return;
        Value ptr = getMemAccessPtr(op);
        if (!ptr)
          return;
        auto tt = dyn_cast<RankedTensorType>(ptr.getType());
        if (!tt || tt.getRank() < 2 ||
            !isa<triton::PointerType>(tt.getElementType()))
          return;
        auto be = dyn_cast_or_null<ttg::BlockedEncodingAttr>(tt.getEncoding());
        if (!be)
          return;
        auto rebuilt = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
            buildIMEncoding(tt, threadsPerWarp, numWarps, bankAxis));
        if (rebuilt && rebuilt != be) {
          oldEnc = be;
          newEnc = rebuilt;
        }
      });
      if (oldEnc && newEnc) {
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](ttg::BlockedEncodingAttr a) -> std::optional<Attribute> {
              if (a == oldEnc)
                return Attribute(newEnc);
              return std::nullopt;
            });
        replacer.recursivelyReplaceElementsIn(mod, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
        // A dense constant carries its own shaped type. The replacer rewrites the
        // result type but leaves the attribute's, so the two disagree and the
        // verifier rejects it. Rebuild the attribute against the result.
        mod.walk([&](arith::ConstantOp c) {
          auto rt = dyn_cast<RankedTensorType>(c.getType());
          auto de = dyn_cast<DenseElementsAttr>(c.getValue());
          if (!rt || !de || de.getType() == rt)
            return;
          if (!de.isSplat())
            return; // non-splat would need element reordering, not just retyping
          c.setValueAttr(DenseElementsAttr::get(rt, de.getSplatValue<Attribute>()));
        });
      }
    }

    // Walk all memory ops and compute the IM-optimal encoding.
    llvm::MapVector<Operation *, Attribute> layoutMap;

    mod.walk([&](Operation *op) {
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorType = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorType || !isa<triton::PointerType>(tensorType.getElementType()))
        return;

      Attribute newEnc = buildIMEncoding(tensorType, threadsPerWarp, numWarps, bankAxis);
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
