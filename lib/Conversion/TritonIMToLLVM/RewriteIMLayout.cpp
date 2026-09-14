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

/// Which axis of THIS tensor carries the banks, given the choice stated on the output.
///
/// The choice cannot be a shared index. This kernel family loads A as [M,K], B as [K,N]
/// and forms [M,K,N], so index 0 is the output's row dim in one tensor and the reduction
/// in another. It is stated on the output's rank and mapped here by position: the output's
/// first dim maps to this tensor's first, its last to this tensor's last.
///
/// A tensor that does not carry the chosen dim is the REPLICATED operand, and it still
/// needs all its lanes somewhere. It gets its longest axis that can hold them, which is
/// what the shape-driven default already did for the replicated operand and is why the
/// existing layouts are valid. Returning -1 means no axis can, and the caller refuses
/// rather than emitting an under-filled layout for the verifier to reject three passes on.
static int laneAxisFor(ArrayRef<int64_t> shape, int rank, int64_t bankExtent,
                       int threadsPerWarp) {
  if (bankExtent < threadsPerWarp)
    return -1; // the tile cannot host the banks at all; the caller reports it once
  // The chosen dim is named by the OUTPUT's extent along it, because an index means
  // different things to [M,K], [K,N] and [M,K,N]. A tensor without that extent does not
  // carry the dim: it is the replicated operand and keeps whatever layout it had, which
  // is what the shape-driven default already gave it.
  for (int d = 0; d < rank; ++d)
    if (shape[d] == bankExtent)
      return d;
  return -1;
}

static Attribute buildIMEncoding(RankedTensorType tensorType,
                                 int threadsPerWarp, int numWarps,
                                 int bankAxis, int64_t bankExtent = 0) {
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
  //
  // It used to clamp the lane count to the pinned axis, which handed the verifier an
  // 8-lane layout whenever that axis was short (measured on every matmul10 shape,
  // both axes, 2026-09-13). A short axis means the CHOICE is wrong, not the layout,
  // so say so here where the choice was made.
  int laneAxis = laneAxisFor(shape, rank, bankExtent, threadsPerWarp);
  if (laneAxis < 0)
    return {}; // does not carry the chosen dim, or the tile cannot host the banks
  // KEEP THE MEMORY ORDER. `order` states which dim is contiguous, not where the banks
  // are, and the lane placement below says that outright. Moving the bank axis to the
  // front declared a row-major tensor column-major and broke the index machinery in the
  // shared lowering. The original flip existed only because the shape-driven overload
  // had no other way to move the lanes.
  order = SmallVector<unsigned>(blocked.getOrder());
  SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), spt(rank, 1);
  tpw[laneAxis] = (unsigned)threadsPerWarp;
  wpc[order.front() == (unsigned)laneAxis ? (rank - 1) : 0] = numWarps;
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
    bool userPinnedBankAxis = false;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.bank_axis")) {
      userPinnedBankAxis = true;
      bankAxis = (int)a.getInt();
      if (bankAxis < 0) {
        mod.emitError() << "im.bank_axis must be a non-negative tensor axis";
        return signalPassFailure();
      }
    }
    // REDUCE AXIS. Banks-as-threads means one lane per bank, and a PIM bank cannot
    // exchange data with another bank. So a tt.reduce whose axis carries lanes would
    // need a cross-lane reduction the hardware cannot perform, and the lowering does
    // not fail cleanly: it asserts with "dyn_cast on a non-existent value"
    // (Casting.h:656).
    //
    // Triton's default layout happily splits lanes across both axes. A blocked
    // matvec, tile (BLOCK, PACK_K), gets threadsPerWarp=[4,8], putting 8 lanes on the
    // reduction, which is why matvec_kpacked_kernel and make_2d_tiled_matvec_config
    // never compiled. A comment in the harness asserted this pass already handled it;
    // it did not. Written 2026-09-09.
    int reduceAxis = -1, reduceRank = 0;
    int storeRankForReduce = 2;
    mod.walk([&](triton::StoreOp st) {
      if (auto tt = dyn_cast<RankedTensorType>(st.getValue().getType()))
        storeRankForReduce = tt.getRank();
    });
    bool reduceAmbiguous = false;
    mod.walk([&](triton::ReduceOp red) {
      if (reduceAmbiguous || red.getOperands().empty())
        return;
      auto opnd = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!opnd || opnd.getRank() < 2)
        return;
      int axis = (int)red.getAxis();
      if (axis < 0 || axis >= opnd.getRank())
        return;
      if (reduceAxis >= 0 && reduceAxis != axis)
        reduceAmbiguous = true;
      else {
        reduceAxis = axis;
        reduceRank = opnd.getRank();
      }
    });

    if (userPinnedBankAxis) {
      // THE USER'S CHOICE WINS, but a choice that cannot lower must say so here
      // rather than through an assertion 3 passes later. This is the same
      // silent-failure class the im.schedule validator exists to prevent.
      // ONLY when the two indices live in the same space. bankAxis numbers the OUTPUT's
      // axes; a rank-3 reduce numbers [M,K,N], so its axis 1 is K while the output's is N.
      // Comparing them refused the correct default on every k-packed matmul. When the
      // ranks differ the constraint is satisfied anyway, because the reduced dim is gone
      // from the output and cannot be chosen.
      if (!reduceAmbiguous && reduceAxis >= 0 && reduceRank == storeRankForReduce &&
          bankAxis == reduceAxis) {
        mod.emitError()
            << "im.bank_axis=" << bankAxis << " is the reduction axis of a tt.reduce "
            << "in this kernel. Banks-as-threads puts one lane per bank, so that "
            << "would require a cross-bank reduction, which PIM hardware cannot do "
            << "and the lowering cannot express. Use axis "
            << (1 - reduceAxis) << ", or drop im.bank_axis and let the pass derive it.";
        return signalPassFailure();
      }
    } else if (reduceAmbiguous) {
      // Two reduces disagree. Deriving would be a guess, so leave the upstream
      // choice and record that we declined.
      mod->setAttr("im.bank-axis-ambiguous-reduce", UnitAttr::get(mod.getContext()));
    } else if (reduceAxis >= 0 && reduceRank == 2) {
      bankAxis = 1 - reduceAxis; // banks on the non-reduced axis, only meaningful at rank 2
      mod->setAttr("im.bank-axis-from-reduce",
                   IntegerAttr::get(IntegerType::get(mod.getContext(), 64), bankAxis));
    }

    // The choice is stated on the OUTPUT's rank, so every other tensor's placement is
    // mapped from it by position. Without this anchor the same index means the row dim
    // in [M,K] and the reduction in [K,N].
    int storeRank = 2;
    int64_t bankExtent = 0;
    triton::StoreOp anchorStore;
    mod.walk([&](triton::StoreOp st) {
      auto tt = dyn_cast<RankedTensorType>(st.getValue().getType());
      if (!tt)
        return;
      storeRank = tt.getRank();
      anchorStore = st;
      if (bankAxis >= 0 && bankAxis < tt.getRank())
        bankExtent = tt.getShape()[bankAxis];
    });
    if (bankAxis >= 0 && bankExtent > 0 && bankExtent < threadsPerWarp) {
      // The one genuinely illegal case, reported where the choice was made rather than
      // by the verifier three passes later. The tile, not the layout, is what is wrong.
      anchorStore->emitError()
          << "im.bank_axis=" << bankAxis << " selects an output dimension of extent "
          << bankExtent << ", below the " << threadsPerWarp << " banks. That axis cannot "
          << "carry the banks whatever layout is built; size the tile so it reaches the "
          << "bank count, or choose the other axis.";
      return signalPassFailure();
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
      // EVERY distinct encoding, not the first. The operand tiles are ttg.slice views of
      // the rank-3 layout, so a kernel carries several, and replacing one left the module
      // with lanes on N in [BM,BN] and on M in its slices at once. That mix asserts in
      // Casting.h rather than failing, which is how it read as "bank axis is blocked".
      // EVERY tensor, not only the ones behind a memory-access pointer. The rank-3
      // product in a k-packed matmul is an arithmetic value, so a pointer-only walk left
      // it on the old encoding while the store pointer moved. The reduce then produced
      // the accumulator as a slice of the UNCONVERTED layout, giving two layouts for one
      // [BM,BN] tile that disagreed about which dim holds the banks, and the lowering
      // asked for a conversion between them that does not exist. Invisible while banks
      // sat on N, because there the two already agreed.
      llvm::MapVector<Attribute, Attribute> encSubst;
      auto note = [&](Type ty) {
        auto tt = dyn_cast<RankedTensorType>(ty);
        if (!tt || tt.getRank() < 2)
          return;
        auto be = dyn_cast_or_null<ttg::BlockedEncodingAttr>(tt.getEncoding());
        if (!be || encSubst.count(be))
          return;
        auto rebuilt = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
            buildIMEncoding(tt, threadsPerWarp, numWarps, bankAxis, bankExtent));
        if (rebuilt && rebuilt != be)
          encSubst.insert({be, rebuilt});
      };
      mod.walk([&](Operation *op) {
        for (Value v : op->getOperands())
          note(v.getType());
        for (Value v : op->getResults())
          note(v.getType());
        for (Region &r : op->getRegions())
          for (Block &b : r)
            for (BlockArgument a : b.getArguments())
              note(a.getType());
      });
      if (!encSubst.empty()) {
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](ttg::BlockedEncodingAttr a) -> std::optional<Attribute> {
              auto it = encSubst.find(a);
              if (it != encSubst.end())
                return it->second;
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

      Attribute newEnc = buildIMEncoding(tensorType, threadsPerWarp, numWarps,
                                         bankAxis, bankExtent);
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
