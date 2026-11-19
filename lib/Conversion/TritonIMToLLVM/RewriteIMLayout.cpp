/// RewriteIMLayout.cpp — Rewrite tensor layout for PIM bank-consecutive access.
///
/// For im: targets ConvertTritonToTritonGPU already emits the IM encoding, where each
/// PIM bank owns a consecutive chunk with sizePerThread = shape / (threadsPerWarp ×
/// warpsPerCTA) per dimension, lanes on the innermost axis. This pass rebuilds that
/// encoding with the banks on the axis the module states (im.bank_axis, im.bank_split,
/// or one derived from a reduce or from im.derive_bank_axis). ConvertLayoutOps are
/// inserted around load/store ops, and the downstream tritongpu-remove-layout-conversions
/// pass propagates the new encoding through the full IR.
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

/// Which axis of THIS tensor carries the banks: its first dim whose extent equals the
/// output's extent along the chosen axis. -1 when the tile cannot host the banks, which
/// chooseBankAxis reports once, or when the tensor has no such dim, and that tensor keeps
/// its layout as the replicated operand.
static int laneAxisFor(ArrayRef<int64_t> shape, int rank, int64_t bankExtent,
                       int threadsPerWarp) {
  if (bankExtent < threadsPerWarp)
    return -1; // the tile cannot host the banks at all, which chooseBankAxis reports
  // The chosen dim is named by the OUTPUT's extent along it, because an index means
  // different things to [M,K], [K,N] and [M,K,N]. A tensor without that extent does not
  // carry the dim: it is the replicated operand and keeps whatever layout it had, which
  // is what the shape-driven default already gave it.
  for (int d = 0; d < rank; ++d)
    if (shape[d] == bankExtent)
      return d;
  return -1;
}

/// The split, stated on the widest tile, mapped onto a tensor of lower rank.
///
/// A tt.reduce is the only thing that drops an axis here, and the pass refuses a reduce
/// over any axis carrying lanes, so the axis it removes is always one the split left at
/// 1. Projecting is therefore just dropping those entries: a (SPLIT_K, KS, BLOCK_M)
/// tile split "4,1,8" reduces over KS to a (SPLIT_K, BLOCK_M) store, which is "4,8".
/// Empty means this tensor is not one the split describes, and the caller falls through.
static SmallVector<int64_t> projectSplit(ArrayRef<int64_t> split, int rank) {
  if (split.empty() || rank > (int)split.size())
    return {};
  if ((int)split.size() == rank)
    return SmallVector<int64_t>(split.begin(), split.end());
  int drop = (int)split.size() - rank;
  SmallVector<int64_t> out;
  for (int64_t v : split) {
    if (v == 1 && drop > 0) {
      --drop;
      continue;
    }
    out.push_back(v);
  }
  return (int)out.size() == rank ? out : SmallVector<int64_t>{};
}

/// Compute the IM-optimal BlockedEncodingAttr for the given tensor type.
///
/// For each dimension d:
///   sizePerThread[d] = shape[d] / (threadsPerWarp[d] × warpsPerCTA[d])
///
/// This makes the "register" (per-thread) dimension contiguous so that
/// each PIM bank accesses a consecutive slice of the tensor.
static Attribute buildIMEncoding(RankedTensorType tensorType,
                                 int threadsPerWarp, int numWarps,
                                 int bankAxis, int64_t bankExtent = 0,
                                 ArrayRef<int64_t> bankSplit = {}) {
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  if (!blocked)
    return {};

  // Lanes on MORE THAN ONE axis. Everything below places all the banks on a single
  // axis, which is what a kernel reducing over the other axis needs. A split states a
  // lane count per axis instead, so a bank owns a slice of the reduction too and holds
  // only K/split[k] of the operand. That is the tiling shape DCC's tuner picks and the
  // one thing its search space expresses that ours could not.
  //
  // The cross-bank reduction this would otherwise imply is NOT solved here: the pass
  // refuses a split over any axis a tt.reduce runs on, so the partials must leave the
  // kernel and be merged by a second one. That merge is real traffic and is measured.
  SmallVector<int64_t> split = projectSplit(bankSplit, tensorType.getRank());
  if (!split.empty()) {
    auto shp = tensorType.getShape();
    int rk = tensorType.getRank();
    SmallVector<unsigned> tpw(rk, 1), wpc(rk, 1), spt(rk, 1);
    for (int d = 0; d < rk; ++d) {
      tpw[d] = (unsigned)std::max<int64_t>(1, split[d]);
      spt[d] = std::max<unsigned>(1, llvm::bit_floor((unsigned)(shp[d] / tpw[d])));
    }
    return ttg::BlockedEncodingAttr::get(
        tensorType.getContext(), spt, tpw, wpc,
        SmallVector<unsigned>(blocked.getOrder()), blocked.getCGALayout());
  }

  auto shape = tensorType.getShape();
  int rank = shape.size();

  SmallVector<unsigned> oldTPW(blocked.getThreadsPerWarp());
  SmallVector<unsigned> oldWPC(blocked.getWarpsPerCTA());
  // The memory order, which says which dim is contiguous, never where the banks are.
  SmallVector<unsigned> order(blocked.getOrder());
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

  // If nothing changed, skip. A stated bank axis that is not the fastest dim counts as a
  // change, since moving every lane there can leave sizePerThread as it was.
  bool movesLanes = bankAxis >= 0 && bankAxis < rank && rank >= 2 &&
                    order.front() != (unsigned)bankAxis;
  if (SmallVector<unsigned>(blocked.getSizePerThread()) == newSizePerThread && !movesLanes)
    return {};

  if (bankAxis < 0 || rank < 2)
    return ttg::BlockedEncodingAttr::get(tensorType.getContext(), shape,
                                         newSizePerThread, order, numWarps,
                                         threadsPerWarp, cgaLayout);

  // Explicit per-dim lane placement. The shape-driven overload above distributes the lane
  // count itself and always lands on the contiguous axis, so this states the placement
  // outright. An axis too short for the banks is refused where the choice is made
  // (chooseBankAxis), since the choice is what is wrong, not the layout.
  int laneAxis = laneAxisFor(shape, rank, bankExtent, threadsPerWarp);
  if (laneAxis < 0)
    return {}; // does not carry the chosen dim, or the tile cannot host the banks
  SmallVector<unsigned> tpw(rank, 1), wpc(rank, 1), spt(rank, 1);
  tpw[laneAxis] = (unsigned)threadsPerWarp;
  wpc[order.front() == (unsigned)laneAxis ? (rank - 1) : 0] = numWarps;
  for (int d = 0; d < rank; ++d)
    spt[d] = std::max<unsigned>(1, llvm::bit_floor((unsigned)(shape[d] /
                                                              (tpw[d] * wpc[d]))));
  return ttg::BlockedEncodingAttr::get(tensorType.getContext(), spt, tpw, wpc,
                                       order, cgaLayout);
}

/// Where the banks go: lane counts per tensor axis (`split`), or one output axis that carries
/// them all (`axis`, -1 to keep the upstream layout) with the output's extent along it.
struct BankChoice {
  SmallVector<int64_t> split;
  int axis = -1;
  int64_t extent = 0;
};

/// The placement the module states or derives. Refuses a placement that puts banks on a
/// reduced axis, since a bank cannot read another bank's data, and an axis the output tile
/// is too short to hold the banks on.
static FailureOr<BankChoice> chooseBankAxis(ModuleOp mod, int threadsPerWarp) {
  BankChoice c;
  // Per-axis lane counts. Product must be the bank count: every bank still gets
  // exactly one lane, the split only says which axes they spread over.
  if (auto a = mod->getAttrOfType<DenseI64ArrayAttr>("im.bank_split")) {
    c.split.assign(a.asArrayRef().begin(), a.asArrayRef().end());
    int64_t prod = 1;
    for (int64_t v : c.split) {
      if (v < 1)
        return mod.emitError() << "im.bank_split entries must be >= 1";
      prod *= v;
    }
    if (prod != threadsPerWarp)
      return mod.emitError() << "im.bank_split product is " << prod
                             << " but this kernel has " << threadsPerWarp
                             << " banks. Banks-as-threads gives each bank "
                             << "exactly one lane, so the split must account for all of them.";
    if (mod->hasAttr("im.bank_axis"))
      return mod.emitError() << "im.bank_split and im.bank_axis both set; the split already "
                             << "says where every lane goes. Drop im.bank_axis.";
  }

  // Which axis carries the banks, stated by the im_bank_axis compile option. Absent
  // keeps the upstream choice.
  bool userPinnedBankAxis = false;
  if (auto a = mod->getAttrOfType<IntegerAttr>("im.bank_axis")) {
    userPinnedBankAxis = true;
    c.axis = (int)a.getInt();
    if (c.axis < 0)
      return mod.emitError() << "im.bank_axis must be a non-negative tensor axis";
  }
  // A bank cannot exchange data with another bank, so a tt.reduce whose axis carries lanes
  // needs a reduction across banks that the hardware cannot perform. Triton's default layout
  // can put lanes on both axes of a 2-D tile, so the reduce axis is found here and kept free
  // of lanes.
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

  // A split over an axis something reduces IS the cross-bank reduction. Refuse it
  // here, naming the fix, rather than asserting in Casting.h three passes later.
  if (!c.split.empty() && !reduceAmbiguous && reduceAxis >= 0 &&
      reduceRank == (int)c.split.size() && c.split[reduceAxis] > 1)
    return mod.emitError()
           << "im.bank_split puts " << c.split[reduceAxis] << " banks on axis "
           << reduceAxis << ", which a tt.reduce in this kernel reduces over. A bank "
           << "cannot read another bank's data, so that reduction cannot be expressed. "
           << "Split the kernel: write the partial sums out, and reduce them in a second "
           << "kernel whose lanes sit on a different axis.";

  if (userPinnedBankAxis) {
    // THE USER'S CHOICE WINS, but a choice that cannot lower must say so here
    // rather than through an assertion 3 passes later. This is the same
    // silent-failure class the im.schedule validator exists to prevent.
    // The two indices are compared only when they live in the same space. The bank axis
    // numbers the output's axes, and a rank-3 reduce numbers [M,K,N], so when the ranks
    // differ the reduced dim is gone from the output and cannot be chosen.
    if (!reduceAmbiguous && reduceAxis >= 0 && reduceRank == storeRankForReduce &&
        c.axis == reduceAxis)
      return mod.emitError()
             << "im.bank_axis=" << c.axis << " is the reduction axis of a tt.reduce "
             << "in this kernel. Banks-as-threads puts one lane per bank, so that "
             << "would require a cross-bank reduction, which PIM hardware cannot do "
             << "and the lowering cannot express. Use axis "
             << (1 - reduceAxis) << ", or drop im.bank_axis and let the pass derive it.";
  } else if (!reduceAmbiguous && reduceAxis >= 0 && reduceRank == 2) {
    // Two reduces that disagree would make deriving a guess, so they keep the
    // upstream choice. Otherwise banks go on the non-reduced axis (rank 2 only).
    c.axis = 1 - reduceAxis;
  }

  // Nothing stated and no rank-2 reduce decided: banks on the output's outermost axis
  // when every store can hold them, so the contiguous axis stays inside a lane and its
  // loads vectorize. Other tensors take the lanes on the first dim of that extent, so
  // the rule declines when that dim is a higher-rank reduce's axis. Opt-in, since every
  // other path keeps the upstream layout.
  if (c.axis < 0 && c.split.empty() && !reduceAmbiguous &&
      mod->hasAttr("im.derive_bank_axis")) {
    bool any = false, all = true;
    int64_t extent = 0;
    mod.walk([&](triton::StoreOp st) {
      auto tt = dyn_cast<RankedTensorType>(st.getValue().getType());
      if (!tt || tt.getRank() < 2)
        return;
      any = true;
      extent = tt.getShape()[0];
      all &= tt.getShape()[0] >= threadsPerWarp;
    });
    bool hitsReduce = false;
    mod.walk([&](triton::ReduceOp red) {
      auto opnd = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!opnd || opnd.getRank() < 3)
        return;
      auto shp = opnd.getShape();
      for (int d = 0; d < opnd.getRank(); ++d)
        if (shp[d] == extent) {
          hitsReduce |= d == (int)red.getAxis();
          break;
        }
    });
    if (any && all && !hitsReduce)
      c.axis = 0;
  }

  // The choice is an index into the output's dims. Its extent there names the lane dim
  // of every other tensor (laneAxisFor), because one index means the row dim in [M,K]
  // and the reduction in [K,N].
  triton::StoreOp anchorStore;
  mod.walk([&](triton::StoreOp st) {
    auto tt = dyn_cast<RankedTensorType>(st.getValue().getType());
    if (!tt)
      return;
    anchorStore = st;
    if (c.axis >= 0 && c.axis < tt.getRank())
      c.extent = tt.getShape()[c.axis];
  });
  if (c.axis >= 0 && c.extent > 0 && c.extent < threadsPerWarp)
    // The one genuinely illegal case, reported where the choice was made rather than
    // by the verifier three passes later. The tile, not the layout, is what is wrong.
    return anchorStore->emitError()
           << "im.bank_axis=" << c.axis << " selects an output dimension of extent "
           << c.extent << ", below the " << threadsPerWarp << " banks. That axis cannot "
           << "carry the banks whatever layout is built; size the tile so it reaches the "
           << "bank count, or choose the other axis.";
  return c;
}

/// Rebuilds every distinct blocked encoding in the module for `c`, the encodings of every
/// value and block argument at once. A bank-axis flip is a whole-kernel substitution: a
/// tensor left on its old encoding, such as a loop-carried accumulator or a slice of the
/// rank-3 product, would need a lane transpose to meet the new one, which on PIM is
/// cross-bank data movement.
static void substituteEncodings(ModuleOp mod, const BankChoice &c, int threadsPerWarp,
                                int numWarps) {
  llvm::MapVector<Attribute, Attribute> encSubst;
  auto note = [&](Type ty) {
    auto tt = dyn_cast<RankedTensorType>(ty);
    if (!tt || tt.getRank() < 2)
      return;
    auto be = dyn_cast_or_null<ttg::BlockedEncodingAttr>(tt.getEncoding());
    if (!be || encSubst.count(be))
      return;
    auto rebuilt = dyn_cast_or_null<ttg::BlockedEncodingAttr>(
        buildIMEncoding(tt, threadsPerWarp, numWarps, c.axis, c.extent, c.split));
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
  if (encSubst.empty())
    return;
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
  mod.walk([&](arith::ConstantOp k) {
    auto rt = dyn_cast<RankedTensorType>(k.getType());
    auto de = dyn_cast<DenseElementsAttr>(k.getValue());
    if (!rt || !de || de.getType() == rt)
      return;
    if (!de.isSplat())
      return; // non-splat would need element reordering, not just retyping
    k.setValueAttr(DenseElementsAttr::get(rt, de.getSplatValue<Attribute>()));
  });
}

// -----------------------------------------------------------------------
// Pass implementation
// -----------------------------------------------------------------------

struct RewriteIMLayoutPass
    : public triton::im::impl::RewriteIMLayoutBase<RewriteIMLayoutPass> {
  using RewriteIMLayoutBase::RewriteIMLayoutBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int threadsPerWarp =
        ttg::TritonGPUDialect::getThreadsPerWarp(mod); // = num_banks
    int numWarps = ttg::lookupNumWarps(mod.getOperation());

    FailureOr<BankChoice> choice = chooseBankAxis(mod, threadsPerWarp);
    if (failed(choice))
      return signalPassFailure();
    if (choice->axis >= 0 || !choice->split.empty())
      substituteEncodings(mod, *choice, threadsPerWarp, numWarps);

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
                                         choice->axis, choice->extent, choice->split);
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
