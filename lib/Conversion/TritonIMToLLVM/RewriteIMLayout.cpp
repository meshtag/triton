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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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

/// Re-creates `v`, a pointer, mask or offset tensor of the store's shape, in `enc`, by
/// cloning its producers before `at`. Only cheap index ops are followed. Null when the
/// chain holds anything else.
static Value retypeChain(Value v, Attribute enc, Operation *at,
                         llvm::DenseMap<Value, Value> &memo) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t)
    return v;
  if (t.getEncoding() == enc)
    return v;
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  Operation *d = v.getDefiningOp();
  if (!d || d->getNumResults() != 1)
    return {};
  if (auto c = dyn_cast<ttg::ConvertLayoutOp>(d))
    return memo[v] = retypeChain(c.getSrc(), enc, at, memo);
  auto nt = RankedTensorType::get(t.getShape(), t.getElementType(), enc);
  OpBuilder b(at);
  if (auto cst = dyn_cast<arith::ConstantOp>(d)) {
    auto de = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (!de || !de.isSplat())
      return {};
    return memo[v] = arith::ConstantOp::create(
                         b, d->getLoc(), nt,
                         DenseElementsAttr::get(nt, de.getSplatValue<Attribute>()));
  }
  if (!isa<triton::AddPtrOp, triton::SplatOp, triton::MakeRangeOp>(d) &&
      d->getDialect()->getNamespace() != "arith")
    return {};
  SmallVector<Value> ops;
  for (Value o : d->getOperands()) {
    Value r = retypeChain(o, enc, at, memo);
    if (!r)
      return {};
    ops.push_back(r);
  }
  Operation *n = b.clone(*d);
  n->setOperands(ops);
  n->getResult(0).setType(nt);
  return memo[v] = n->getResult(0);
}

/// Keeps the pair reduce's sum on the reduce's own slice layout downstream, which leaves
/// lane bit 0 free. A blocked layout of the lower rank cannot, so the convert into one
/// would move the sum between lanes. Converts are dropped, in-lane reduces and
/// elementwise ops re-typed. A store keeps its operand, retyped by its owner pass.
static LogicalResult keepSliced(Value sum) {
  SmallVector<Value> work{sum};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    auto curTy = cast<RankedTensorType>(cur.getType());
    for (Operation *u : llvm::make_early_inc_range(cur.getUsers())) {
      if (auto c = dyn_cast<ttg::ConvertLayoutOp>(u)) {
        c.getResult().replaceAllUsesWith(cur);
        c.erase();
        seen.erase(cur);
        work.push_back(cur);
        continue;
      }
      if (auto r = dyn_cast<triton::ReduceOp>(u)) {
        auto parent = cast<ttg::DistributedEncodingTrait>(curTy.getEncoding());
        auto enc = ttg::SliceEncodingAttr::get(cur.getContext(), r.getAxis(), parent);
        for (Value res : r.getResults()) {
          auto rt = cast<RankedTensorType>(res.getType());
          res.setType(RankedTensorType::get(rt.getShape(), rt.getElementType(), enc));
          work.push_back(res);
        }
        continue;
      }
      if (isa<triton::StoreOp>(u))
        continue;
      StringRef dialect = u->getName().getDialectNamespace();
      if (u->getNumResults() == 1 && (dialect == "arith" || dialect == "math")) {
        auto rt = dyn_cast<RankedTensorType>(u->getResult(0).getType());
        if (!rt || rt.getShape() != curTy.getShape())
          return failure();
        llvm::DenseMap<Value, Value> memo;
        for (OpOperand &o : u->getOpOperands())
          if (o.get() != cur) {
            Value r = retypeChain(o.get(), curTy.getEncoding(), u, memo);
            if (!r)
              return failure();
            o.set(r);
          }
        u->getResult(0).setType(
            RankedTensorType::get(rt.getShape(), rt.getElementType(), curTy.getEncoding()));
        work.push_back(u->getResult(0));
        continue;
      }
      // Anything else is refused by the user walk that follows.
    }
  }
  return success();
}

/// Result i of an scf.for whose iter_arg i starts at zero and only ever has a value added to
/// it: a MAC accumulation, which GRF_B keeps from the even pass into the odd one.
static bool isGrfBAccumulation(Value v) {
  auto res = dyn_cast<OpResult>(v);
  auto loop = res ? dyn_cast<scf::ForOp>(res.getOwner()) : scf::ForOp();
  if (!loop)
    return false;
  unsigned i = res.getResultNumber();
  Value carried = loop.getRegionIterArg(i);
  Operation *upd = cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(i)
                       .getDefiningOp();
  if (!upd || !isa<arith::AddFOp, arith::AddIOp>(upd) ||
      (upd->getOperand(0) != carried && upd->getOperand(1) != carried))
    return false;
  for (Operation *u : carried.getUsers())
    if (u != upd)
      return false;
  DenseElementsAttr init;
  if (!matchPattern(loop.getInitArgs()[i], m_Constant(&init)) || !init.isSplat())
    return false;
  if (isa<FloatType>(init.getElementType()))
    return init.getSplatValue<APFloat>().isZero();
  return init.getSplatValue<APInt>().isZero();
}

static bool isSingleAdd(Region &combine) {
  Block &blk = combine.front();
  if (blk.getOperations().size() != 2)
    return false;
  Operation &op = blk.front();
  return isa<arith::AddFOp, arith::AddIOp>(op) && op.getOperand(0) == blk.getArgument(0) &&
         op.getOperand(1) == blk.getArgument(1);
}

/// im.pcu_pair_accumulate. A reduce over lane bit 0 alone, the two banks of one PCU, is
/// the PCU accumulating both passes into one GRF_B entry. Stamp it and its stores, which
/// only the odd bank performs, and refuse every other reduce over lanes.
static LogicalResult applyPcuPair(ModuleOp mod, int64_t pcuLanes) {
  MLIRContext *ctx = mod.getContext();
  auto kLane = StringAttr::get(ctx, "lane"), kReg = StringAttr::get(ctx, "register"),
       kWarp = StringAttr::get(ctx, "warp");
  Type i64 = IntegerType::get(ctx, 64);
  bool bad = false;
  SmallVector<std::pair<triton::StoreOp, int64_t>> owners;
  int64_t stamped = 0;
  auto onAxis = [](const triton::LinearLayout &ll, StringAttr dim, unsigned axis) {
    SmallVector<unsigned> bits;
    if (ll.hasInDim(dim))
      for (auto [i, basis] : llvm::enumerate(ll.getBases().lookup(dim)))
        if (basis[axis] != 0)
          bits.push_back(i);
    return bits;
  };
  // Collected first: keepSliced erases the converts after a pair reduce.
  SmallVector<triton::ReduceOp> reduces;
  mod.walk([&](triton::ReduceOp red) { reduces.push_back(red); });
  for (triton::ReduceOp red : reduces) {
    if (bad)
      break;
    [&] {
      auto ty = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!ty || !ty.getEncoding())
        return;
      unsigned axis = red.getAxis();
      triton::LinearLayout ll = ttg::toLinearLayout(ty);
      SmallVector<unsigned> lanes = onAxis(ll, kLane, axis);
      if (lanes.empty())
        return;
      auto fail = [&](const Twine &why) {
        red.emitError() << "im.pcu_pair_accumulate: this reduce runs over lanes, but " << why;
        bad = true;
      };
      if (lanes.size() != 1 || lanes[0] != 0)
        return fail("not over lane bit 0 alone, the two banks of one PCU. Put the pair "
                    "axis fastest among the lane axes in the tile's order.");
      if (ty.getShape()[axis] != pcuLanes)
        return fail("its axis is not the PCU's " + Twine(pcuLanes) + " banks");
      for (auto [d, x] : llvm::enumerate(ll.getBases().lookup(kLane)[0]))
        if (d != axis && x != 0)
          return fail("lane bit 0 also moves another axis");
      if (!onAxis(ll, kReg, axis).empty() || !onAxis(ll, kWarp, axis).empty())
        return fail("its axis also spans registers or warps");
      if (red.getNumOperands() != 1 || !isSingleAdd(red.getCombineOp()))
        return fail("its combine is not one add, the only thing a MAC into GRF_B does");
      if (ty.getElementType().getIntOrFloatBitWidth() > 32)
        return fail("its elements are wider than the carry's 32 bits");
      if (red->getParentOfType<scf::ForOp>() || red->getParentOfType<scf::WhileOp>())
        return fail("it sits in a loop, and GRF_B carries one sum per dispatch");
      Value src = red.getOperands()[0];
      while (auto c = src.getDefiningOp<ttg::ConvertLayoutOp>())
        src = c.getSrc();
      if (!isGrfBAccumulation(src))
        return fail("its operand is not a loop that adds into a zeroed accumulator, which "
                    "is the only thing GRF_B carries from one pass into the next");
      if (failed(keepSliced(red->getResult(0))))
        return fail("its sum reaches an op whose layout cannot follow it");
      // The pair's sum lives in the PCU, so only the owner may observe it.
      SmallVector<Value> work(red.getResults().begin(), red.getResults().end());
      llvm::DenseSet<Value> seen;
      SmallVector<triton::StoreOp> mine;
      while (!work.empty() && !bad) {
        Value v = work.pop_back_val();
        if (!seen.insert(v).second)
          continue;
        for (Operation *u : v.getUsers()) {
          if (auto st = dyn_cast<triton::StoreOp>(u)) {
            if (st.getValue() != v || st.getPtr() == v || st.getMask() == v)
              return fail("its sum is used as an address or mask");
            mine.push_back(st);
            continue;
          }
          if (auto r = dyn_cast<triton::ReduceOp>(u)) {
            auto rt = cast<RankedTensorType>(r.getOperands()[0].getType());
            if (!onAxis(ttg::toLinearLayout(rt), kLane, r.getAxis()).empty())
              return fail("its sum feeds a second reduce over lanes");
            work.append(r.getResults().begin(), r.getResults().end());
            continue;
          }
          StringRef dialect = u->getName().getDialectNamespace();
          if (u->getNumResults() == 1 &&
              (isa<ttg::ConvertLayoutOp, triton::ExpandDimsOp>(u) || dialect == "arith" ||
               dialect == "math")) {
            work.push_back(u->getResult(0));
            continue;
          }
          return fail("its sum reaches an op that is not elementwise, an in-lane reduce or "
                      "its store");
        }
      }
      if (bad)
        return;
      if (mine.empty())
        return fail("its sum is never stored");
      if (mine.size() > 1)
        return fail("its sum reaches more than one store, and the PCU holds one");
      // GRF_B cells the PCU carries from the even pass into the odd one.
      int64_t cells = ttg::getTotalElemsPerThread(ty);
      red->setAttr("im.pcu-pair", IntegerAttr::get(i64, cells));
      ++stamped;
      for (triton::StoreOp st : mine)
        owners.push_back({st, cells});
    }();
  }
  if (bad)
    return failure();
  if (!stamped)
    return mod.emitError() << "im.pcu_pair_accumulate is set, but no reduce sums the two "
                              "banks of a PCU";

  // Every owner store takes its value's encoding, lane bit 0 free, with its pointer and
  // mask rebuilt in it. Converting the value to a blocked pointer layout would move it
  // between lanes, since no blocked layout leaves lane bit 0 free.
  for (auto [st, cells] : owners) {
    Value val = st.getValue();
    while (auto c = val.getDefiningOp<ttg::ConvertLayoutOp>())
      val = c.getSrc();
    Attribute enc = cast<RankedTensorType>(val.getType()).getEncoding();
    llvm::DenseMap<Value, Value> memo;
    Value ptr = retypeChain(st.getPtr(), enc, st, memo);
    Value mask = st.getMask() ? retypeChain(st.getMask(), enc, st, memo) : Value();
    if (!ptr || (st.getMask() && !mask)) {
      st.emitError() << "im.pcu_pair_accumulate: this store's address is not built from "
                        "ranges, splats and arithmetic, so it cannot take its value's layout";
      return failure();
    }
    st.getPtrMutable().assign(ptr);
    st.getValueMutable().assign(val);
    if (mask)
      st.getMaskMutable().assign(mask);
    // [owner side, carried cells]. The odd bank owns it: GRF_B writes back to the odd
    // bank, and the odd pass is the last MAC into the entry.
    st->setAttr("im.pcu-owner", DenseI64ArrayAttr::get(ctx, {1, cells}));
  }

  // Only the odd bank stores, so anything traced after the owner store would sit at
  // different access ordinals on the PCU's two banks and the lockstep collapse would record
  // it twice. The owner stores are the function's last traced ops, outside any loop.
  for (auto [st, cells] : owners) {
    if (!isa<triton::FuncOp>(st->getParentOp()))
      return st.emitError() << "im.pcu_pair_accumulate: the owner store sits inside a loop "
                               "or branch, where later traced ops would follow it";
    for (Operation *n = st->getNextNode(); n; n = n->getNextNode()) {
      bool traced = false;
      n->walk([&](Operation *o) {
        traced |= isa<triton::LoadOp, triton::ReduceOp, triton::AtomicRMWOp,
                      triton::AtomicCASOp>(o) ||
                  (isa<triton::StoreOp>(o) && !o->hasAttr("im.pcu-owner"));
      });
      if (traced)
        return n->emitError() << "im.pcu_pair_accumulate: this follows an owner store, which "
                                 "only the odd bank of each PCU performs";
    }
  }
  return success();
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
    // Per-axis lane counts. Product must be the bank count: every bank still gets
    // exactly one lane, the split only says which axes they spread over.
    SmallVector<int64_t> bankSplit;
    if (auto a = mod->getAttrOfType<DenseI64ArrayAttr>("im.bank_split")) {
      bankSplit.assign(a.asArrayRef().begin(), a.asArrayRef().end());
      int64_t prod = 1;
      for (int64_t v : bankSplit) {
        if (v < 1) {
          mod.emitError() << "im.bank_split entries must be >= 1";
          return signalPassFailure();
        }
        prod *= v;
      }
      if (prod != threadsPerWarp) {
        mod.emitError() << "im.bank_split product is " << prod << " but this kernel has "
                        << threadsPerWarp << " banks. Banks-as-threads gives each bank "
                        << "exactly one lane, so the split must account for all of them.";
        return signalPassFailure();
      }
      if (mod->hasAttr("im.bank_axis")) {
        mod.emitError() << "im.bank_split and im.bank_axis both set; the split already "
                        << "says where every lane goes. Drop im.bank_axis.";
        return signalPassFailure();
      }
    }

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

    // A pair kernel reduces over its split's pair axis by design. applyPcuPair checks every
    // reduce over lanes on the final layout instead.
    int64_t pcuLanes = 0;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.pcu_lanes"))
      pcuLanes = a.getInt();
    const bool pairOpt = mod->hasAttr("im.pcu_pair_accumulate");

    // A split over an axis something reduces IS the cross-bank reduction. Refuse it
    // here, naming the fix, rather than asserting in Casting.h three passes later.
    if (!pairOpt && !bankSplit.empty() && !reduceAmbiguous && reduceAxis >= 0 &&
        reduceRank == (int)bankSplit.size() && bankSplit[reduceAxis] > 1) {
      mod.emitError()
          << "im.bank_split puts " << bankSplit[reduceAxis] << " banks on axis "
          << reduceAxis << ", which a tt.reduce in this kernel reduces over. A bank "
          << "cannot read another bank's data, so that reduction cannot be expressed. "
          << "Split the kernel: write the partial sums out, and reduce them in a second "
          << "kernel whose lanes sit on a different axis.";
      return signalPassFailure();
    }

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
    } else if (reduceAxis >= 0 && reduceRank == 2 && !(pairOpt && !bankSplit.empty())) {
      bankAxis = 1 - reduceAxis; // banks on the non-reduced axis, only meaningful at rank 2
      mod->setAttr("im.bank-axis-from-reduce",
                   IntegerAttr::get(IntegerType::get(mod.getContext(), 64), bankAxis));
    }

    // Nothing stated and no rank-2 reduce decided: banks on the output's outermost axis
    // when every store can hold them, so the contiguous axis stays inside a lane and its
    // loads vectorize. Other tensors take the lanes on the first dim of that extent, so
    // the rule declines when that dim is a higher-rank reduce's axis. Opt-in, because
    // every other path was measured on the upstream layout.
    if (bankAxis < 0 && bankSplit.empty() && !reduceAmbiguous &&
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
      if (any && all && !hitsReduce) {
        bankAxis = 0;
        mod->setAttr("im.bank-axis-from-shape",
                     IntegerAttr::get(IntegerType::get(mod.getContext(), 64), bankAxis));
      } else if (any) {
        mod->setAttr("im.bank-axis-derive-declined", UnitAttr::get(mod.getContext()));
      }
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
    if (bankAxis >= 0 || !bankSplit.empty()) {
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
            buildIMEncoding(tt, threadsPerWarp, numWarps, bankAxis, bankExtent,
                            bankSplit));
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

    if (pairOpt && failed(applyPcuPair(mod, pcuLanes)))
      return signalPassFailure();

    // Walk all memory ops and compute the IM-optimal encoding.
    llvm::MapVector<Operation *, Attribute> layoutMap;

    mod.walk([&](Operation *op) {
      if (op->hasAttr("im.pcu-owner"))
        return;
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorType = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorType || !isa<triton::PointerType>(tensorType.getElementType()))
        return;

      Attribute newEnc = buildIMEncoding(tensorType, threadsPerWarp, numWarps,
                                         bankAxis, bankExtent, bankSplit);
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
