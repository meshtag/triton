/// IMOperandResidencyLayout.cpp. The residency analysis: where the value of every memory
/// access lives across the banks. The lowering writes the stamps into the kernel's layout
/// table (IMArtifact.cpp), and every runtime places a tensor by its record.
///
/// The pass is the same for every IM target. It stamps each tensor-of-pointer load or store,
/// and each scalar load splatted across the lanes, with an `im.residency` dictionary:
///   operand_arg      the kernel pointer argument the address derives from, -1 when none
///   bank_replicated  how many banks receive a copy: 0 when the value is partitioned across
///                    the lanes, n > 1 when it is replicated over n banks, and 1, which reads
///                    as every bank, for a scalar splat or a store with no lane axis. Absent
///                    when the pass cannot tell. im.schedule can force it.
///   cell_fanout      how many output cells in one lane consume one loaded value, when more
///                    than one
///   live_split       on a store, how much wider the live accumulator is than the stored
///                    value, when more than one
///   footprint        the element offsets one program instance touches, as (extent, stride)
///                    axes over the ranges and the induction variable of a reduction loop
///                    with a constant trip count. Absent when the address is not affine in
///                    them. The runtimes read only the extents.
///   reuse_class      how the address depends on the program ids and on the reduction loop
///                    (reuseClass). Only the layout table's merge reads it, to keep a record
///                    with a real class over the StreamedElementwise catch-all.
///
/// bank_replicated follows the dims a value occupies from the access to every store it
/// reaches, and the value is partitioned only when it occupies every lane dim of each store.
/// A loaded value starts from all the dims its shape spans, so a load broadcast over a dim,
/// as its mask can force, counts as occupying that dim. The paper states the address rule,
/// under which a value whose address stays the same along a lane dim is replicated, and the
/// OptiPIM lane mask follows it by starting from the dims the address varies along.
///
/// Under im.optipim_conv, im.optipim_roles and im.optipim_hold_input the pass also stamps
/// OptiPIM's conventions on the module as im.opt_<x>, which the lowering states as
/// @__pim_opt_<x>.

#include "IMContraction.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include <utility>

// -----------------------------------------------------------------------
// TableGen pass base class
// -----------------------------------------------------------------------
namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMOPERANDRESIDENCYLAYOUT
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"

} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::im::contraction;

namespace ttg_ = mlir::triton::gpu;

namespace {

/// Mask of the dims of a STORE tile that carry lanes, several under im.bank_split. Only
/// store encodings are trusted: remove-layout-conversions has not run yet, so a load's
/// type still puts the lanes on its own axis and says nothing about its place in the tile.
static uint32_t laneAxisMask(Type ty) {
  auto tt = dyn_cast<RankedTensorType>(ty);
  if (!tt)
    return 0u;
  auto blocked = dyn_cast_or_null<ttg_::BlockedEncodingAttr>(tt.getEncoding());
  if (!blocked)
    return 0u;
  auto tpw = blocked.getThreadsPerWarp();
  uint32_t m = 0u;
  for (int d = 0; d < tt.getRank(); ++d)
    if (tpw[d] > 1 && tt.getShape()[d] > 1)
      m |= 1u << d;
  return m;
}

/// The lane id decomposed over a store tile's dims, fastest first, as [dim, lanes] pairs.
/// Every dim with tpw > 1 is in it, so the extents multiply to the lane count.
static std::optional<SmallVector<std::pair<int32_t, int32_t>>> laneAxes(Type ty) {
  auto tt = dyn_cast<RankedTensorType>(ty);
  auto blocked = tt ? dyn_cast_or_null<ttg_::BlockedEncodingAttr>(tt.getEncoding())
                    : ttg_::BlockedEncodingAttr();
  if (!blocked)
    return std::nullopt;
  SmallVector<std::pair<int32_t, int32_t>> out;
  auto tpw = blocked.getThreadsPerWarp();
  for (unsigned d : blocked.getOrder())
    if (tpw[d] > 1)
      out.push_back({(int32_t)d, (int32_t)tpw[d]});
  return out;
}

/// How many banks receive a copy of one value: the product of the lane extents over the store
/// tile's lane dims the value does not occupy, 1 when it occupies them all.
static int64_t laneReplicationFactor(Type ty, uint32_t occ) {
  auto tt = dyn_cast<RankedTensorType>(ty);
  if (!tt)
    return 1;
  auto blocked = dyn_cast_or_null<ttg_::BlockedEncodingAttr>(tt.getEncoding());
  if (!blocked)
    return 1;
  auto tpw = blocked.getThreadsPerWarp();
  int64_t f = 1;
  for (int d = 0; d < tt.getRank(); ++d)
    if (tpw[d] > 1 && tt.getShape()[d] > 1 && !((occ >> d) & 1u))
      f *= tpw[d];
  return f;
}

/// Dims of a tensor of pointers along which the address varies. A mask broadcasts a load
/// over dims its address ignores, so the load's shape alone says it occupies them. Anything
/// unrecognised occupies every dim of extent > 1.
static uint32_t addressDims(Value v, llvm::DenseMap<Value, uint32_t> &memo,
                            int depth = 0) {
  auto tt = dyn_cast<RankedTensorType>(v.getType());
  if (!tt)
    return 0u;
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  uint32_t full = 0u;
  for (int d = 0; d < tt.getRank(); ++d)
    if (tt.getShape()[d] > 1)
      full |= 1u << d;
  uint32_t occ = full;
  Operation *def = depth > 64 ? nullptr : v.getDefiningOp();
  if (!def) {
    // a block argument, or too deep to follow
  } else if (isa<triton::SplatOp>(def)) {
    occ = 0u;
  } else if (auto c = dyn_cast<arith::ConstantOp>(def)) {
    auto de = dyn_cast<DenseElementsAttr>(c.getValue());
    occ = de && de.isSplat() ? 0u : full;
  } else if (auto ed = dyn_cast<triton::ExpandDimsOp>(def)) {
    uint32_t src = addressDims(ed.getSrc(), memo, depth + 1);
    unsigned a = ed.getAxis();
    occ = (src & ((1u << a) - 1)) | ((src >> a) << (a + 1));
  } else if (isa<triton::BroadcastOp, triton::gpu::ConvertLayoutOp>(def)) {
    occ = addressDims(def->getOperand(0), memo, depth + 1) & full;
  } else if (isa<triton::AddPtrOp>(def) ||
             def->getName().getDialectNamespace() == "arith") {
    occ = 0u;
    for (Value o : def->getOperands())
      occ |= addressDims(o, memo, depth + 1);
    occ &= full;
  }
  memo[v] = occ;
  return occ;
}

/// Where a value lives across the banks, from the dims it occupies on its way to the stores
/// it reaches. A load's own type is never read for lanes (see laneAxisMask).
struct Placement {
  /// The bank_replicated stamp: n > 1 when replicated over n banks, 0 when partitioned, 1 for
  /// a store with no lane axis, which every bank holds, and -1 when undetermined.
  int banks = -1;
  /// How many output cells in one lane consume one loaded value, the most over the stores it
  /// reaches. SIMDRAM computes across the columns of a row, so an operand that feeds F cells
  /// is written into F columns.
  int64_t cellFanout = 1;
  /// The store tile's lane dims the value does not occupy, ORed over the stores it reaches.
  uint32_t replicatedLanes = 0u;
};

/// The placement of `memOp`'s value. A loaded value starts out occupying the dims its shape
/// spans with an extent above 1, within `startMask`, and its occupancy follows it through the
/// dataflow: expand_dims inserts a dim it does not occupy, and a reduce deletes one. Every
/// store it reaches gives a verdict, and replicated wins, since an operand charge never
/// under-bills. A walk that meets an op it cannot follow stops with banks -1 and what it
/// found so far.
static Placement placementOf(Operation *memOp, uint32_t startMask) {
  Placement p;
  if (auto st = dyn_cast<triton::StoreOp>(memOp)) {
    p.banks = laneAxisMask(st.getValue().getType()) != 0u ? 0 : 1;
    return p;
  }
  if (memOp->getNumResults() != 1)
    return p;
  Value v0 = memOp->getResult(0);
  auto tt0 = dyn_cast<RankedTensorType>(v0.getType());
  // A [BM,1] load is the same value in every lane along its size-1 dim, so it does not
  // occupy that dim.
  uint32_t occ0 = 0u;
  if (tt0)
    for (int d = 0; d < tt0.getRank(); ++d)
      if (tt0.getShape()[d] > 1)
        occ0 |= 1u << d;
  occ0 &= startMask;
  SmallVector<std::pair<Value, uint32_t>, 16> work{{v0, occ0}};
  llvm::DenseSet<Value> seen;
  int hops = 0;
  bool anyRep = false, anyPart = false;
  int64_t repFactor = 1;
  while (!work.empty() && hops++ < 4096) {
    auto [cur, occ] = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (Operation *u : cur.getUsers()) {
      if (auto st = dyn_cast<triton::StoreOp>(u)) {
        if (st.getValue() != cur)
          continue; // reached through the pointer or mask, not the stored value
        uint32_t laneM = laneAxisMask(st.getValue().getType());
        if (laneM == 0u)
          return p;
        if (auto axes = laneAxes(st.getValue().getType()))
          for (auto [d, n] : *axes)
            if (!((occ >> d) & 1u))
              p.replicatedLanes |= 1u << d;
        // Partitioned only if the value occupies every lane dim. Miss one and every bank
        // along it is delivered the value.
        if ((occ & laneM) == laneM) {
          anyPart = true;
        } else {
          anyRep = true;
          repFactor =
              std::max(repFactor, laneReplicationFactor(st.getValue().getType(), occ));
        }
        // The store tile's extents over the dims that are neither a lane dim nor occupied.
        auto stt = cast<RankedTensorType>(st.getValue().getType());
        int64_t f = 1;
        for (int d = 0; d < stt.getRank(); ++d)
          if (!((laneM >> d) & 1u) && !((occ >> d) & 1u))
            f *= stt.getShape()[d];
        p.cellFanout = std::max(p.cellFanout, f);
        continue;
      }
      if (auto tr = dyn_cast<triton::TransOp>(u)) {
        // result dim i is source dim order[i]
        uint32_t nocc = 0;
        auto order = tr.getOrder();
        for (unsigned i = 0; i < order.size(); ++i)
          if ((occ >> order[i]) & 1u)
            nocc |= 1u << i;
        work.push_back({tr.getResult(), nocc});
        continue;
      }
      // Ops that move values between dims in ways not tracked here.
      if (isa<triton::ReshapeOp, triton::JoinOp, triton::SplitOp, triton::CatOp>(u))
        return p;
      if (isa<triton::DotOp>(u))
        return p; // operand roles inside a dot are not tracked here
      if (auto ed = dyn_cast<triton::ExpandDimsOp>(u)) {
        unsigned a = ed.getAxis();
        uint32_t lo = occ & ((1u << a) - 1), hi = occ >> a;
        work.push_back({ed.getResult(), lo | (hi << (a + 1))});
        continue;
      }
      if (auto rd = dyn_cast<triton::ReduceOp>(u)) {
        unsigned a = rd.getAxis();
        uint32_t lo = occ & ((1u << a) - 1), hi = occ >> (a + 1);
        for (Value r : rd.getResults())
          work.push_back({r, lo | (hi << a)});
        continue;
      }
      // The accumulator is loop-carried: its last in-loop user is the yield, and the store
      // sits outside on the for's result. Both edges are crossed.
      if (auto y = dyn_cast<scf::YieldOp>(u)) {
        auto forOp = dyn_cast<scf::ForOp>(y->getParentOp());
        if (!forOp)
          continue;
        for (unsigned i = 0; i < y.getNumOperands(); ++i)
          if (y.getOperand(i) == cur) {
            work.push_back({forOp.getResult(i), occ});
            work.push_back({forOp.getRegionIterArg(i), occ});
          }
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(u)) { // cur is an init operand
        auto inits = forOp.getInitArgs();
        for (unsigned i = 0; i < inits.size(); ++i)
          if (inits[i] == cur) {
            work.push_back({forOp.getRegionIterArg(i), occ});
            work.push_back({forOp.getResult(i), occ});
          }
        continue;
      }
      if (u->getNumResults() == 1) {
        // Elementwise or layout-only: occupancy passes through. Anything else is unknown
        // and stays undetermined rather than guessed.
        StringRef dialect = u->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math" ||
            isa<triton::gpu::ConvertLayoutOp, triton::BroadcastOp, triton::SplatOp,
                triton::BitcastOp, triton::FpToFpOp>(u))
          work.push_back({u->getResult(0), occ});
        else
          return p;
      }
    }
  }
  if (anyRep)
    p.banks = (int)repFactor; // banks that receive a copy, >= 2 here
  else if (anyPart)
    p.banks = 0;
  return p;
}

// Footprint descriptor limits. The layout table record is fixed width, so
// these must match PIM_MAX_FP_AXES / PIM_MAX_STRIDE_ARGS in pim_layout_table.h.
static constexpr int64_t kMaxFpAxes = 3;
static constexpr int64_t kMaxStrideArgs = 3;

// Backward def-chain reachability: bitmask over program-id axes the address
// depends on (bit a set => address derives from tt.get_program_id with axis a).
static unsigned reachesProgramId(Value v, llvm::DenseSet<Value> &visited) {
  if (!v || !visited.insert(v).second)
    return 0;
  Operation *def = v.getDefiningOp();
  if (!def)
    return 0; // block argument (func arg, loop IV, iter arg), not a pid source
  if (auto pid = dyn_cast<triton::GetProgramIdOp>(def))
    return 1u << pid.getAxisAsInt();
  unsigned mask = 0;
  for (Value operand : def->getOperands())
    mask |= reachesProgramId(operand, visited);
  return mask;
}

// Backward def-chain reachability: does the address derive from `target`
// (used with a reduction loop's induction variable)
static bool reachesValue(Value v, Value target, llvm::DenseSet<Value> &visited) {
  if (!v)
    return false;
  if (v == target)
    return true;
  if (!visited.insert(v).second)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  for (Value operand : def->getOperands())
    if (reachesValue(operand, target, visited))
      return true;
  return false;
}

/// Compile-time value of a scalar or splatted integer, seen through integer casts.
static std::optional<int64_t> uniformIntOf(Value v) {
  return constIntThrough<triton::SplatOp, arith::ExtSIOp, arith::TruncIOp>(v);
}

/// A stride the pass could not fold to a number: `factor` times the product of
/// the kernel scalar arguments named in `args`. Conv needs this, its weight row
/// pitch is CI*R*S and all three are kernel arguments.
struct SymStride {
  int64_t factor = 1;
  SmallVector<int64_t, kMaxStrideArgs> args;
  bool ok = true;
};

/// One footprint axis: the address takes `extent` positions `stride` apart.
struct FpAxis {
  int64_t extent = 0;
  SymStride stride;
};

/// Fold `v` into a SymStride, or give up. Only products of constants and kernel
/// scalar arguments are foldable, which is what index arithmetic on shapes
/// looks like.
static bool foldStride(Value v, SymStride &out, int depth = 0) {
  if (!v || depth > 16)
    return false;
  if (auto k = uniformIntOf(v)) {
    out.factor *= *k;
    return true;
  }
  if (auto sp = v.getDefiningOp<triton::SplatOp>())
    return foldStride(sp.getSrc(), out, depth + 1);
  if (auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>())
    return foldStride(cvt.getSrc(), out, depth + 1);
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>())
    return foldStride(ext.getIn(), out, depth + 1);
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
    return foldStride(trunc.getIn(), out, depth + 1);
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (!isa<triton::FuncOp>(barg.getOwner()->getParentOp()))
      return false;
    if ((int64_t)out.args.size() >= kMaxStrideArgs)
      return false;
    out.args.push_back(barg.getArgNumber());
    return true;
  }
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    return foldStride(mul.getLhs(), out, depth + 1) &&
           foldStride(mul.getRhs(), out, depth + 1);
  return false;
}

/// Accumulator keyed by the varying value (a tl.arange or the reduction IV) so
/// a variable appearing twice in one address adds its strides instead of
/// counting as two axes.
using FpAcc = llvm::MapVector<Value, FpAxis>;

/// Decompose an element offset into `const + sum stride_i * v_i` and record the
/// varying part. `scale` carries the enclosing multiplications down. A false return
/// means the address is not modelled and its operand gets no footprint.
///
/// `key` names the axis. It is the outermost layout-changing op the walk went
/// through, because layout assignment CSEs one tl.arange into several tile
/// dimensions and the range value alone cannot tell them apart. Null until the
/// walk passes such an op, so a range used twice in one index expression still
/// merges into a single axis.
static bool collectAxes(Value v, Value iv, int64_t ivTrip, int64_t ivStep,
                        const SymStride &scale, FpAcc &acc, int depth,
                        Value key) {
  if (!v || depth > 32)
    return false;
  if (uniformIntOf(v))
    return true; // shifts the base, does not spread it
  auto note = [&](Value var, int64_t extent, int64_t stepScale) {
    SymStride s = scale;
    s.factor *= stepScale;
    if (key)
      var = key;
    auto it = acc.find(var);
    if (it == acc.end()) {
      acc.insert({var, FpAxis{extent, s}});
      return;
    }
    // Same variable twice. Adding symbolic strides is not representable, so
    // keep the axis and mark it unusable rather than silently under-counting.
    if (it->second.stride.args == s.args && it->second.stride.args.empty())
      it->second.stride.factor += s.factor;
    else
      it->second.stride.ok = false;
  };
  if (iv && v == iv) {
    if (ivTrip <= 0)
      return false;
    note(v, ivTrip, ivStep);
    return true;
  }
  if (auto barg = dyn_cast<BlockArgument>(v))
    // A kernel scalar argument is fixed for the launch. Any other block arg is
    // loop-carried, which this does not model.
    return isa<triton::FuncOp>(barg.getOwner()->getParentOp());
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<triton::GetProgramIdOp>(def))
    return true; // fixed within one program instance
  if (auto mr = dyn_cast<triton::MakeRangeOp>(def)) {
    int64_t n = (int64_t)mr.getEnd() - (int64_t)mr.getStart();
    if (n <= 0)
      return false;
    note(v, n, 1);
    return true;
  }
  // ttg.convert_layout only moves values between layouts, so it is a pass-through here.
  if (isa<triton::BroadcastOp, triton::ExpandDimsOp,
          triton::gpu::ConvertLayoutOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key ? key : v);
  if (isa<triton::SplatOp, arith::ExtSIOp, arith::TruncIOp,
          arith::IndexCastOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key);
  if (isa<arith::AddIOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key) &&
           collectAxes(def->getOperand(1), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key);
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    for (int i = 0; i < 2; ++i) {
      SymStride sub = scale;
      if (foldStride(def->getOperand(i), sub))
        return collectAxes(def->getOperand(1 - i), iv, ivTrip, ivStep, sub, acc,
                           depth + 1, key);
    }
    return false;
  }
  return false;
}

/// The widest loop-carried tile the stored value comes from, divided by the stored tile,
/// which is the factor a rank-collapsing reduce hid from the store's footprint. It only
/// scales the occupancy the footprint gives, so a kernel with no such reduce gets 1.
/// Chase the stored value through every operand, since a reduction kernel stores after its
/// loop and an epilogue can combine the accumulator with other values. Take the max over
/// loops, never the sum, and skip pointer carries.
static int64_t liveSplitFactor(triton::StoreOp st) {
  auto storedTy = dyn_cast<RankedTensorType>(st.getValue().getType());
  if (!storedTy)
    return 1;
  int64_t storedCells = 1;
  for (int64_t d : storedTy.getShape())
    storedCells *= d;
  if (storedCells <= 0)
    return 1;

  SmallVector<Value, 16> work{st.getValue()};
  DenseSet<Value> seen;
  int64_t best = storedCells;
  int hops = 0;
  while (!work.empty() && hops++ < 256) {
    Value v = work.pop_back_val();
    if (!v || !seen.insert(v).second)
      continue;
    auto res = dyn_cast<OpResult>(v);
    if (!res)
      continue; // block argument: this branch ends
    Operation *def = res.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      BlockArgument arg = forOp.getRegionIterArg(res.getResultNumber());
      auto rt = dyn_cast<RankedTensorType>(arg.getType());
      if (!rt)
        continue;
      if (isa<triton::PointerType>(rt.getElementType()))
        continue; // a carried pointer tile is not an accumulator
      int64_t cells = 1;
      for (int64_t d : rt.getShape())
        cells *= d;
      if (cells > best)
        best = cells; // widest wins; NEVER accumulate
      continue;       // do not descend into the loop body
    }
    for (Value o : def->getOperands())
      work.push_back(o);
  }
  return best / storedCells;
}

/// Cells a loop nest keeps live in fresh tiles: a loop's float carries whose initial value
/// is not an enclosing loop's carry (that one is the same registers), plus the widest of its
/// own nests. A block loop carrying a running sum around a k loop holds both tiles.
template <typename PerLane>
static int64_t freshNestCells(scf::ForOp loop, PerLane &perLane) {
  int64_t cells = 0;
  for (auto [init, arg] : llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs())) {
    auto rt = dyn_cast<RankedTensorType>(arg.getType());
    if (!rt || !isa<FloatType>(rt.getElementType()))
      continue;
    auto ba = dyn_cast<BlockArgument>(init);
    bool threaded = ba && isa<scf::ForOp>(ba.getOwner()->getParentOp());
    if (!threaded)
      cells += perLane(rt);
  }
  int64_t inner = 0;
  loop.getBody()->walk<WalkOrder::PreOrder>([&](scf::ForOp child) {
    inner = std::max(inner, freshNestCells(child, perLane));
    return WalkResult::skip();
  });
  return cells + inner;
}

/// Cells one lane holds of the accumulator a store drains, 0 when nothing accumulates. The
/// register file keeps them live across a reduction, so a GRF bound compares against this.
/// Every tile one loop carries is live at once, so a loop counts the sum of its carried
/// tiles, and the widest loop wins. A reduce with no carry, such as tl.sum over a whole tile,
/// keeps the stored tile live instead. An elementwise store neither carries nor reduces and
/// reports 0.
static int64_t accCellsPerLane(triton::StoreOp st) {
  auto storedTy = dyn_cast<RankedTensorType>(st.getValue().getType());
  if (!storedTy)
    return 0;
  // Lanes come off the carried tensor's own encoding; the stored type stands in when
  // the carried value has none, and no layout anywhere means one lane holds it all.
  auto perLane = [&](RankedTensorType rt) -> int64_t {
    auto blocked = dyn_cast_or_null<ttg_::BlockedEncodingAttr>(rt.getEncoding());
    if (!blocked)
      blocked = dyn_cast_or_null<ttg_::BlockedEncodingAttr>(storedTy.getEncoding());
    int64_t cells = 1;
    for (int d = 0; d < rt.getRank(); ++d) {
      int64_t tpw = 1;
      if (blocked && d < (int)blocked.getThreadsPerWarp().size())
        tpw = blocked.getThreadsPerWarp()[d];
      cells *= (rt.getShape()[d] + tpw - 1) / tpw;
    }
    return cells;
  };
  SmallVector<Value, 16> work{st.getValue()};
  DenseSet<Value> seen;
  int64_t best = 0;
  int hops = 0;
  while (!work.empty() && hops++ < 256) {
    Value v = work.pop_back_val();
    if (!v || !seen.insert(v).second)
      continue;
    auto res = dyn_cast<OpResult>(v);
    if (!res)
      continue;
    Operation *def = res.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      // Float carries only, and the one reaching the store: an integer offset carried
      // beside the accumulator lives in the host's registers, not GRF_B.
      int64_t live = 0;
      Value stored = forOp.getRegionIterArg(res.getResultNumber());
      for (Value arg : forOp.getRegionIterArgs()) {
        auto rt = dyn_cast<RankedTensorType>(arg.getType());
        if (rt && !isa<triton::PointerType>(rt.getElementType()) &&
            (isa<FloatType>(rt.getElementType()) || arg == stored))
          live += perLane(rt);
      }
      int64_t nested = 0;
      forOp.getBody()->walk<WalkOrder::PreOrder>([&](scf::ForOp child) {
        nested = std::max(nested, freshNestCells(child, perLane));
        return WalkResult::skip();
      });
      live += nested;
      if (live > best)
        best = live;
      continue;
    }
    for (Value o : def->getOperands())
      work.push_back(o);
  }
  if (best > 0)
    return best;
  // No carry. If a reduce feeds the store, the stored tile is the accumulator.
  SmallVector<Value, 16> w2{st.getValue()};
  DenseSet<Value> seen2;
  hops = 0;
  while (!w2.empty() && hops++ < 256) {
    Value v = w2.pop_back_val();
    if (!v || !seen2.insert(v).second)
      continue;
    auto res = dyn_cast<OpResult>(v);
    if (!res)
      continue;
    Operation *def = res.getOwner();
    if (isa<triton::ReduceOp>(def))
      return perLane(storedTy);
    if (isa<scf::ForOp>(def))
      continue;
    for (Value o : def->getOperands())
      w2.push_back(o);
  }
  return 0;
}

/// Footprint of the whole address: walk the addptr chain back to the kernel
/// pointer argument, summing every offset on the way.
///
/// The result is the set of element offsets one program instance touches. The pass
/// stops at the logical description, and the runtimes read only its extents.
static bool operandFootprint(Value ptr, scf::ForOp redFor,
                             SmallVectorImpl<FpAxis> &out) {
  Value iv = redFor ? redFor.getInductionVar() : Value();
  int64_t ivTrip = 0, ivStep = 1;
  if (redFor) {
    auto trip = constTripCount(redFor);
    if (!trip || *trip == 0)
      return false;
    ivTrip = *trip;
    ivStep = *getConstantIntValue(redFor.getStep());
  }
  FpAcc acc;
  Value cur = ptr;
  llvm::DenseSet<Value> guard;
  while (cur && guard.insert(cur).second) {
    if (isa<BlockArgument>(cur))
      break;
    Operation *def = cur.getDefiningOp();
    if (!def)
      return false;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      SymStride unit;
      if (!collectAxes(addptr.getOffset(), iv, ivTrip, ivStep, unit, acc, 0,
                       Value()))
        return false;
      cur = addptr.getPtr();
      continue;
    }
    if (isa<triton::SplatOp, triton::BroadcastOp>(def)) {
      cur = def->getOperand(0);
      continue;
    }
    return false;
  }
  for (const auto &kv : acc) {
    const FpAxis &ax = kv.second;
    if (!ax.stride.ok)
      return false;
    if (ax.extent > 1 && (ax.stride.factor != 0 || !ax.stride.args.empty()))
      out.push_back(ax);
  }
  if ((int64_t)out.size() > kMaxFpAxes)
    return false;
  return true;
}

// Is `op` a scalar tt.load whose (scalar) result feeds a tt.splat?  This is the
// matvec `x_val = tl.load(x + k)` operand, invisible to the tensor-pointer
// walk because its pointer is a plain !tt.ptr, not a tensor of pointers.
static bool isScalarSplatLoad(Operation *op) {
  auto load = dyn_cast<triton::LoadOp>(op);
  if (!load)
    return false;
  if (isa<RankedTensorType>(load.getType()))
    return false; // tensor load handled by the tensor-pointer path
  for (Operation *user : load->getUsers())
    if (isa<triton::SplatOp>(user))
      return true;
  return false;
}

/// Whether `ptr` is a tensor of pointers, an access the pass stamps whatever its users.
static bool isTensorOfPointers(Value ptr) {
  auto tt = dyn_cast<RankedTensorType>(ptr.getType());
  return tt && isa<triton::PointerType>(tt.getElementType());
}

// -----------------------------------------------------------------------
// User-expressible schedule (roadmap step 4)
// -----------------------------------------------------------------------
/// The classification below is a DEFAULT. A schedule forces policy on top of it,
/// from `im.schedule` on the module (keyed `arg<N>` by kernel operand index).
/// Input lives in its own attribute so re-running the pass cannot mistake its own
/// output for user intent.
///
/// Only policy is forcible. operand_arg, footprint and the other stamped facts are
/// read out of the IR, and forcing those would misdescribe the kernel to the
/// runtime rather than remap it.
///
/// Every rejection is a hard error, including a schedule entry that never matched
/// an access. A silently dropped field yields a run that looks tuned and measures
/// the default.

namespace {

struct OvrSpec {
  const char *name;
  /// Whether emitPimLayoutTable carries this field into a record word. Forcing one
  /// that does not is a compile error.
  bool travels;
};

static const OvrSpec kForcible[] = {
    // reuse_class reaches the artifact only through emitPimLayoutTable's merge rule,
    // which picks the record kept when several accesses share an operand.
    {"reuse_class", false},
    {"bank_replicated", true},
};

static const OvrSpec *findForcible(StringRef name) {
  for (const OvrSpec &s : kForcible)
    if (name == s.name)
      return &s;
  return nullptr;
}

static void setField(OpBuilder &b, SmallVectorImpl<NamedAttribute> &fields,
                     StringRef name, Attribute val) {
  for (NamedAttribute &f : fields)
    if (f.getName() == name) {
      f.setValue(val);
      return;
    }
  fields.push_back(b.getNamedAttr(name, val));
}

static LogicalResult checkSchedule(DictionaryAttr d, Location loc,
                                   const Twine &where) {
  for (NamedAttribute e : d) {
    const OvrSpec *s = findForcible(e.getName());
    if (!s)
      return emitError(loc) << where << ": '" << e.getName().strref()
                            << "' is not a forcible schedule field";
    if (!s->travels)
      return emitError(loc) << where << ": '" << e.getName().strref()
                            << "' is classified but not carried into the "
                               "artifact, so forcing it would change no "
                               "simulated behaviour. Drop it, or teach "
                               "emitPimLayoutTable to carry it";
    // Every field that travels takes a bool.
    if (!isa<BoolAttr>(e.getValue()))
      return emitError(loc) << where << ": '" << e.getName().strref()
                            << "' has the wrong type for a schedule field";
  }
  return success();
}

/// Overlay `d` on `fields`. checkSchedule admits only bank_replicated, as a bool.
static void applySchedule(OpBuilder &b, DictionaryAttr d,
                          SmallVectorImpl<NamedAttribute> &fields) {
  // The classifier writes bank_replicated as a receiver count, so a forced bool is
  // stamped the same way, true as every bank.
  for (NamedAttribute e : d)
    setField(b, fields, e.getName(),
             b.getI32IntegerAttr(cast<BoolAttr>(e.getValue()).getValue() ? 1 : 0));
}

} // namespace

// -----------------------------------------------------------------------
// Stamping one access
// -----------------------------------------------------------------------

/// im.schedule by kernel argument, checked before any access is classified, so a malformed
/// schedule fails before the pass stamps anything.
static FailureOr<llvm::MapVector<int64_t, DictionaryAttr>> readSchedule(ModuleOp mod) {
  llvm::MapVector<int64_t, DictionaryAttr> out;
  auto schedule = mod->getAttrOfType<DictionaryAttr>("im.schedule");
  if (!schedule)
    return out;
  for (NamedAttribute e : schedule) {
    StringRef k = e.getName().strref();
    int64_t argIdx = 0;
    if (!k.starts_with("arg") || k.drop_front(3).getAsInteger(10, argIdx))
      return mod.emitError() << "im.schedule: key '" << k
                             << "' must be arg<N> with N a kernel operand index";
    auto d = dyn_cast<DictionaryAttr>(e.getValue());
    if (!d)
      return mod.emitError() << "im.schedule: '" << k
                             << "' must map to a dictionary of schedule fields";
    if (failed(checkSchedule(d, mod.getLoc(), "im.schedule." + k)))
      return failure();
    out[argIdx] = d;
  }
  return out;
}

/// The reuse class of an access. In a function with a reduction loop, an scf.for that
/// carries a value, it follows from whether the address reaches a program id (`pidMask`) and
/// the induction variable of the reduction loop around the access (`ivDep`):
///   both       ReductionStridedMatrix
///   iv only    BroadcastReplicate
///   pid only   ParallelSpread
///   neither    StreamedElementwise
/// Every access of a function without one is StreamedElementwise, and a scalar splatted
/// across the lanes is BroadcastReplicate anywhere.
static StringRef reuseClass(bool scalarSplat, bool reductionFunc, bool ivDep,
                            unsigned pidMask) {
  if (scalarSplat)
    return "BroadcastReplicate";
  if (!reductionFunc)
    return "StreamedElementwise";
  if (ivDep && pidMask != 0)
    return "ReductionStridedMatrix";
  if (ivDep)
    return "BroadcastReplicate";
  if (pidMask != 0)
    return "ParallelSpread";
  return "StreamedElementwise";
}

/// What the pass reads off the module before it stamps an access.
struct StampContext {
  /// The functions that hold a reduction loop.
  llvm::DenseSet<Operation *> reductionFuncs;
  /// im.schedule by kernel argument.
  llvm::MapVector<int64_t, DictionaryAttr> schedule;
};

/// The im.residency stamp of one access, null for an op that is none, and failure for a store
/// the schedule forces replicated.
static FailureOr<DictionaryAttr> stampResidency(Operation *op, const StampContext &cx,
                                                OpBuilder &b) {
  Value ptr = getMemAccessPtr(op);
  if (!ptr)
    return DictionaryAttr();
  bool isTensorPtr = isTensorOfPointers(ptr);
  bool scalarSplat = !isTensorPtr && isScalarSplatLoad(op);
  if (!isTensorPtr && !scalarSplat)
    return DictionaryAttr();

  // The reduction loop around the access, the innermost scf.for that carries a value.
  scf::ForOp redFor;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p); f && f.getNumRegionIterArgs() > 0) {
      redFor = f;
      break;
    }
  llvm::DenseSet<Value> pidSeen, ivSeen;
  unsigned pidMask = reachesProgramId(ptr, pidSeen);
  bool ivDep = redFor && reachesValue(ptr, redFor.getInductionVar(), ivSeen);
  int64_t operandArg = kernelArgOf(ptr, /*throughReshape=*/false);
  auto fn = op->getParentOfType<triton::FuncOp>();
  bool reductionFunc = fn && cx.reductionFuncs.contains(fn.getOperation());

  // The verdict is read off the lane layout rather than asserted by the host, and a scalar
  // splatted across the lanes is in every bank.
  Placement place;
  if (scalarSplat)
    place.banks = 1;
  else
    place = placementOf(op, ~0u);

  SmallVector<NamedAttribute> fields;
  fields.push_back(b.getNamedAttr(
      "reuse_class", b.getStringAttr(reuseClass(scalarSplat, reductionFunc, ivDep, pidMask))));
  // Omitted when undetermined. The layout record then says -1, and the runtime keeps the
  // host's role.
  if (place.banks >= 0)
    fields.push_back(b.getNamedAttr("bank_replicated", b.getI32IntegerAttr(place.banks)));
  // Only a broadcast inside the tile counts, since the runtime counts sharing that reaches
  // memory from the loads themselves. Only a partitioned tensor is charged by it.
  if (place.cellFanout > 1)
    fields.push_back(b.getNamedAttr("cell_fanout", b.getI64IntegerAttr(place.cellFanout)));
  fields.push_back(b.getNamedAttr("operand_arg", b.getI64IntegerAttr(operandArg)));
  // Five words per axis, [extent, stride factor, arg, arg, arg] with -1 padding.
  SmallVector<FpAxis> fp;
  if (operandFootprint(ptr, redFor, fp) && !fp.empty()) {
    SmallVector<int64_t> flat;
    for (const FpAxis &ax : fp) {
      flat.push_back(ax.extent);
      flat.push_back(ax.stride.factor);
      for (int64_t i = 0; i < kMaxStrideArgs; ++i)
        flat.push_back(i < (int64_t)ax.stride.args.size() ? ax.stride.args[i] : -1);
    }
    fields.push_back(b.getNamedAttr("footprint", b.getDenseI64ArrayAttr(flat)));
  }
  if (auto st = dyn_cast<triton::StoreOp>(op))
    if (int64_t ls = liveSplitFactor(st); ls > 1)
      fields.push_back(b.getNamedAttr("live_split", b.getI64IntegerAttr(ls)));

  // The schedule's fields replace the classifier's for the arguments it names.
  auto forced = operandArg >= 0 ? cx.schedule.find(operandArg) : cx.schedule.end();
  if (forced != cx.schedule.end()) {
    applySchedule(b, forced->second, fields);
    setField(b, fields, "user_forced", b.getUnitAttr());
    // Lanes store distinct elements, so a stored tensor cannot have identical copies.
    if (isa<triton::StoreOp>(op))
      for (NamedAttribute f : fields)
        if (f.getName() == "bank_replicated")
          if (auto n = dyn_cast<IntegerAttr>(f.getValue()); n && n.getInt() != 0)
            return op->emitError() << "a stored tensor cannot be forced bank_replicated: each "
                                   << "lane stores its own elements";
  }
  return b.getDictionaryAttr(fields);
}

// -----------------------------------------------------------------------
// OptiPIM's conventions
// -----------------------------------------------------------------------

/// OptiPIM's lanes under im.optipim_conv. im.opt_lane_rep holds [arg, replicated lane dims]
/// pairs by the address rule of the file header, and im.opt_lane_ext the store's lane id as
/// [dim, lanes] pairs, fastest first. Dims number the store tile. A scalar splat is
/// replicated along every lane dim, and an access the pass could not place, or whose verdict
/// the schedule forced, states none.
static LogicalResult stampOptiPimLanes(ModuleOp mod) {
  std::optional<SmallVector<std::pair<int32_t, int32_t>>> ext;
  LogicalResult ok = success();
  mod.walk([&](triton::StoreOp st) {
    auto axes = laneAxes(st.getValue().getType());
    if (!axes || failed(ok))
      return;
    if (!ext)
      ext = axes;
    else if (*ext != *axes)
      ok = st.emitError() << "im.optipim_conv states one lane decomposition, and this "
                             "store's differs from the kernel's first store";
  });
  if (failed(ok))
    return failure();
  auto lanes = ext.value_or(SmallVector<std::pair<int32_t, int32_t>>{});
  uint32_t all = 0u;
  SmallVector<int64_t> extFlat;
  for (auto [d, n] : lanes) {
    all |= 1u << d;
    extFlat.append({d, n});
  }
  // arg -> replicated lane dims, -1 for a scalar splat.
  llvm::MapVector<int64_t, int64_t> rep;
  mod.walk([&](Operation *op) {
    auto stamp = dyn_cast_or_null<DictionaryAttr>(op->getDiscardableAttr("im.residency"));
    if (!stamp || stamp.contains("user_forced") || !stamp.contains("bank_replicated"))
      return;
    int64_t arg = stamp.getAs<IntegerAttr>("operand_arg").getInt();
    if (arg < 0)
      return;
    Value ptr = getMemAccessPtr(op);
    bool scalarSplat = !isTensorOfPointers(ptr);
    uint32_t repLanes = 0u;
    if (!scalarSplat && isa<triton::LoadOp>(op)) {
      llvm::DenseMap<Value, uint32_t> memo;
      repLanes = placementOf(op, addressDims(ptr, memo)).replicatedLanes;
    }
    int64_t &m = rep.insert({arg, 0}).first->second;
    m = (scalarSplat || m < 0) ? -1 : (m | repLanes);
  });
  SmallVector<int64_t> repFlat;
  for (auto [arg, m] : rep)
    repFlat.append({arg, m < 0 ? (int64_t)all : m});
  Builder b(mod.getContext());
  mod->setAttr("im.opt_lane_rep", b.getDenseI64ArrayAttr(repFlat));
  mod->setAttr("im.opt_lane_ext", b.getDenseI64ArrayAttr(extFlat));
  // im.opt_lane_on, the active lanes of each lane axis where their mapping leaves some idle,
  // as [dim, active] pairs in im.opt_lane_ext's order. im.bank_active numbers im.bank_split's
  // dims, and the store tile's lane dims are those dims less the reduced one, never a lane
  // axis, so the two pair up in dim order.
  auto activeAttr = mod->getAttrOfType<DenseI64ArrayAttr>("im.bank_active");
  auto splitAttr = mod->getAttrOfType<DenseI64ArrayAttr>("im.bank_split");
  if (!activeAttr)
    return success();
  if (!splitAttr || splitAttr.size() != activeAttr.size())
    return mod.emitError() << "im.bank_active names the active lanes of im.bank_split, which "
                              "this module lacks or sizes differently";
  ArrayRef<int64_t> active = activeAttr.asArrayRef(), split = splitAttr.asArrayRef();
  SmallVector<int64_t> splitDims;
  for (int64_t d = 0; d < (int64_t)split.size(); ++d)
    if (split[d] > 1)
      splitDims.push_back(d);
  SmallVector<int32_t> storeDims;
  for (auto [d, n] : lanes)
    storeDims.push_back(d);
  llvm::sort(storeDims);
  if (storeDims.size() != splitDims.size())
    return mod.emitError() << "im.bank_active: the store tile has " << storeDims.size()
                           << " lane axes and im.bank_split " << splitDims.size();
  SmallVector<int64_t> onFlat;
  for (auto [d, n] : lanes) {
    int64_t p = splitDims[llvm::find(storeDims, d) - storeDims.begin()];
    if (split[p] != n || active[p] < 1 || active[p] > n)
      return mod.emitError() << "im.bank_active gives " << active[p] << " active lanes on "
                             << "split dim " << p << " of " << split[p] << ", stored on dim " << d
                             << " with " << n;
    onFlat.append({d, active[p]});
  }
  mod->setAttr("im.opt_lane_on", b.getDenseI64ArrayAttr(onFlat));
  return success();
}

/// Residency by place in the kernel's one contraction (im.optipim_roles): the product's
/// first factor is delivered to the PE (1), its second is resident in the bank (2), and
/// the stored sum is drained by read-out (3). Their codegen fixes roles by position too.
static FailureOr<llvm::MapVector<int64_t, int64_t>> optiPimResidency(ModuleOp mod) {
  FailureOr<Contraction> c = findOneContraction(mod, "im.optipim_roles states residency for");
  if (failed(c))
    return failure();
  auto argOf = [](Value ptr) { return kernelArgOf(ptr, /*throughReshape=*/false); };
  int64_t inArg = c->input ? argOf(c->input.getPtr()) : -1;
  int64_t wtArg = c->weight ? argOf(c->weight.getPtr()) : -1;
  triton::StoreOp out = storeOf(c->reduce);
  int64_t outArg = out ? argOf(out.getPtr()) : -1;
  if (inArg < 0 || wtArg < 0 || outArg < 0 || inArg == wtArg || outArg == inArg ||
      outArg == wtArg)
    return mod.emitError() << "im.optipim_roles cannot name the contraction's operands: "
                           << "input arg " << inArg << ", weight arg " << wtArg
                           << ", output arg " << outArg;
  llvm::MapVector<int64_t, int64_t> res;
  res.insert({inArg, 1});
  res.insert({wtArg, 2});
  res.insert({outArg, 3});
  return res;
}

/// The program ids `v` derives from, each value on the way added to `seen`. Unlike
/// reachesProgramId it fails where a def chain hides a dependence: a non-kernel block
/// argument (loop induction variable, iter_arg), a region op's result or a memory read.
static FailureOr<unsigned> pidDependence(Value v, llvm::DenseSet<Value> &seen) {
  unsigned mask = 0;
  SmallVector<Value> work{v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    if (auto barg = dyn_cast<BlockArgument>(cur)) {
      Block *owner = barg.getOwner();
      if (owner->isEntryBlock() && isa<triton::FuncOp>(owner->getParentOp()))
        continue;
      return emitError(cur.getLoc()) << "im.optipim_hold_input follows the input's "
                                        "address through def chains, and it meets a loop "
                                        "or region block argument";
    }
    Operation *def = cur.getDefiningOp();
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(def)) {
      mask |= 1u << pid.getAxisAsInt();
      continue;
    }
    if (def->getNumRegions() || !isMemoryEffectFree(def))
      return def->emitError() << "im.optipim_hold_input follows the input's address "
                                 "through def chains, and it meets "
                              << (def->getNumRegions() ? "a region op's result"
                                                       : "a read of memory");
    llvm::append_range(work, def->getOperands());
  }
  return mask;
}

/// OptiPIM's unbounded input memo, THEIR bug reproduced on purpose and disclosed: their
/// codegen never re-sends an input element the leader once held, though their PE declares
/// reg_per_pe 1. Record [arg, held ids, bound x, y, z (-1 none)].
static FailureOr<SmallVector<int64_t>>
optiPimHold(ModuleOp mod, const llvm::MapVector<int64_t, int64_t> &res) {
  int64_t inArg = -1;
  for (auto [arg, r] : res)
    if (r == 1)
      inArg = arg;
  auto pids = programIds(mod);
  if (!pids)
    return mod.emitError() << "im.optipim_hold_input reads each program id once";
  // What the input delivers is its address, mask and other. A load in a loop or branch can
  // run a pid-dependent number of times, which a hold per program instance cannot state.
  SmallVector<Value> sources;
  bool stored = false, nested = false;
  mod.walk([&](Operation *op) {
    Value ptr = getMemAccessPtr(op);
    if (!ptr || inArg < 0 || kernelArgOf(ptr, /*throughReshape=*/false) != inArg)
      return;
    auto ld = dyn_cast<triton::LoadOp>(op);
    stored |= !ld;
    nested |= !isa<triton::FuncOp>(op->getParentOp());
    sources.push_back(ptr);
    if (ld && ld.getMask())
      sources.push_back(ld.getMask());
    if (ld && ld.getOther())
      sources.push_back(ld.getOther());
  });
  if (sources.empty() || stored)
    return mod.emitError() << "im.optipim_hold_input holds an input the kernel only loads, "
                              "and arg " << inArg << " is "
                           << (stored ? "stored to" : "never loaded");
  llvm::DenseSet<Value> seen;
  unsigned reach = 0u, used = 0u;
  for (Value s : sources) {
    auto m = pidDependence(s, seen);
    if (failed(m))
      return failure();
    reach |= *m;
  }
  if (nested)
    return mod.emitError() << "im.optipim_hold_input holds an input loaded once per program "
                              "instance, and arg " << inArg << " is loaded in a loop or branch";
  for (int a = 0; a < 3; ++a)
    if ((*pids)[a])
      used |= 1u << a;
  unsigned held = used & ~reach;
  // A stated hash (im.optipim_memo) can also hold steps the input reaches: their misread
  // strides alias them, on a gemm too.
  if (!held && !mod->hasAttr("im.optipim_memo"))
    return mod.emitError() << "im.optipim_hold_input holds the input across the program ids "
                              "it ignores, and arg " << inArg
                           << " reaches every program id the kernel reads";
  // Their memo needs the statement only across ids slower than all the input reaches, as
  // conv's K. Held on a faster id, a persistent loop with triton-licm holds it (gemm, matvec).
  if (held && (!reach || llvm::countr_zero(held) < llvm::Log2_32(reach))) {
    auto diag = mod.emitError() << "im.optipim_hold_input holds the input only across "
                                   "program ids slower than every one it reaches, and arg "
                                << inArg << " is held across program id "
                                << llvm::countr_zero(held);
    if (reach)
      diag << " but reaches program id " << llvm::Log2_32(reach);
    else
      diag << " and reaches none";
    return diag;
  }
  SmallVector<int64_t> rec{inArg, (int64_t)held};
  for (int a = 0; a < 3; ++a)
    rec.push_back((held >> a) & 1u ? 1 : -1);
  return rec;
}

/// OptiPIM's conventions, stamped on the module beside the layout records, which they leave
/// as they are: the lanes under im.optipim_conv, the roles under im.optipim_roles and the
/// input's hold under im.optipim_hold_input.
static LogicalResult runOptiPimConventions(ModuleOp mod) {
  const bool roles = mod->hasAttr("im.optipim_roles");
  llvm::MapVector<int64_t, int64_t> residency;
  if (roles) {
    auto res = optiPimResidency(mod);
    if (failed(res))
      return failure();
    residency = std::move(*res);
  }
  SmallVector<int64_t> hold;
  if (mod->hasAttr("im.optipim_hold_input")) {
    if (!roles)
      return mod.emitError() << "im.optipim_hold_input holds the input im.optipim_roles names";
    auto rec = optiPimHold(mod, residency);
    if (failed(rec))
      return failure();
    hold = std::move(*rec);
  }
  if (mod->hasAttr("im.optipim_conv") && failed(stampOptiPimLanes(mod)))
    return failure();
  Builder b(mod.getContext());
  if (roles) {
    SmallVector<int64_t> flat;
    for (auto [arg, r] : residency)
      flat.append({arg, r});
    mod->setAttr("im.opt_residency", b.getDenseI64ArrayAttr(flat));
  }
  if (!hold.empty())
    mod->setAttr("im.opt_hold", b.getDenseI64ArrayAttr(hold));
  return success();
}

// -----------------------------------------------------------------------
// Pass implementation
// -----------------------------------------------------------------------

struct IMOperandResidencyLayoutPass
    : public triton::im::impl::IMOperandResidencyLayoutBase<
          IMOperandResidencyLayoutPass> {
  using IMOperandResidencyLayoutBase::IMOperandResidencyLayoutBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    StampContext cx;
    auto schedule = readSchedule(mod);
    if (failed(schedule))
      return signalPassFailure();
    cx.schedule = std::move(*schedule);
    // A reduction loop is an scf.for that carries a value, the accumulator.
    mod.walk([&](scf::ForOp forOp) {
      if (forOp.getNumRegionIterArgs() > 0)
        if (auto fn = forOp->getParentOfType<triton::FuncOp>())
          cx.reductionFuncs.insert(fn.getOperation());
    });

    int64_t numForced = 0;
    llvm::DenseSet<int64_t> matched;
    bool refused = false;
    mod.walk([&](Operation *op) {
      FailureOr<DictionaryAttr> stamp = stampResidency(op, cx, b);
      if (failed(stamp)) {
        refused = true;
        return;
      }
      if (!*stamp)
        return;
      if (stamp->contains("user_forced")) {
        matched.insert(stamp->getAs<IntegerAttr>("operand_arg").getInt());
        ++numForced;
      }
      op->setDiscardableAttr("im.residency", *stamp);
    });
    if (refused)
      return signalPassFailure();
    // A schedule entry nothing matched would read downstream as a tuned run that measured
    // the default.
    for (const auto &kv : cx.schedule) {
      if (!matched.contains(kv.first)) {
        mod.emitError() << "im.schedule: arg" << kv.first
                        << " matched no classified memory access";
        return signalPassFailure();
      }
    }

    // The widest accumulator one lane holds, for the runtime's register-file bound. One
    // number per module, since the machine has one GRF_B per bank.
    int64_t accCells = 0;
    mod.walk([&](triton::StoreOp st) { accCells = std::max(accCells, accCellsPerLane(st)); });
    if (accCells > 0)
      mod->setAttr("im.acc_cells_per_lane", b.getI64IntegerAttr(accCells));

    if (failed(runOptiPimConventions(mod)))
      return signalPassFailure();

    // Whether a reported number came from the classifier or from a forced schedule.
    mod->setAttr("im.residency-user-forced", b.getI64IntegerAttr(numForced));
  }
};

} // anonymous namespace

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

namespace mlir {
namespace triton {
namespace im {

std::unique_ptr<OperationPass<ModuleOp>> createIMOperandResidencyLayoutPass() {
  return std::make_unique<IMOperandResidencyLayoutPass>();
}

} // namespace im
} // namespace triton
} // namespace mlir
