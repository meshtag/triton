/// IMOperandResidencyLayout.cpp. Decide per-operand physical layout and residency
/// for the in-memory (IM) backend.
///
/// Target scope: this pass is substrate-agnostic. Its classification is a pure IR
/// analysis of the kernel's SSA and loop structure with no architecture branch, so
/// it runs for every IM target, both near-memory PIM (HBM-PIM) and
/// processing-using-memory PUM (SIMDRAM). Both runtimes read the layout table
/// emitPimLayoutTable builds from its stamps.
///
/// Phase 1 (classification): for each tensor-of-pointer load or store, and each
/// scalar load that feeds a broadcast (the matvec `x[k]` operand), classify the
/// operand's reuse axes and stamp an `im.residency` dictionary attribute.
///
///   reuse_class is decided from two reachability facts about the address:
///     pidMask: does the address depend on tt.get_program_id (output tile)?
///     ivDep:   does the address depend on the reduction loop induction var?
///   In a function containing a reduction (accumulator-carrying scf.for):
///     ivDep & pid   -> ReductionStridedMatrix  (reduction axis -> columns)
///     ivDep & !pid  -> BroadcastReplicate      (invariant in output tile)
///     !ivDep & pid  -> ParallelSpread          (output, spread over banks)
///     otherwise     -> StreamedElementwise
///   With no reduction loop (pure elementwise, e.g. axpy) every op is
///   StreamedElementwise.
///
/// Phase 2 (footprint): the element offsets one program instance touches, as
/// (extent, stride) axes over the tl.arange axes and the reduction induction
/// variable. It stays logical. The register-slot count those offsets need
/// depends on the bank/row/column interleaving, which lives in the runtime, and
/// a stride the pass cannot fold (conv's CI*R*S weight pitch) travels as a
/// product of kernel argument indices for the runtime to resolve. Resolving at
/// all requires a constant trip count, so an untiled reduction gets no
/// footprint and the host allocator still sizes it.

#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdlib>
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

namespace ttg_ = mlir::triton::gpu;

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

/// How many banks receive a copy of one value: the product of the lane extents over the
/// lane axes the value does NOT occupy. 1 means bank-partitioned. This is the MAGNITUDE
/// that the replicated/partitioned verdict below was missing: with a single-axis split the
/// two agree (miss the one lane axis and every bank gets a copy), but a multi-axis split
/// like 2 on M and 16 on N has a value occupying N replicated over only the 2 M lanes, and
/// calling that "replicated" charged it to all 32. Measured on OptiPIM's own mapping for
/// matmul 128x128x64: B cost 16,384 operand writes where 1,024 is the delivery it needs.
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

/// >1 replicated over that many banks, 0 bank-partitioned, -1 undetermined.
/// A load's own type is never read (see laneAxisMask). Track the dims the loaded value
/// OCCUPIES through the dataflow to the store and ask whether the store's lane dim is
/// one of them: expand_dims inserts a dim it does not occupy, reduce deletes one.
/// TRAP this replaces: reading a rank-2 load directly because it matched the store's
/// rank flipped kpacked matmul's A to partitioned, 65,537 -> 246 cycles on HBM.
/// `cellFanout` (out) is how many output cells IN ONE LANE consume a single loaded
/// value: the product of the output tile's extents over the dims that are neither the
/// lane axis nor occupied by the value. SIMDRAM computes across the columns of a row,
/// so an operand feeding F cells must be written into F columns; charging it once per
/// bank was a 64x discount on matmul (review 2026-09-12). The largest fanout over the
/// stores a value reaches wins, for the same reason replicated wins below.
/// `repLanes` (out) is the store dims with lanes that the value does not occupy, ORed over
/// the stores it reaches. `occMask` narrows what the loaded value occupies at the start.
static int bankReplicatedTri(Operation *memOp, int64_t *cellFanout,
                             uint32_t *repLanes = nullptr, uint32_t occMask = ~0u) {
  if (cellFanout)
    *cellFanout = 1;
  if (repLanes)
    *repLanes = 0u;
  if (auto st = dyn_cast<triton::StoreOp>(memOp))
    return laneAxisMask(st.getValue().getType()) != 0u ? 0 : 1;
  if (memOp->getNumResults() != 1)
    return -1;
  Value v0 = memOp->getResult(0);
  auto tt0 = dyn_cast<RankedTensorType>(v0.getType());
  // Only dims of extent > 1 are occupied: a [BM,1] load is the same value in every
  // lane along the size-1 dim, so counting it stamped a bias vector partitioned.
  uint32_t occ0 = 0u;
  if (tt0)
    for (int d = 0; d < tt0.getRank(); ++d)
      if (tt0.getShape()[d] > 1)
        occ0 |= 1u << d;
  occ0 &= occMask;
  SmallVector<std::pair<Value, uint32_t>, 16> work{{v0, occ0}};
  llvm::DenseSet<Value> seen;
  int hops = 0;
  // Every store the value reaches gets a verdict; replicated wins because OPERAND is
  // the charge that never under-bills. First-store-wins stamped a tensor with a
  // partitioned copy-out and a replicated outer-product use as partitioned.
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
          return -1;
        if (repLanes)
          if (auto axes = laneAxes(st.getValue().getType()))
            for (auto [d, n] : *axes)
              if (!((occ >> d) & 1u))
                *repLanes |= 1u << d;
        // Partitioned only if the value occupies EVERY lane axis. Miss one and it is
        // replicated along it, so every bank on that axis is delivered the value.
        if ((occ & laneM) == laneM) {
          anyPart = true;
        } else {
          anyRep = true;
          // The largest over the stores a value reaches, matching "replicated wins".
          int64_t f = laneReplicationFactor(st.getValue().getType(), occ);
          if (f > repFactor)
            repFactor = f;
        }
        if (cellFanout) {
          auto stt = cast<RankedTensorType>(st.getValue().getType());
          int64_t f = 1;
          for (int d = 0; d < stt.getRank(); ++d)
            if (!((laneM >> d) & 1u) && !((occ >> d) & 1u))
              f *= stt.getShape()[d];
          if (f > *cellFanout)
            *cellFanout = f;
        }
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
      // Ops that move values between dims in ways not tracked here: say so rather
      // than pass occ through unchanged, which stamped tt.trans wrong.
      if (isa<triton::ReshapeOp, triton::JoinOp, triton::SplitOp, triton::CatOp>(u))
        return -1;
      if (isa<triton::DotOp>(u))
        return -1; // operand roles inside a dot are not tracked here
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
      // The accumulator is loop-carried: its last in-loop user is the yield, and
      // the store sits outside on the for's RESULT. Cross both edges or every load
      // that feeds an accumulator comes back undetermined.
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
        // Elementwise or layout-only: occupancy passes through. Anything else is
        // unknown and stays -1 rather than guessed.
        StringRef dialect = u->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math" ||
            isa<triton::gpu::ConvertLayoutOp, triton::BroadcastOp, triton::SplatOp,
                triton::BitcastOp, triton::FpToFpOp>(u))
          work.push_back({u->getResult(0), occ});
        else
          return -1;
      }
    }
  }
  if (anyRep)
    return (int)repFactor;   // banks that receive a copy, >= 2 here
  if (anyPart)
    return 0;
  return -1;
}

namespace {
// Footprint descriptor limits. The layout table record is fixed width, so
// these must match PIM_MAX_FP_AXES / PIM_MAX_STRIDE_ARGS in pim_runtime.c.
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

// Follow the addptr/splat/broadcast chain of an address back to the kernel
// !tt.ptr argument it bottoms out at. Return its func-argument index, or -1.
// This is the join key the runtime emitter maps to a registered tensor id.
static int64_t resolveOperandArg(Value ptr) {
  Value cur = ptr;
  llvm::DenseSet<Value> guard;
  while (cur && guard.insert(cur).second) {
    if (auto barg = dyn_cast<BlockArgument>(cur)) {
      Operation *parent = barg.getOwner()->getParentOp();
      if (isa<triton::FuncOp>(parent))
        return barg.getArgNumber();
      return -1; // a loop/region block arg, not a kernel pointer argument
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      return -1;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      cur = addptr.getPtr();
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      cur = splat.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      cur = bcast.getSrc();
      continue;
    }
    // A layout conversion is pure relabelling, the pointer is the same tensor.
    // Without this a 2-D operand tile resolves to -1 and drops out of the
    // descriptor entirely: the k-packed matmul puts two ttg.convert_layout in the
    // A chain, so the compiler classified the load correctly and then could not
    // say which argument it belonged to.
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
      cur = cvt.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      cur = expand.getSrc();
      continue;
    }
    return -1; // unknown producer of a pointer value
  }
  return -1;
}

// Is `op` a scalar tt.load whose (scalar) result feeds a tt.splat?  This is the
// matvec `x_val = tl.load(x + k)` operand, invisible to the tensor-pointer
// walk because its pointer is a plain !tt.ptr, not a tensor of pointers.
/// Constant integer behind a Value, or nullopt.
static std::optional<int64_t> constIntOf(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Trip count of `f`, if all three bounds are compile-time constants.
static std::optional<int64_t> constTripCount(scf::ForOp f) {
  if (!f)
    return std::nullopt;
  auto lb = constIntOf(f.getLowerBound());
  auto ub = constIntOf(f.getUpperBound());
  auto st = constIntOf(f.getStep());
  if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *st - 1) / *st;
}

/// Compile-time value of a scalar (or uniformly splatted) integer, or nullopt.
static std::optional<int64_t> uniformIntOf(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
    if (auto de = dyn_cast<DenseElementsAttr>(c.getValue()))
      if (de.isSplat())
        return de.getSplatValue<APInt>().getSExtValue();
  }
  if (auto sp = v.getDefiningOp<triton::SplatOp>())
    return uniformIntOf(sp.getSrc());
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>())
    return uniformIntOf(ext.getIn());
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
    return uniformIntOf(trunc.getIn());
  return std::nullopt;
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
/// varying part. `scale` carries the enclosing multiplications down.
/// Report why an address was not modelled. Giving up here is silent otherwise:
/// the operand just falls back to the host budget, which looks like success.
/// IM_FP_DEBUG=1 to see it.
static bool fpBail(Value v, const char *why) {
  static int on = -1;
  if (on < 0)
    on = std::getenv("IM_FP_DEBUG") ? 1 : 0;
  if (on) {
    llvm::errs() << "[fp] bail " << why << " : ";
    if (Operation *d = v ? v.getDefiningOp() : nullptr)
      llvm::errs() << d->getName();
    else
      llvm::errs() << "<block arg>";
    llvm::errs() << "\n";
  }
  return false;
}
/// `key` names the axis. It is the outermost layout-changing op the walk went
/// through, because layout assignment CSEs one tl.arange into several tile
/// dimensions and the range value alone cannot tell them apart. Null until the
/// walk passes such an op, so a range used twice in one index expression still
/// merges into a single axis.
static bool collectAxes(Value v, Value iv, int64_t ivTrip, int64_t ivStep,
                        const SymStride &scale, FpAcc &acc, int depth,
                        Value key) {
  if (!v || depth > 32)
    return fpBail(v, "depth");
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
    return fpBail(v, "no-def");
  if (isa<triton::GetProgramIdOp>(def))
    return true; // fixed within one program instance
  if (auto mr = dyn_cast<triton::MakeRangeOp>(def)) {
    int64_t n = (int64_t)mr.getEnd() - (int64_t)mr.getStart();
    if (n <= 0)
      return false;
    note(v, n, 1);
    return true;
  }
  // ttg.convert_layout only moves values between layouts, so it is a
  // pass-through here. Missing it hid every 2D accumulator address.
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
    return fpBail(v, "mul-nonconst-stride");
  }
  return fpBail(v, "unhandled-op");
}

/// How many times wider the LIVE accumulator is than the value that reaches the
/// store, i.e. the factor a rank-collapsing reduce hid from the store's footprint.
///
/// WHY A MULTIPLIER AND NOT THE ACCUMULATOR'S OWN SIZE. The occupancy denominator
/// keeps coming from the store footprint; this only scales it. That makes the change
/// the IDENTITY on every kernel without such a reduce, by arithmetic rather than by
/// inspection, which matters because the alternative had to get all 24 kernels right:
/// kernels with no accumulator at all (axpy, matadd, the persistent three) would have
/// fallen back to the 8192 default and lost a factor of 4096 in the divisor, and the
/// CO-fused conv unrolls into BLOCK_CO separate accumulator-carrying loops, so an
/// absolute rule that summed across them would rebuild BLOCK_CO*BLOCK_HW, which is
/// the retracted 20.9-33.9x. Here that factor is 1 and the question never arises.
///
/// TRAPS, all of them load-bearing:
///  - redFor is USELESS for this. It walks parents upward, and every reduction kernel
///    here stores AFTER the loop, so a store's redFor is null. Chase the stored VALUE.
///  - Walk ALL operands, not operand 0. Epilogues are normal: the K-tiled matmul
///    stores c_old + acc and attention stores min(max(acc + bias, 0), MAX_VAL).
///  - Skip pointer-typed iter_args. matmul_loopcarried carries b_ptrs at 16x32, which
///    TIES acc on cell count, so picking by size alone is ambiguous on a real kernel.
///  - MAX across loops reached, never sum. Summing is how BLOCK_CO comes back.
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

/// Cells ONE LANE holds of the accumulator a store drains, 0 when nothing accumulates.
/// This is what the register file keeps live across a reduction, so it is what a GRF
/// bound compares against.
///
/// TWO shapes reach it. A loop-carried accumulator (scf.for iter_arg) is the widest
/// carried tile. A single-shot reduce -- tl.sum over a tile with no carry, which is what
/// the persistent matvec, matmul and conv kernels emit, and what canonicalize leaves when
/// a reduction loop has trip count 1 -- keeps the STORED tile live instead, so that is
/// the figure. Taking only the carried form reported 0 for those kernels and the bound
/// never fired on them.
///
/// An elementwise store (VA, RELU) neither carries nor reduces and reports 0: it streams
/// through the ALU without occupying the accumulator file.
///
/// Every tensor one loop carries is live at once, so a loop's figure is the SUM of its
/// carried tiles, and the widest loop wins. Taking the widest carried value admitted an
/// interchanged GEMV holding four 128-cell accumulators as 128.
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
/// The result is the set of element offsets one program instance touches, which
/// is what the runtime turns into a register-slot count. The pass deliberately
/// stops at the logical description: how many physical slots those offsets need
/// depends on the bank/row/column interleaving, which lives in the runtime.
static bool operandFootprint(Value ptr, scf::ForOp redFor,
                             SmallVectorImpl<FpAxis> &out) {
  Value iv = redFor ? redFor.getInductionVar() : Value();
  int64_t ivTrip = 0, ivStep = 1;
  if (redFor) {
    auto t = constTripCount(redFor);
    auto s = constIntOf(redFor.getStep());
    if (!t || !s)
      return fpBail(ptr, "runtime-trip-count");
    ivTrip = *t;
    ivStep = *s;
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
    return fpBail(cur, "ptr-chain");
  }
  for (const auto &kv : acc) {
    const FpAxis &ax = kv.second;
    if (!ax.stride.ok)
      return fpBail(kv.first, "stride-collision");
    if (ax.extent > 1 && (ax.stride.factor != 0 || !ax.stride.args.empty()))
      out.push_back(ax);
  }
  if ((int64_t)out.size() > kMaxFpAxes)
    return fpBail(ptr, "too-many-axes");
  return true;
}

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

// -----------------------------------------------------------------------
// User-expressible schedule (roadmap step 4)
// -----------------------------------------------------------------------
/// The classification below is a DEFAULT. A schedule forces policy on top of it,
/// from `im.schedule` on the module (keyed `arg<N>` by kernel operand index) or
/// from `im.residency.force` on one access op, the op winning where both apply.
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
    // Checked here, not at the call site, so both entry points get it. The first
    // version checked this only on the module path and the per-op path accepted
    // a field that cannot do anything, which is this codebase's oldest bug shape.
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

/// OptiPIM's lanes under im.optipim_conv, beside the layout record so it does not change:
/// __pim_opt_lane_rep holds [arg, replicated lane dims] pairs and __pim_opt_lane_ext the
/// store's lane id as [dim, lanes] pairs, fastest first. Dims number the store tile.
static LogicalResult emitOptiPimLanes(ModuleOp mod,
                                      const llvm::MapVector<int64_t, int64_t> &rep) {
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
  if (!mod.getContext()->getLoadedDialect<LLVM::LLVMDialect>())
    return mod.emitError() << "im.optipim_conv emits LLVM globals here, load the LLVM "
                              "dialect first";
  uint32_t all = 0u;
  SmallVector<int32_t> extFlat;
  for (auto [d, n] : ext.value_or(SmallVector<std::pair<int32_t, int32_t>>{})) {
    all |= 1u << d;
    extFlat.append({d, n});
  }
  SmallVector<int32_t> repFlat;
  for (auto [arg, m] : rep)
    repFlat.append({(int32_t)arg, (int32_t)(m < 0 ? all : (uint32_t)m)});
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  Location loc = mod.getLoc();
  Type i32 = b.getI32Type();
  auto table = [&](StringRef name, ArrayRef<int32_t> v) {
    if (v.empty())
      return;
    auto dataTy = RankedTensorType::get({(int64_t)v.size()}, i32);
    LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, v.size()),
                           /*isConstant=*/true, LLVM::Linkage::External, name,
                           DenseElementsAttr::get(dataTy, v));
  };
  auto scalar = [&](StringRef name, int32_t v) {
    LLVM::GlobalOp::create(b, loc, i32, /*isConstant=*/true, LLVM::Linkage::External,
                           name, b.getI32IntegerAttr(v));
  };
  table("__pim_opt_lane_rep", repFlat);
  scalar("__pim_opt_lane_rep_count", (int32_t)(repFlat.size() / 2));
  table("__pim_opt_lane_ext", extFlat);
  scalar("__pim_opt_lane_ext_count", (int32_t)(extFlat.size() / 2));
  // __pim_opt_lane_on, the active lanes of each lane axis where their mapping leaves some idle,
  // as [dim, active] pairs in __pim_opt_lane_ext's order. im.bank_active numbers im.bank_split's
  // dims, the store tile's lane dims are those dims with the reduced one, never a lane axis,
  // dropped, so the two pair up in dim order.
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
  auto lanes = ext.value_or(SmallVector<std::pair<int32_t, int32_t>>{});
  SmallVector<int32_t> storeDims;
  for (auto [d, n] : lanes)
    storeDims.push_back(d);
  llvm::sort(storeDims);
  if (storeDims.size() != splitDims.size())
    return mod.emitError() << "im.bank_active: the store tile has " << storeDims.size()
                           << " lane axes and im.bank_split " << splitDims.size();
  SmallVector<int32_t> onFlat;
  for (auto [d, n] : lanes) {
    int64_t p = splitDims[llvm::find(storeDims, d) - storeDims.begin()];
    if (split[p] != n || active[p] < 1 || active[p] > n)
      return mod.emitError() << "im.bank_active gives " << active[p] << " active lanes on "
                             << "split dim " << p << " of " << split[p] << ", stored on dim " << d
                             << " with " << n;
    onFlat.append({d, (int32_t)active[p]});
  }
  table("__pim_opt_lane_on", onFlat);
  scalar("__pim_opt_lane_on_count", (int32_t)(onFlat.size() / 2));
  return success();
}

/// Residency by place in the kernel's one contraction (im.optipim_roles): the product's
/// first factor is delivered to the PE (1), its second is resident in the bank (2), and
/// the stored sum is drained by read-out (3). Their codegen fixes roles by position too.
static FailureOr<llvm::MapVector<int64_t, int64_t>> optiPimResidency(ModuleOp mod) {
  auto loadOf = [](Value v) -> triton::LoadOp {
    while (Operation *d = v.getDefiningOp()) {
      if (auto ld = dyn_cast<triton::LoadOp>(d))
        return ld;
      if (!isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
               triton::gpu::ConvertLayoutOp, arith::ExtFOp, arith::TruncFOp>(d))
        return {};
      v = d->getOperand(0);
    }
    return {};
  };
  SmallVector<std::pair<triton::ReduceOp, arith::MulFOp>> found;
  mod.walk([&](triton::ReduceOp red) {
    if (red.getNumOperands() != 1)
      return;
    Value src = red.getOperands()[0];
    while (auto cvt = src.getDefiningOp<triton::gpu::ConvertLayoutOp>())
      src = cvt.getSrc();
    bool adds = false;
    red.getCombineOp().walk([&](Operation *op) { adds |= isa<arith::AddFOp>(op); });
    if (auto mul = src.getDefiningOp<arith::MulFOp>(); mul && adds)
      found.push_back({red, mul});
  });
  if (found.size() != 1)
    return mod.emitError() << "im.optipim_roles states residency for exactly one "
                              "contraction, a sum over a product, and this kernel has "
                           << found.size();
  auto [red, mul] = found.front();
  triton::LoadOp in = loadOf(mul.getLhs()), wt = loadOf(mul.getRhs());
  int64_t inArg = in ? resolveOperandArg(in.getPtr()) : -1;
  int64_t wtArg = wt ? resolveOperandArg(wt.getPtr()) : -1;
  int64_t outArg = -1;
  SmallVector<Value> work{red.getResult()[0]};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (!seen.insert(u).second)
        continue;
      if (auto st = dyn_cast<triton::StoreOp>(u)) {
        if (st.getValue() == v)
          outArg = resolveOperandArg(st.getPtr());
        continue;
      }
      if (u->getNumResults() == 1 && isa<RankedTensorType>(u->getResult(0).getType()))
        work.push_back(u->getResult(0));
    }
  }
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

/// __pim_opt_residency holds [arg, residency] pairs, beside the layout record so it does
/// not change. bank_replicated stays the placement fact.
static LogicalResult emitOptiPimResidency(ModuleOp mod,
                                          const llvm::MapVector<int64_t, int64_t> &res) {
  if (!mod.getContext()->getLoadedDialect<LLVM::LLVMDialect>())
    return mod.emitError() << "im.optipim_roles emits LLVM globals here, load the LLVM "
                              "dialect first";
  SmallVector<int32_t> flat;
  for (auto [arg, r] : res)
    flat.append({(int32_t)arg, (int32_t)r});
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  Location loc = mod.getLoc();
  Type i32 = b.getI32Type();
  auto dataTy = RankedTensorType::get({(int64_t)flat.size()}, i32);
  LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, flat.size()),
                         /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_residency",
                         DenseElementsAttr::get(dataTy, ArrayRef<int32_t>(flat)));
  LLVM::GlobalOp::create(b, loc, i32, /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_residency_count",
                         b.getI32IntegerAttr((int32_t)(flat.size() / 2)));
  return success();
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
static FailureOr<SmallVector<int32_t>>
optiPimHold(ModuleOp mod, const llvm::MapVector<int64_t, int64_t> &res) {
  int64_t inArg = -1;
  for (auto [arg, r] : res)
    if (r == 1)
      inArg = arg;
  std::array<triton::GetProgramIdOp, 3> pids{};
  bool twice = false;
  mod.walk([&](triton::GetProgramIdOp op) {
    if (op->use_empty())
      return;
    twice |= (bool)pids[op.getAxisAsInt()];
    pids[op.getAxisAsInt()] = op;
  });
  if (twice)
    return mod.emitError() << "im.optipim_hold_input reads each program id once";
  // What the input delivers is its address, mask and other. A load in a loop or branch can
  // run a pid-dependent number of times, which a hold per program instance cannot state.
  SmallVector<Value> sources;
  bool stored = false, nested = false;
  mod.walk([&](Operation *op) {
    Value ptr = getMemAccessPtr(op);
    if (!ptr || inArg < 0 || resolveOperandArg(ptr) != inArg)
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
    if (pids[a])
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
  SmallVector<int32_t> rec{(int32_t)inArg, (int32_t)held};
  for (int a = 0; a < 3; ++a)
    rec.push_back((held >> a) & 1u ? 1 : -1);
  return rec;
}

/// __pim_opt_hold is that record, beside the layout record so it does not change.
static void emitOptiPimHold(ModuleOp mod, ArrayRef<int32_t> rec) {
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  Location loc = mod.getLoc();
  Type i32 = b.getI32Type();
  auto dataTy = RankedTensorType::get({(int64_t)rec.size()}, i32);
  LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, rec.size()),
                         /*isConstant=*/true, LLVM::Linkage::External, "__pim_opt_hold",
                         DenseElementsAttr::get(dataTy, rec));
  LLVM::GlobalOp::create(b, loc, i32, /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_hold_count", b.getI32IntegerAttr(1));
  // __pim_opt_memo, their memo's hash of a step as (bound, weight) pairs, slowest loop first.
  auto memo = mod->getAttrOfType<DenseI64ArrayAttr>("im.optipim_memo");
  if (!memo)
    return;
  SmallVector<int32_t> flat(memo.asArrayRef().begin(), memo.asArrayRef().end());
  auto memoTy = RankedTensorType::get({(int64_t)flat.size()}, i32);
  LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, flat.size()),
                         /*isConstant=*/true, LLVM::Linkage::External, "__pim_opt_memo",
                         DenseElementsAttr::get(memoTy, ArrayRef<int32_t>(flat)));
  LLVM::GlobalOp::create(b, loc, i32, /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_memo_count", b.getI32IntegerAttr((int32_t)flat.size() / 2));
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

    // Step 4 user schedule, validated up front so a malformed one fails before
    // any classification rather than half way through it.
    llvm::MapVector<int64_t, DictionaryAttr> argSchedule;
    llvm::DenseSet<int64_t> argScheduleUsed;
    if (auto schedule = mod->getAttrOfType<DictionaryAttr>("im.schedule")) {
      for (NamedAttribute e : schedule) {
        StringRef k = e.getName().strref();
        int64_t argIdx = 0;
        if (!k.starts_with("arg") || k.drop_front(3).getAsInteger(10, argIdx)) {
          mod.emitError() << "im.schedule: key '" << k
                          << "' must be arg<N> with N a kernel operand index";
          return signalPassFailure();
        }
        auto d = dyn_cast<DictionaryAttr>(e.getValue());
        if (!d) {
          mod.emitError() << "im.schedule: '" << k
                          << "' must map to a dictionary of schedule fields";
          return signalPassFailure();
        }
        if (failed(checkSchedule(d, mod.getLoc(), "im.schedule." + k)))
          return signalPassFailure();
        argSchedule[argIdx] = d;
      }
    }

    // Pass A, reduction-loop detection.  A reduction loop is an scf.for that
    // carries iter_args (the accumulator).  Record the count and which
    // functions contain one.
    int64_t numReductionLoops = 0;
    llvm::DenseSet<Operation *> funcsWithReduction;
    mod.walk([&](scf::ForOp forOp) {
      if (forOp.getNumRegionIterArgs() > 0) {
        ++numReductionLoops;
        if (auto fn = forOp->getParentOfType<triton::FuncOp>())
          funcsWithReduction.insert(fn.getOperation());
      }
    });

    llvm::MapVector<int64_t, int64_t> optResidency;
    const bool optipimRoles = mod->hasAttr("im.optipim_roles");
    if (optipimRoles) {
      auto res = optiPimResidency(mod);
      if (failed(res))
        return signalPassFailure();
      optResidency = std::move(*res);
    }
    SmallVector<int32_t> optHold;
    if (mod->hasAttr("im.optipim_hold_input")) {
      if (!optipimRoles) {
        mod.emitError() << "im.optipim_hold_input holds the input im.optipim_roles names";
        return signalPassFailure();
      }
      auto hold = optiPimHold(mod, optResidency);
      if (failed(hold))
        return signalPassFailure();
      optHold = std::move(*hold);
    }

    // Pass B, classify each memory-access op and stamp im.residency.
    int64_t numAnalyzed = 0, numClassified = 0, numForced = 0;
    bool scheduleError = false;
    const bool optipimConv = mod->hasAttr("im.optipim_conv");
    // arg -> replicated lane dims, -1 for a scalar splat (every lane dim).
    llvm::MapVector<int64_t, int64_t> optLaneRep;
    mod.walk([&](Operation *op) {
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      bool isTensorPtr =
          tensorTy && isa<triton::PointerType>(tensorTy.getElementType());
      bool scalarSplat = !isTensorPtr && isScalarSplatLoad(op);
      if (!isTensorPtr && !scalarSplat)
        return;
      ++numAnalyzed;

      // Innermost enclosing reduction loop (for the induction-var test).
      scf::ForOp redFor;
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
        if (auto f = dyn_cast<scf::ForOp>(p)) {
          if (f.getNumRegionIterArgs() > 0) {
            redFor = f;
            break;
          }
        }
      }

      unsigned pidMask = 0;
      {
        llvm::DenseSet<Value> vis;
        pidMask = reachesProgramId(ptr, vis);
      }
      bool ivDep = false;
      if (redFor) {
        llvm::DenseSet<Value> vis;
        ivDep = reachesValue(ptr, redFor.getInductionVar(), vis);
      }
      int64_t operandArg = resolveOperandArg(ptr);

      auto fn = op->getParentOfType<triton::FuncOp>();
      bool funcReduction =
          fn && funcsWithReduction.contains(fn.getOperation());

      // A scalar load splatted across the lanes is a bank-broadcast: the value
      // is identical across all banks for a given access, exactly what per-BG
      // replication models, independent of pid/reduction dependence.
      StringRef cls;
      if (scalarSplat)
        cls = "BroadcastReplicate";
      else if (!funcReduction)
        cls = "StreamedElementwise";
      else if (ivDep && pidMask != 0)
        cls = "ReductionStridedMatrix";
      else if (ivDep && pidMask == 0)
        cls = "BroadcastReplicate";
      else if (!ivDep && pidMask != 0)
        cls = "ParallelSpread";
      else
        cls = "StreamedElementwise";

      // DERIVED ROLE. Replication is a property of the layout, so read it off the
      // encoding rather than letting the host assert it.
      // scalarSplat is itself a positive determination of replication.
      int64_t cellFanout = 1;
      const int bankRep = scalarSplat ? 1 : bankReplicatedTri(op, &cellFanout);

      SmallVector<NamedAttribute> fields;
      fields.push_back(b.getNamedAttr("reuse_class", b.getStringAttr(cls)));
      // OMIT on -1. The emitter defaults the record word to -1, which the runtime
      // reads as "no compiler decision, keep the host's role".
      // The VALUE is how many banks receive a copy: 0 partitioned, >1 replicated over
      // that many. It used to be a bool, which charged every replicated tensor to all
      // 32 banks even when the split replicated it over 2.
      if (bankRep >= 0)
        fields.push_back(b.getNamedAttr("bank_replicated",
                                        b.getI32IntegerAttr(bankRep)));
      // Cells per lane that one CONSUMED value feeds through a TILE BROADCAST. Sharing
      // that reaches memory as repeated loads is counted by the runtime from the loads,
      // so this covers only what never reaches memory. 1 is the identity and the safe
      // default; only a partitioned (STREAMED) tensor is charged by it.
      if (cellFanout > 1)
        fields.push_back(
            b.getNamedAttr("cell_fanout", b.getI64IntegerAttr(cellFanout)));
      fields.push_back(
          b.getNamedAttr("operand_arg", b.getI64IntegerAttr(operandArg)));
      // The residency stamp on each access, like every fact here. The runtime reads the
      // global instead.
      if (auto it = optResidency.find(operandArg); it != optResidency.end())
        fields.push_back(
            b.getNamedAttr("opt_residency", b.getI32IntegerAttr((int32_t)it->second)));
      if (!optHold.empty() && operandArg == optHold[0])
        fields.push_back(b.getNamedAttr(
            "opt_hold", b.getDenseI32ArrayAttr(ArrayRef<int32_t>(optHold).drop_front())));
      // Is the contraction axis TILED INTO the access shape? A rank-1 load inside
      // the reduction loop takes ONE value out of each DQ word, so the operand bus
      // memo cannot fire (matmul measured 0 skips of 1,048,576); tiling the
      // reduction into the load makes word-mates consecutive and the memo collapses
      // them 7/8, worth 8.0x on matmul.
      //
      // SUFFICIENT, NOT NECESSARY, and the name says tiled for that reason. Packing
      // can also be earned by address contiguity without any tiling: conv's scalar
      // weight load walks s with unit stride and its memo fires at exactly 7/8
      // (64512/73728 measured 2026-09-09) while this flag reads false on all three
      // of its tensors. So false here does NOT mean the lever is off. Never use this
      // as the gate for the lever or as evidence that packing is absent; the runtime
      // skip counter is the ground truth.
      bool reductionTiled = false;
      if (redFor && ivDep && op->getNumResults() > 0)
        if (auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
          reductionTiled = rt.getRank() >= 2;
      fields.push_back(
          b.getNamedAttr("reduction_tiled", b.getBoolAttr(reductionTiled)));
      // Address footprint of one program instance, flattened five words per
      // axis as [extent, stride factor, arg, arg, arg] with -1 padding. A
      // stride the pass cannot fold to a number is left as a product of kernel
      // argument indices for the runtime to resolve, which is how conv's
      // CI*R*S weight pitch survives. Absent when the address is not affine in
      // the ranges and the reduction IV.
      SmallVector<FpAxis> fp;
      if (operandFootprint(ptr, redFor, fp) && !fp.empty()) {
        SmallVector<int64_t> flat;
        for (const FpAxis &ax : fp) {
          flat.push_back(ax.extent);
          flat.push_back(ax.stride.factor);
          for (int64_t i = 0; i < kMaxStrideArgs; ++i)
            flat.push_back(i < (int64_t)ax.stride.args.size()
                               ? ax.stride.args[i]
                               : -1);
        }
        fields.push_back(
            b.getNamedAttr("footprint", b.getDenseI64ArrayAttr(flat)));
      }

      // The reduce-collapse factor, stamped on stores only. 1 means the stored value
      // IS the accumulator, which is every kernel here but the split-K matvec.
      if (auto st = dyn_cast<triton::StoreOp>(op)) {
        int64_t ls = liveSplitFactor(st);
        if (ls > 1)
          fields.push_back(b.getNamedAttr("live_split", b.getI64IntegerAttr(ls)));
      }

      // Step 4: the classification above was the default, this is the decision.
      // Per-arg first so the per-op force wins where both name a field.
      bool forced = false;
      if (operandArg >= 0) {
        auto it = argSchedule.find(operandArg);
        if (it != argSchedule.end()) {
          applySchedule(b, it->second, fields);
          argScheduleUsed.insert(operandArg);
          forced = true;
        }
      }
      if (auto perOp = dyn_cast_or_null<DictionaryAttr>(
              op->getDiscardableAttr("im.residency.force"))) {
        if (failed(checkSchedule(perOp, op->getLoc(), "im.residency.force"))) {
          scheduleError = true;
          return;
        }
        applySchedule(b, perOp, fields);
        forced = true;
      }
      if (forced) {
        setField(b, fields, "user_forced", b.getUnitAttr());
        ++numForced;
      }
      // Lanes store distinct elements, so a stored tensor cannot have identical copies.
      if (forced && isa<triton::StoreOp>(op))
        for (NamedAttribute f : fields)
          if (f.getName() == "bank_replicated")
            if (auto n = dyn_cast<IntegerAttr>(f.getValue()); n && n.getInt() != 0) {
              op->emitError() << "a stored tensor cannot be forced bank_replicated: each "
                              << "lane stores its own elements";
              scheduleError = true;
              return;
            }

      // A forced verdict is not the layout's, so it states no lane mask. The mask reads the
      // address, not the load's shape, which a mask operand can broadcast.
      if (optipimConv && operandArg >= 0 && bankRep >= 0 && !forced) {
        uint32_t repLanes = 0u;
        llvm::DenseMap<Value, uint32_t> memo;
        if (!scalarSplat && isa<triton::LoadOp>(op))
          bankReplicatedTri(op, nullptr, &repLanes, addressDims(ptr, memo));
        int64_t &m = optLaneRep.insert({operandArg, 0}).first->second;
        m = (scalarSplat || m < 0) ? -1 : (m | repLanes);
      }

      op->setDiscardableAttr("im.residency", b.getDictionaryAttr(fields));
      ++numClassified;
    });

    if (scheduleError)
      return signalPassFailure();
    // An arg nobody matched means the schedule silently did nothing, which reads
    // downstream as a tuned run that measured the default.
    for (const auto &kv : argSchedule) {
      if (!argScheduleUsed.contains(kv.first)) {
        mod.emitError() << "im.schedule: arg" << kv.first
                        << " matched no classified memory access";
        return signalPassFailure();
      }
    }

    // The widest loop-carried accumulator per lane, for the runtime's register-file
    // bound. One number at module level because the machine has one GRF_B per bank.
    int64_t accCells = 0;
    mod.walk([&](triton::StoreOp st) {
      int64_t c = accCellsPerLane(st);
      if (c > accCells)
        accCells = c;
    });
    if (accCells > 0)
      mod->setAttr("im.acc_cells_per_lane", b.getI64IntegerAttr(accCells));

    if (optipimConv && failed(emitOptiPimLanes(mod, optLaneRep)))
      return signalPassFailure();
    if (optipimRoles && failed(emitOptiPimResidency(mod, optResidency)))
      return signalPassFailure();
    if (!optHold.empty())
      emitOptiPimHold(mod, optHold);

    // Provenance. A reported number has to be able to say whether it came from
    // the classifier or from a forced schedule.
    mod->setAttr("im.residency-user-forced", b.getI64IntegerAttr(numForced));
    mod->setAttr("im.residency-analyzed-mem-ops",
                 b.getI64IntegerAttr(numAnalyzed));
    mod->setAttr("im.residency-classified-mem-ops",
                 b.getI64IntegerAttr(numClassified));
    mod->setAttr("im.residency-reduction-loops",
                 b.getI64IntegerAttr(numReductionLoops));
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
