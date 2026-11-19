// IR matchers the IM passes share, so that every pass reads a kernel's contraction and its
// addresses the same way.
#ifndef TRITONIM_CONVERSION_IMCONTRACTION_H
#define TRITONIM_CONVERSION_IMCONTRACTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>

namespace mlir {
namespace triton {
namespace im {
namespace contraction {

inline Value skipConvert(Value v) {
  while (auto c = v.getDefiningOp<gpu::ConvertLayoutOp>())
    v = c.getSrc();
  return v;
}

inline bool isBroadcastOperand(Value v) {
  v = skipConvert(v);
  Operation *d = v.getDefiningOp();
  return d && isa<triton::BroadcastOp, triton::ExpandDimsOp>(d);
}

/// The kernel argument a pointer derives from through addptr, splat, broadcast, expand_dims,
/// reshape and layout conversions, or null. The residency pass stops at a tt.reshape
/// (`throughReshape` false), so an address read through one gets no layout record.
inline Value baseArg(Value p, bool throughReshape = true) {
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(p))
      return isa<triton::FuncOp>(ba.getOwner()->getParentOp()) ? p : Value();
    Operation *d = p.getDefiningOp();
    if (auto a = dyn_cast_if_present<triton::AddPtrOp>(d))
      p = a.getPtr();
    else if (isa_and_nonnull<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
                             gpu::ConvertLayoutOp>(d) ||
             (throughReshape && isa_and_nonnull<triton::ReshapeOp>(d)))
      p = d->getOperand(0);
    else
      return {};
  }
}

/// baseArg's position in the kernel signature, -1 when there is none.
inline int64_t kernelArgOf(Value p, bool throughReshape = true) {
  Value a = baseArg(p, throughReshape);
  return a ? (int64_t)cast<BlockArgument>(a).getArgNumber() : -1;
}

/// An integer constant, scalar or splat, seen through any chain of `Through` ops.
template <typename... Through>
inline std::optional<int64_t> constIntThrough(Value v) {
  while (Operation *d = v.getDefiningOp()) {
    if (!isa<Through...>(d))
      break;
    v = d->getOperand(0);
  }
  return getConstantIntValue(v);
}

/// Trip count of `f` when its bounds and step are constants and the step is positive, 0 for a
/// loop that never runs.
inline std::optional<int64_t> constTripCount(scf::ForOp f) {
  auto lb = getConstantIntValue(f.getLowerBound());
  auto ub = getConstantIntValue(f.getUpperBound());
  auto st = getConstantIntValue(f.getStep());
  if (!lb || !ub || !st || *st <= 0)
    return std::nullopt;
  return *ub > *lb ? (*ub - *lb + *st - 1) / *st : 0;
}

/// The tt.get_program_id that reads each axis, null for an axis nothing reads. nullopt when two
/// ops with uses read one axis.
inline std::optional<std::array<triton::GetProgramIdOp, 3>> programIds(ModuleOp mod) {
  std::array<triton::GetProgramIdOp, 3> pids{};
  bool twice = false;
  mod.walk([&](triton::GetProgramIdOp op) {
    if (op->use_empty())
      return;
    twice |= (bool)pids[op.getAxisAsInt()];
    pids[op.getAxisAsInt()] = op;
  });
  if (twice)
    return std::nullopt;
  return pids;
}

/// The product `red` sums, when its one operand is an arith.mulf behind layout conversions and
/// its combine adds. Null for any other reduce.
inline arith::MulFOp sumOfProduct(triton::ReduceOp red) {
  if (red.getNumOperands() != 1)
    return {};
  bool adds = false;
  red.getCombineOp().walk([&](arith::AddFOp) { adds = true; });
  if (!adds)
    return {};
  return skipConvert(red.getOperands()[0]).getDefiningOp<arith::MulFOp>();
}

/// The load a product's factor reads through broadcasts, reshapes, layout conversions and
/// float casts, or null.
inline triton::LoadOp loadOf(Value v) {
  while (Operation *d = v.getDefiningOp()) {
    if (auto ld = dyn_cast<triton::LoadOp>(d))
      return ld;
    if (!isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
             gpu::ConvertLayoutOp, arith::ExtFOp, arith::TruncFOp>(d))
      return {};
    v = d->getOperand(0);
  }
  return {};
}

/// A reduce summing a product. Roles are positional, as OptiPIM's codegen assigns them: the
/// first factor is the input and the second the weight. A factor that is not a load is null.
struct Contraction {
  triton::ReduceOp reduce;
  arith::MulFOp product;
  triton::LoadOp input, weight;
};

/// The module's one contraction. `claim` opens the error for a module with none or several,
/// as in "im.optipim_mac lowers".
inline FailureOr<Contraction> findOneContraction(ModuleOp mod, StringRef claim) {
  SmallVector<Contraction> found;
  mod.walk([&](triton::ReduceOp red) {
    if (arith::MulFOp mul = sumOfProduct(red))
      found.push_back({red, mul, loadOf(mul.getLhs()), loadOf(mul.getRhs())});
  });
  if (found.size() != 1)
    return mod.emitError() << claim << " exactly one contraction, a sum over a product, and "
                           << "this kernel has " << found.size();
  return found.front();
}

/// The store `red`'s result reaches through single-result tensor ops, the last one the walk
/// meets when there are several, or null.
inline triton::StoreOp storeOf(triton::ReduceOp red) {
  triton::StoreOp out;
  SmallVector<Value> work{red.getResult()[0]};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *u : v.getUsers()) {
      if (!seen.insert(u).second)
        continue;
      if (auto st = dyn_cast<triton::StoreOp>(u)) {
        if (st.getValue() == v)
          out = st;
        continue;
      }
      if (u->getNumResults() == 1 && isa<RankedTensorType>(u->getResult(0).getType()))
        work.push_back(u->getResult(0));
    }
  }
  return out;
}

/// Element stride per dim of a pointer tensor, when each term is an arange along one dim
/// times a constant, plus terms that are uniform across the tensor.
inline std::optional<SmallVector<int64_t>> elemStrides(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return SmallVector<int64_t>{};  // a scalar is uniform
  SmallVector<int64_t> zero(ty.getRank(), 0);
  if (isa<BlockArgument>(v))
    return std::nullopt;
  Operation *d = v.getDefiningOp();
  if (isa<triton::SplatOp, arith::ConstantOp>(d))
    return zero;
  if (isa<triton::MakeRangeOp>(d))
    return SmallVector<int64_t>{1};
  if (isa<gpu::ConvertLayoutOp, triton::BroadcastOp, arith::ExtSIOp, arith::ExtUIOp,
          arith::TruncIOp, arith::IndexCastOp>(d))
    return elemStrides(d->getOperand(0));
  if (auto e = dyn_cast<triton::ExpandDimsOp>(d)) {
    auto in = elemStrides(e.getSrc());
    if (!in)
      return std::nullopt;
    in->insert(in->begin() + e.getAxis(), 0);
    return in;
  }
  if (isa<triton::AddPtrOp, arith::AddIOp>(d)) {
    auto a = elemStrides(d->getOperand(0)), b = elemStrides(d->getOperand(1));
    if (!a || !b || a->size() != b->size())
      return std::nullopt;
    for (size_t i = 0; i < a->size(); ++i)
      (*a)[i] += (*b)[i];
    return a;
  }
  if (isa<arith::MulIOp>(d))
    for (int side = 0; side < 2; ++side)
      if (auto c = constIntThrough<gpu::ConvertLayoutOp, triton::SplatOp>(
              d->getOperand(side))) {
        auto o = elemStrides(d->getOperand(1 - side));
        if (!o)
          return std::nullopt;
        for (auto &x : *o)
          x *= *c;
        return o;
      }
  return std::nullopt;
}

/// One conjunct of a load's mask. `dim` is set only for `lhs cmp rhs` whose lhs is one
/// make_range carried to the load's rank, `pure` when that carry has no arithmetic.
struct MaskTerm {
  arith::CmpIOp cmp;
  int64_t dim = -1;
  bool pure = false;
  std::optional<int64_t> bound;    // constant rhs
  std::optional<int64_t> constant; // the conjunct itself is a constant
  bool opaque = false;             // no defining op
};

/// The conjuncts of `mask` on a load of rank `rank`, in the order they are visited.
inline SmallVector<MaskTerm> maskTerms(Value mask, int64_t rank) {
  auto constOf = [](Value v) {
    return constIntThrough<triton::BroadcastOp, triton::SplatOp>(v);
  };
  // Replay expand_dims applied in this order (nearest the range first) on a rank-1 range.
  auto replay = [](ArrayRef<int64_t> axes, int64_t pos) {
    for (int64_t a : axes)
      if (a <= pos)
        ++pos;
    return pos;
  };
  // The dimension of a value that is one make_range carried to `r` by expand_dims,
  // broadcast, layout conversions and integer arithmetic with constants. `pure` stays true
  // only without arithmetic. -1 when it is anything else.
  auto rangeDim = [&](Value v, int64_t r, bool &pure) -> int64_t {
    SmallVector<int64_t> inserted; // outermost first
    pure = true;
    while (Operation *d = v.getDefiningOp()) {
      if (auto ex = dyn_cast<triton::ExpandDimsOp>(d)) {
        inserted.push_back(ex.getAxis());
        v = ex.getSrc();
      } else if (isa<triton::BroadcastOp, gpu::ConvertLayoutOp>(d)) {
        v = d->getOperand(0);
      } else if (auto mr = dyn_cast<triton::MakeRangeOp>(d)) {
        pure &= mr.getStart() == 0;
        SmallVector<int64_t> fwd(inserted.rbegin(), inserted.rend());
        return 1 + (int64_t)inserted.size() == r ? replay(fwd, 0) : -1;
      } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                     arith::RemSIOp, arith::DivUIOp, arith::RemUIOp>(d)) {
        Value a = d->getOperand(0), b = d->getOperand(1);
        if (constOf(b))
          v = a;
        else if (constOf(a))
          v = b;
        else
          return -1;
        pure = false;
      } else {
        return -1;
      }
    }
    return -1;
  };
  SmallVector<MaskTerm> out;
  // Each conjunct with the expand_dims applied after it, outermost first.
  SmallVector<std::pair<Value, SmallVector<int64_t>>> terms{{mask, {}}};
  while (!terms.empty()) {
    auto [v, outer] = terms.pop_back_val();
    Operation *d = v.getDefiningOp();
    if (!d) {
      out.push_back({});
      out.back().opaque = true;
      continue;
    }
    if (isa<arith::AndIOp>(d)) {
      for (Value o : d->getOperands())
        terms.push_back({o, outer});
      continue;
    }
    if (auto ex = dyn_cast<triton::ExpandDimsOp>(d)) {
      auto o = outer;
      o.push_back(ex.getAxis());
      terms.push_back({ex.getSrc(), o});
      continue;
    }
    if (isa<triton::BroadcastOp, gpu::ConvertLayoutOp>(d)) {
      terms.push_back({d->getOperand(0), outer});
      continue;
    }
    MaskTerm t;
    t.constant = constOf(v);
    if (!t.constant && (t.cmp = dyn_cast<arith::CmpIOp>(d))) {
      auto cmpTy = dyn_cast<RankedTensorType>(t.cmp.getLhs().getType());
      int64_t inner = cmpTy ? rangeDim(t.cmp.getLhs(), cmpTy.getRank(), t.pure) : -1;
      if (inner >= 0) {
        SmallVector<int64_t> fwd(outer.rbegin(), outer.rend());
        t.dim = cmpTy.getRank() + (int64_t)outer.size() == rank ? replay(fwd, inner) : -1;
      }
      t.bound = constOf(t.cmp.getRhs());
    }
    out.push_back(t);
  }
  return out;
}

} // namespace contraction
} // namespace im
} // namespace triton
} // namespace mlir

#endif // TRITONIM_CONVERSION_IMCONTRACTION_H
