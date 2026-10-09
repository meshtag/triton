// Matching helpers for a contraction, reduce(load * broadcast(operand)), shared by the
// passes that stamp it (im-dcc-tile-mac, im-acc-init-fold, im-optipim-mac) and by the
// OptiPIM cell count in the IM lowering.
#ifndef TRITONIM_CONVERSION_IMCONTRACTION_H
#define TRITONIM_CONVERSION_IMCONTRACTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

inline Value baseArg(Value p) {
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(p))
      return isa<triton::FuncOp>(ba.getOwner()->getParentOp()) ? p : Value();
    Operation *d = p.getDefiningOp();
    if (auto a = dyn_cast<triton::AddPtrOp>(d)) {
      p = a.getPtr();
      continue;
    }
    if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
            triton::ReshapeOp, gpu::ConvertLayoutOp>(d)) {
      p = d->getOperand(0);
      continue;
    }
    return {};
  }
}

inline std::optional<int64_t> constSplat(Value v) {
  v = skipConvert(v);
  if (auto s = v.getDefiningOp<triton::SplatOp>())
    v = s.getSrc();
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  return std::nullopt;
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
      if (auto c = constSplat(d->getOperand(side))) {
        auto o = elemStrides(d->getOperand(1 - side));
        if (!o)
          return std::nullopt;
        for (auto &x : *o)
          x *= *c;
        return o;
      }
  return std::nullopt;
}

/// An integer constant, through broadcast and splat.
inline std::optional<int64_t> maskConstOf(Value v) {
  while (auto bc = v.getDefiningOp<triton::BroadcastOp>())
    v = bc.getSrc();
  if (auto sp = v.getDefiningOp<triton::SplatOp>())
    return getConstantIntValue(sp.getSrc());
  if (auto k = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<SplatElementsAttr>(k.getValue()))
      if (dense.getElementType().isInteger())
        return dense.getSplatValue<llvm::APInt>().getSExtValue();
  return getConstantIntValue(v);
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
        if (maskConstOf(b))
          v = a;
        else if (maskConstOf(a))
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
    t.constant = maskConstOf(v);
    if (!t.constant && (t.cmp = dyn_cast<arith::CmpIOp>(d))) {
      auto cmpTy = dyn_cast<RankedTensorType>(t.cmp.getLhs().getType());
      int64_t inner = cmpTy ? rangeDim(t.cmp.getLhs(), cmpTy.getRank(), t.pure) : -1;
      if (inner >= 0) {
        SmallVector<int64_t> fwd(outer.rbegin(), outer.rend());
        t.dim = cmpTy.getRank() + (int64_t)outer.size() == rank ? replay(fwd, inner) : -1;
      }
      t.bound = maskConstOf(t.cmp.getRhs());
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
