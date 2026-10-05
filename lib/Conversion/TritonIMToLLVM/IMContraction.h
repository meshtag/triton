// Matching helpers for a contraction, reduce(load * broadcast(operand)), shared by the
// passes that stamp it (im-dcc-tile-mac, im-acc-init-fold).
#ifndef TRITONIM_CONVERSION_IMCONTRACTION_H
#define TRITONIM_CONVERSION_IMCONTRACTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    if (!d)
      return {};
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
  if (!d)
    return std::nullopt;
  if (isa<triton::SplatOp, arith::ConstantOp>(d))
    return zero;
  if (auto r = dyn_cast<triton::MakeRangeOp>(d))
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

} // namespace contraction
} // namespace im
} // namespace triton
} // namespace mlir

#endif // TRITONIM_CONVERSION_IMCONTRACTION_H
