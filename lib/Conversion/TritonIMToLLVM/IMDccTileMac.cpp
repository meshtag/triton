/*
 * IMDccTileMac -- select DCC's tile MAC for a contraction whose per-lane tile is one.
 *
 * DCC's PIM_MAC_OP1 multiplies an n_mac x n_mac tile, n_mac k values into the 16 outputs
 * of one column, in one command (gen_trace_HBMPIM_GEMV.py:136-145). Our lane issues one
 * command per column it reads. A reduce over load * broadcast(operand) is that tile when
 * the load's lane-fastest axis holds one column of outputs and the next is the reduce axis
 * with whole tiles of k, so n_mac consecutive column reads of the lane are one tile. The
 * load is stamped with the reads one MAC covers and the runtime issues one command per
 * that many. This adopts DCC's ISA pricing, not Samsung's lane-wise MAC, so it is opt-in
 * (`im.dcc_tile_mac`) and, like im-relu-opcode, a convention of the DCC path. A load of
 * an argument that is also read any other way is refused rather than guessed.
 *
 * With `im.dcc_mac_addressing` the stamp also states the matrix geometry their generator
 * addresses a MAC by, [head stride, reduce stride, reduce extent] in elements, read off the
 * load's pointer arithmetic, so the runtime can issue each tile at their address.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMDCCTILEMAC
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.dcc_tile_mac";
/// Stamped on a selected load as [k per tile, outputs per tile, reads per MAC].
constexpr llvm::StringLiteral kMarkedAttr = "im.dcc-tile-mac";
constexpr llvm::StringLiteral kAddrGateAttr = "im.dcc_mac_addressing";
/// [head stride, reduce stride, reduce extent] of a selected load, in elements.
constexpr llvm::StringLiteral kAddrAttr = "im.dcc-mac-addr";

static Value skipConvert(Value v) {
  while (auto c = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = c.getSrc();
  return v;
}

static bool isBroadcastOperand(Value v) {
  v = skipConvert(v);
  Operation *d = v.getDefiningOp();
  return d && isa<triton::BroadcastOp, triton::ExpandDimsOp>(d);
}

static Value baseArg(Value p) {
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
            triton::ReshapeOp, ttg::ConvertLayoutOp>(d)) {
      p = d->getOperand(0);
      continue;
    }
    return {};
  }
}

static std::optional<int64_t> constSplat(Value v) {
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
static std::optional<SmallVector<int64_t>> elemStrides(Value v) {
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
  if (isa<ttg::ConvertLayoutOp, triton::BroadcastOp, arith::ExtSIOp, arith::ExtUIOp,
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

/// [head stride, reduce stride, reduce extent] of a DCC tile load, when A is laid out as
/// their generator reads it, mat[head][k][out] with 16 outputs to a column.
static std::optional<SmallVector<int64_t>> macGeometry(triton::LoadOp ld, unsigned axis,
                                                       int64_t vpc) {
  auto ty = cast<RankedTensorType>(ld.getType());
  auto enc = cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
  auto st = elemStrides(ld.getPtr());
  if (!st || st->size() != (size_t)ty.getRank())
    return std::nullopt;
  auto tpw = enc.getThreadsPerWarp();
  int head = -1;
  for (unsigned dim = 0; dim < tpw.size(); ++dim)
    if (tpw[dim] > 1)
      head = head < 0 ? (int)dim : -2;
  if (head < 0)
    return std::nullopt;
  int64_t kExt = ty.getShape()[axis], kStride = (*st)[axis], hStride = (*st)[head];
  // Outputs contiguous: every other non-unit dim steps by 1 within a column or by columns.
  for (unsigned dim = 0; dim < st->size(); ++dim) {
    if ((int)dim == head || dim == axis || ty.getShape()[dim] == 1)
      continue;
    if ((*st)[dim] != 1 && (*st)[dim] != vpc)
      return std::nullopt;
  }
  if (kStride <= 0 || hStride != kExt * kStride)
    return std::nullopt;
  return SmallVector<int64_t>{hStride, kStride, kExt};
}

/// Reads one DCC MAC covers for this load, or 0 when its lane tile is not DCC's.
static int64_t readsPerMac(triton::LoadOp ld, unsigned axis, int64_t dqBits) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  auto enc = ty ? dyn_cast_or_null<ttg::BlockedEncodingAttr>(ty.getEncoding()) : nullptr;
  if (!enc)
    return 0;
  auto spt = enc.getSizePerThread();
  auto order = enc.getOrder();
  auto tpw = enc.getThreadsPerWarp();
  int64_t vpc = std::max<int64_t>(1, dqBits / ty.getElementType().getIntOrFloatBitWidth());
  SmallVector<unsigned> wide;
  for (unsigned d : order)
    if (spt[d] > 1)
      wide.push_back(d);
  // Lane-fastest axis: one column of outputs. Next: the reduce axis, whole tiles of k.
  if (wide.size() < 2 || wide[0] == axis || spt[wide[0]] != vpc || tpw[wide[0]] != 1)
    return 0;
  if (wide[1] != axis || spt[axis] % vpc != 0 || tpw[axis] != 1)
    return 0;
  return vpc; // n_mac k values x one column of outputs
}

struct IMDccTileMacPass : public triton::im::impl::IMDccTileMacBase<IMDccTileMacPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kGateAttr))
      return;
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();

    bool addressing = mod->hasAttr(kAddrGateAttr);
    llvm::DenseMap<Operation *, SmallVector<int64_t>> geometry;
    llvm::DenseMap<Value, SmallVector<std::pair<triton::LoadOp, int64_t>>> picked;
    mod.walk([&](triton::ReduceOp red) {
      if (red.getNumOperands() != 1)
        return;
      bool add = false;
      red.getCombineOp().walk([&](arith::AddFOp) { add = true; });
      auto mul = skipConvert(red.getOperands()[0]).getDefiningOp<arith::MulFOp>();
      if (!add || !mul)
        return;
      for (int side = 0; side < 2; ++side) {
        auto ld = skipConvert(mul->getOperand(side)).getDefiningOp<triton::LoadOp>();
        if (!ld || !isBroadcastOperand(mul->getOperand(1 - side)))
          continue;
        int64_t r = readsPerMac(ld, red.getAxis(), dqBits);
        Value arg = baseArg(ld.getPtr());
        if (r && arg && addressing) {
          auto g = macGeometry(ld, red.getAxis(), r);
          if (!g) {
            ld->emitError() << "im.dcc_mac_addressing: this tile load's matrix is not laid "
                               "out as DCC's generator addresses it, mat[head][k][out]";
            return;
          }
          geometry[ld] = *g;
        }
        if (r && arg)
          picked[arg].push_back({ld, r});
      }
    });
    // Every load of a selected argument has to be one of the tiles, or the per-tensor
    // count the runtime keeps would mix tiles with plain column reads.
    llvm::DenseMap<Value, int64_t> loadsOf;
    mod.walk([&](triton::LoadOp ld) {
      if (Value a = baseArg(ld.getPtr()))
        loadsOf[a]++;
    });
    for (auto &kv : picked) {
      int64_t r = kv.second.front().second;
      bool uniform = llvm::all_of(kv.second, [&](auto &p) { return p.second == r; });
      if (!uniform || loadsOf[kv.first] != (int64_t)kv.second.size())
        continue;
      for (auto &p : kv.second) {
        p.first->setAttr(kMarkedAttr, DenseI64ArrayAttr::get(mod.getContext(), {r, r, r}));
        if (geometry.count(p.first))
          p.first->setAttr(kAddrAttr,
                           DenseI64ArrayAttr::get(mod.getContext(), geometry[p.first]));
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMDccTileMacPass() {
  return std::make_unique<IMDccTileMacPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
