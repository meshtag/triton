/*
 * IMAccInitFold -- an accumulator's first product writes it, so it needs no reset.
 *
 * Triton's optimize-accumulator-init for a bank PE. The HBM-PIM ISA lets MUL write GRF_B
 * (only MAC is held to it), so the first product of each accumulator entry can initialise
 * the entry and the reset before it goes. A tt.reduce has no init: it starts from its first
 * element, so for reduce(load * broadcast(operand)) the load's read at reduce coordinate 0
 * is that product. The load is stamped with [reduce stride, reduce extent] in elements, from
 * which the runtime finds k for each read, and the store with im.acc-init-fold.
 *
 * Runs only when the module carries `im.acc_init_fold`. One contraction per module is folded,
 * and only when every memory access names its argument, the product load is unmasked and is
 * the argument's only load, and the reduce reaches one store, the argument's only store,
 * through elementwise ops. A uniform offset in the product's address is not seen here: the
 * runtime refuses a dispatch whose output column does not open with its k == 0 read.
 */
#include "IMContraction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMACCINITFOLD
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::im::contraction;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.acc_init_fold";
constexpr llvm::StringLiteral kLoadAttr = "im.mac-first";
constexpr llvm::StringLiteral kStoreAttr = "im.acc-init-fold";

/// The combine region is a lone addf.
static bool isAddReduce(triton::ReduceOp red) {
  Block &body = red.getCombineOp().front();
  int adds = 0, others = 0;
  for (Operation &op : body.without_terminator())
    (isa<arith::AddFOp>(op) ? adds : others)++;
  return adds == 1 && others == 0;
}

/// [reduce stride, reduce extent] when k = (e / stride) % extent for every element the load
/// reads: dims stepping below the reduce stride stay below it, dims above step by whole
/// reduce spans, and a column of outputs sits below the stride, which is how the runtime
/// groups reads into output columns.
static std::optional<std::pair<int64_t, int64_t>> reduceGeometry(triton::LoadOp ld,
                                                                 unsigned axis, int64_t dqBits) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  auto st = elemStrides(ld.getPtr());
  if (!ty || !st || st->size() != (size_t)ty.getRank())
    return std::nullopt;
  int64_t vpc = std::max<int64_t>(1, dqBits / ty.getElementType().getIntOrFloatBitWidth());
  int64_t kStride = (*st)[axis], kExt = ty.getShape()[axis], inner = 0;
  if (kStride <= 0 || kStride % vpc)
    return std::nullopt;
  for (unsigned d = 0; d < st->size(); ++d) {
    int64_t s = (*st)[d], n = ty.getShape()[d];
    if (d == axis || n == 1)
      continue;
    if (s <= 0)
      return std::nullopt;
    if (s < kStride)
      inner += s * (n - 1);
    else if (s % (kStride * kExt))
      return std::nullopt;
  }
  if (inner >= kStride)
    return std::nullopt;
  return std::make_pair(kStride, kExt);
}

/// Stores the reduce's result reaches through elementwise ops, or nullopt when it reaches
/// anything else.
static std::optional<SmallVector<triton::StoreOp>> storesOf(Value v) {
  SmallVector<triton::StoreOp> out;
  SmallVector<Value> work{v};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *u = use.getOwner();
      if (!seen.insert(u).second)
        continue;
      if (auto st = dyn_cast<triton::StoreOp>(u)) {
        if (st.getValue() != cur)
          return std::nullopt;
        out.push_back(st);
        continue;
      }
      if (u->getNumResults() != 1 ||
          !(u->hasTrait<OpTrait::Elementwise>() || isa<ttg::ConvertLayoutOp>(u)))
        return std::nullopt;
      work.push_back(u->getResult(0));
    }
  }
  return out;
}

/// Reduces in a value's producers, and whether any of them loads `forbidden`.
static void producers(Value v, Value forbidden, int &reduces, bool &readsOut) {
  SmallVector<Value> work{v};
  llvm::SmallPtrSet<Operation *, 32> seen;
  while (!work.empty()) {
    Operation *d = work.pop_back_val().getDefiningOp();
    if (!d || !seen.insert(d).second)
      continue;
    if (isa<triton::ReduceOp>(d)) {
      reduces++;
      continue;
    }
    if (auto ld = dyn_cast<triton::LoadOp>(d)) {
      readsOut |= baseArg(ld.getPtr()) == forbidden;
      continue;
    }
    for (Value o : d->getOperands())
      work.push_back(o);
  }
}

struct IMAccInitFoldPass
    : public triton::im::impl::IMAccInitFoldBase<IMAccInitFoldPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kGateAttr))
      return;
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();
    // An access whose argument is not traceable, through a loop-carried pointer say, could
    // read the product's tensor or write the accumulator unseen.
    bool opaque = false;
    llvm::DenseMap<Value, int64_t> loadsOf, storesTo;
    mod.walk([&](Operation *op) {
      if (auto ld = dyn_cast<triton::LoadOp>(op)) {
        Value a = baseArg(ld.getPtr());
        opaque |= !a;
        loadsOf[a]++;
      }
      if (auto st = dyn_cast<triton::StoreOp>(op)) {
        Value a = baseArg(st.getPtr());
        opaque |= !a;
        storesTo[a]++;
      }
    });
    if (opaque)
      return;
    struct Fold {
      triton::LoadOp ld;
      std::pair<int64_t, int64_t> geo;
      SmallVector<triton::StoreOp> stores;
    };
    SmallVector<Fold> folds;
    bool refused = false;
    mod.walk([&](triton::ReduceOp red) {
      if (red.getNumOperands() != 1 || red->hasAttr("im.lane-fold") || !isAddReduce(red))
        return;
      auto mul = skipConvert(red.getOperands()[0]).getDefiningOp<arith::MulFOp>();
      if (!mul)
        return;
      for (int side = 0; side < 2; ++side) {
        auto ld = skipConvert(mul->getOperand(side)).getDefiningOp<triton::LoadOp>();
        if (!ld || !isBroadcastOperand(mul->getOperand(1 - side)))
          continue;
        Value arg = baseArg(ld.getPtr());
        auto geo = reduceGeometry(ld, red.getAxis(), dqBits);
        auto stores = storesOf(red.getResult()[0]);
        // A masked product may skip its k == 0 read, and nothing would write the entry.
        if (ld.getMask() || !geo || !stores || stores->size() != 1 || loadsOf[arg] != 1) {
          refused = true;
          return;
        }
        triton::StoreOp st = stores->front();
        Value out = baseArg(st.getPtr());
        int reduces = 0;
        bool readsOut = false;
        producers(st.getValue(), out, reduces, readsOut);
        if (reduces != 1 || readsOut || storesTo[out] != 1) {
          refused = true;
          return;
        }
        folds.push_back({ld, *geo, *stores});
        return;
      }
    });
    if (refused || folds.size() != 1)
      return;
    Fold &f = folds.front();
    f.ld->setAttr(kLoadAttr,
                  DenseI64ArrayAttr::get(mod.getContext(), {f.geo.first, f.geo.second}));
    f.stores.front()->setAttr(kStoreAttr, UnitAttr::get(mod.getContext()));
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMAccInitFoldPass() {
  return std::make_unique<IMAccInitFoldPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
