/*
 * IMReductionDistribute -- interchange a reduction loop with the sum in its body.
 *
 * `for k: acc += y[k] + b[k]` reads its two tensors alternately. Written as the nest
 * `for k: for t in (y, b): acc += t[k]` and interchanged, it becomes `for k: acc += y[k]`
 * then `for k: acc += b[k]`, and each loop walks one tensor's rows in turn. The
 * accumulator and every tile are unchanged. The floating-point sum is reassociated, so
 * this runs only with `im.distribute_reductions`.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMREDUCTIONDISTRIBUTE
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.distribute_reductions";
constexpr llvm::StringLiteral kMarkedAttr = "im.reduction-distribute";

static bool reaches(Value v, Value target, llvm::DenseSet<Value> &seen) {
  if (v == target)
    return true;
  if (!seen.insert(v).second)
    return false;
  if (Operation *d = v.getDefiningOp())
    for (Value o : d->getOperands())
      if (reaches(o, target, seen))
        return true;
  return false;
}

static bool hasLoad(Value v, Block *body) {
  Operation *d = v.getDefiningOp();
  if (!d || d->getBlock() != body)
    return false;
  if (isa<triton::LoadOp>(d))
    return true;
  return llvm::any_of(d->getOperands(), [&](Value o) { return hasLoad(o, body); });
}

/// The acc + (x + z) this loop yields, as the two summands, or nothing.
static std::optional<std::pair<Value, Value>> summands(scf::ForOp f) {
  if (f.getNumRegionIterArgs() != 1)
    return std::nullopt;
  Block *body = f.getBody();
  for (Operation &op : *body)
    if (op.getNumRegions() || isa<triton::StoreOp>(op) ||
        (!isa<triton::LoadOp, scf::YieldOp>(op) && !isMemoryEffectFree(&op)))
      return std::nullopt;
  Value acc = f.getRegionIterArg(0);
  auto next = f.getBody()->getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!next)
    return std::nullopt;
  Value s = next.getLhs() == acc ? next.getRhs() : next.getRhs() == acc ? next.getLhs()
                                                                        : Value();
  auto sum = s ? s.getDefiningOp<arith::AddFOp>() : arith::AddFOp();
  if (!sum || !sum->hasOneUse() || sum->getBlock() != body)
    return std::nullopt;
  for (Value t : {sum.getLhs(), sum.getRhs()}) {
    llvm::DenseSet<Value> seen;
    if (reaches(t, acc, seen) || !hasLoad(t, body))
      return std::nullopt;
  }
  return std::make_pair(sum.getLhs(), sum.getRhs());
}

/// A clone of `f` before it that adds only the summand at `side` (0 or 1), from `init`.
static scf::ForOp cloneWithSummand(scf::ForOp f, int side, Value init) {
  OpBuilder b(f);
  IRMapping map;
  auto c = cast<scf::ForOp>(b.clone(*f, map));
  c.getInitArgsMutable()[0].set(init);
  auto next = c.getBody()->getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
  Value acc = c.getRegionIterArg(0);
  int sIdx = next.getLhs() == acc ? 1 : 0;
  auto sum = next->getOperand(sIdx).getDefiningOp<arith::AddFOp>();
  next->setOperand(sIdx, sum->getOperand(side));
  sum->erase();
  // The other summand's loads and address arithmetic are dead now.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*c.getBody())))
      if (&op != c.getBody()->getTerminator() && isOpTriviallyDead(&op)) {
        op.erase();
        changed = true;
      }
  }
  return c;
}

struct IMReductionDistributePass
    : public triton::im::impl::IMReductionDistributeBase<IMReductionDistributePass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kGateAttr))
      return;
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp f) {
      if (summands(f))
        loops.push_back(f);
    });
    for (scf::ForOp f : loops) {
      scf::ForOp first = cloneWithSummand(f, 0, f.getInitArgs()[0]);
      scf::ForOp second = cloneWithSummand(f, 1, first.getResult(0));
      first->setAttr(kMarkedAttr, UnitAttr::get(mod.getContext()));
      second->setAttr(kMarkedAttr, UnitAttr::get(mod.getContext()));
      f.getResult(0).replaceAllUsesWith(second.getResult(0));
      f.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMReductionDistributePass() {
  return std::make_unique<IMReductionDistributePass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
