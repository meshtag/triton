/*
 * IMOperandHoist -- fully unroll a reduction loop whose operand every tile reloads, so
 * triton-licm can keep that operand in the register file across the tile loop.
 *
 * In `for tile: for k in range(T): acc += A[tile, k] * x[k]` the address of x[k] depends
 * on k and not on the tile, and nothing can hoist it while k is a loop variable. Fully
 * unrolled, each x[k] is tile-invariant and triton-licm moves it above the tile loop
 * (licensed by im.noalias_args). The T copies it hoists stay live for the whole tile
 * loop, so they must fit the operand register file of one lane, counted in entries from
 * the lane layout. That is why this runs after rewrite-im-layout. Runs only with
 * `im.hoist_operand_regs = entries`.
 */
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMOPERANDHOIST
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.hoist_operand_regs";
/// Stamped on the tile loop as [trip count, entries per lane] of what it made hoistable.
constexpr llvm::StringLiteral kMarkedAttr = "im.operand-hoist";

static std::optional<int64_t> tripCount(scf::ForOp f) {
  auto lb = getConstantIntValue(f.getLowerBound());
  auto ub = getConstantIntValue(f.getUpperBound());
  auto st = getConstantIntValue(f.getStep());
  if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *st - 1) / *st;
}

/// True when `v` depends on nothing inside `outer` but `iv` and effect-free ops, so it
/// becomes invariant to `outer` once `iv` is a constant.
static bool invariantModuloIV(Value v, scf::ForOp outer, Value iv,
                              DenseSet<Value> &seen) {
  if (!v || v == iv || !seen.insert(v).second || outer.isDefinedOutsideOfLoop(v))
    return true;
  if (isa<BlockArgument>(v))
    return false; // an iter_arg or the tile loop's own variable
  Operation *d = v.getDefiningOp();
  // A gathered address: licm hoists the index load with the gather, so both count.
  if (auto ld = dyn_cast<triton::LoadOp>(d))
    return invariantModuloIV(ld.getPtr(), outer, iv, seen) &&
           invariantModuloIV(ld.getMask(), outer, iv, seen) &&
           invariantModuloIV(ld.getOther(), outer, iv, seen);
  if (d->getNumRegions() != 0 || !isMemoryEffectFree(d))
    return false;
  for (Value o : d->getOperands())
    if (!invariantModuloIV(o, outer, iv, seen))
      return false;
  return true;
}

/// Register-file entries one lane needs to hold a load's result.
static int64_t entriesPerLane(triton::LoadOp ld, int64_t dqBits) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  if (!ty || !ty.getEncoding())
    return -1;
  int64_t bits = (int64_t)ttg::getTotalElemsPerThread(ty) *
                 ty.getElementType().getIntOrFloatBitWidth();
  return (bits + dqBits - 1) / dqBits;
}

struct IMOperandHoistPass
    : public triton::im::impl::IMOperandHoistBase<IMOperandHoistPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto gate = mod->getAttrOfType<IntegerAttr>(kGateAttr);
    if (!gate || gate.getInt() < 1)
      return;
    int64_t cap = gate.getInt();
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();

    SmallVector<scf::ForOp> inner;
    mod.walk([&](scf::ForOp f) {
      bool nested = false;
      f.getBody()->walk([&](scf::ForOp) { nested = true; });
      if (!nested && f->getParentOfType<scf::ForOp>())
        inner.push_back(f);
    });

    // Everything hoisted out of one tile loop is live across all of it, so the budget
    // is per tile loop, and it starts with the loads already above that loop.
    DenseMap<Operation *, int64_t> used;
    auto liveIn = [&](scf::ForOp outer) {
      int64_t e = 0;
      llvm::DenseSet<Operation *> seen;
      // Through the effect-free ops (broadcast, expand_dims, layout changes) a value
      // defined above the loop reaches it by.
      std::function<void(Value)> reach = [&](Value v) {
        Operation *d = v.getDefiningOp();
        if (!d || !outer.isDefinedOutsideOfLoop(v) || !seen.insert(d).second)
          return;
        if (auto ld = dyn_cast<triton::LoadOp>(d)) {
          e += std::max<int64_t>(0, entriesPerLane(ld, dqBits));
          return;
        }
        if (d->getNumRegions() == 0 && isMemoryEffectFree(d))
          for (Value o : d->getOperands())
            reach(o);
      };
      outer.getBody()->walk([&](Operation *op) {
        for (Value v : op->getOperands())
          reach(v);
      });
      return e;
    };
    for (scf::ForOp f : inner) {
      auto trip = tripCount(f);
      if (!trip || *trip < 2)
        continue;
      scf::ForOp outer = f->getParentOfType<scf::ForOp>();
      if (!used.count(outer))
        used[outer] = liveIn(outer);
      int64_t perTrip = 0;
      bool unknown = false;
      for (Operation &op : *f.getBody()) {
        auto ld = dyn_cast<triton::LoadOp>(op);
        if (!ld)
          continue;
        DenseSet<Value> seen;
        bool inv = invariantModuloIV(ld.getPtr(), outer, f.getInductionVar(), seen) &&
                   invariantModuloIV(ld.getMask(), outer, f.getInductionVar(), seen) &&
                   invariantModuloIV(ld.getOther(), outer, f.getInductionVar(), seen);
        if (!inv)
          continue;
        int64_t e = entriesPerLane(ld, dqBits);
        if (e < 0)
          unknown = true;
        perTrip += e;
      }
      if (unknown || perTrip == 0)
        continue;
      int64_t need = perTrip * *trip;
      if (used[outer] + need > cap)
        continue;
      if (failed(loopUnrollFull(f)))
        continue;
      used[outer] += need;
      outer->setAttr(kMarkedAttr, DenseI64ArrayAttr::get(mod.getContext(),
                                                         {*trip, used[outer]}));
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMOperandHoistPass() {
  return std::make_unique<IMOperandHoistPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
