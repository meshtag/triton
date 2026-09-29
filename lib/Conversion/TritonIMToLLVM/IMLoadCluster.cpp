/*
 * IMLoadCluster -- group a loop's loads by tensor, one strip of G iterations at a time.
 *
 * A loop that loads two tensors per iteration alternates between them. Once each tensor
 * fills its own DRAM row per bank, every load switches rows. DCC's generator walks n_grf
 * columns of one operand before the next (gen_trace_HBMPIM_VA.py:118-131), so a row stays
 * open for G commands. This unrolls such a loop and moves each strip's loads up, grouped
 * by the kernel argument they read. With `im.cluster_serpentine` the unrolled body holds
 * two strips and the second takes the arguments in reverse, so it opens on the row the
 * first one closed.
 *
 * A strip's first group stays live until its last is loaded, so G iterations of any one
 * argument must fit `im.cluster_loads` register-file entries per lane. That count reads
 * the lane layout, which is why this runs after rewrite-im-layout. G is a proper divisor
 * of the trip count, so no epilogue loop and no promotion to straight-line code.
 *
 * Only loads and ops free of memory effects move, and a load never moves above a store
 * to the argument it reads or above an earlier strip's stores. Moving a load past a store
 * to another argument needs im.noalias_args, so the pass does nothing without it. A
 * persistent kernel's tile loop is never touched, because im-tile-boundary marks one
 * dispatch per iteration of it.
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMLOADCLUSTER
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.cluster_loads";
constexpr llvm::StringLiteral kSerpentineAttr = "im.cluster_serpentine";
/// Stamped on the unrolled loop as [G, strips].
constexpr llvm::StringLiteral kMarkedAttr = "im.load-cluster";
/// Unrolled iteration of each body op, only while this pass runs.
constexpr llvm::StringLiteral kIterAttr = "im.load-cluster.iter";

/// The kernel argument a pointer is derived from, or null.
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

static std::optional<int64_t> tripCount(scf::ForOp f) {
  auto lb = getConstantIntValue(f.getLowerBound());
  auto ub = getConstantIntValue(f.getUpperBound());
  auto st = getConstantIntValue(f.getStep());
  if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *st - 1) / *st;
}

/// Register-file entries one lane needs to hold a load's result, -1 if unknown.
static int64_t entriesPerLane(triton::LoadOp ld, int64_t dqBits) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  if (!ty || !ty.getEncoding())
    return -1;
  int64_t bits = (int64_t)ttg::getTotalElemsPerThread(ty) *
                 ty.getElementType().getIntOrFloatBitWidth();
  return (bits + dqBits - 1) / dqBits;
}

/// Loads read two or more arguments, none of them stored, and nothing else in the body
/// touches memory. `widest` gets the most entries per lane one iteration loads of any
/// one argument.
static bool qualifies(scf::ForOp f, int64_t dqBits, int64_t &widest) {
  llvm::MapVector<Value, int64_t> loaded;
  llvm::SetVector<Value> stored;
  for (Operation &op : *f.getBody()) {
    if (op.getNumRegions() > 0)
      return false; // nested control flow: leave the loop alone
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      Value b = baseArg(ld.getPtr());
      int64_t e = entriesPerLane(ld, dqBits);
      if (!b || e < 0)
        return false;
      loaded[b] += e;
    } else if (auto st = dyn_cast<triton::StoreOp>(op)) {
      Value b = baseArg(st.getPtr());
      if (!b)
        return false;
      stored.insert(b);
    } else if (!isMemoryEffectFree(&op) && !isa<scf::YieldOp>(op)) {
      return false;
    }
  }
  if (loaded.size() < 2)
    return false;
  widest = 0;
  for (auto &kv : loaded) {
    if (stored.contains(kv.first))
      return false;
    widest = std::max(widest, kv.second);
  }
  return widest > 0;
}

/// Reorder the body strip by strip: the strip's loads grouped by argument in order of
/// first use (reversed on odd strips), then the rest of the strip in program order.
/// Returns false, touching nothing, when a load depends on an op that cannot move.
static bool clusterStrips(Block &body, int64_t iters, int64_t strips) {
  Operation *term = body.getTerminator();
  auto stripOf = [&](Operation *op) -> int64_t {
    auto a = op->getAttrOfType<IntegerAttr>(kIterAttr);
    return a ? a.getInt() * strips / iters : -1;
  };
  SmallVector<Value> args;
  for (Operation &op : body)
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      if (!llvm::is_contained(args, baseArg(ld.getPtr())))
        args.push_back(baseArg(ld.getPtr()));

  llvm::SetVector<Operation *> order;
  bool ok = true;
  std::function<void(Operation *, bool)> place = [&](Operation *op, bool mayWrite) {
    if (!ok || op == term || order.contains(op) || op->getBlock() != &body)
      return;
    for (Value v : op->getOperands())
      if (Operation *d = v.getDefiningOp())
        place(d, mayWrite);
    if (!mayWrite && !isa<triton::LoadOp>(op) && !isMemoryEffectFree(op)) {
      ok = false;
      return;
    }
    order.insert(op);
  };
  for (int64_t s = 0; s < strips; ++s) {
    SmallVector<Value> byArg(args);
    if (s % 2)
      std::reverse(byArg.begin(), byArg.end());
    for (Value a : byArg)
      for (Operation &op : body)
        if (auto ld = dyn_cast<triton::LoadOp>(op))
          if (stripOf(ld) == s && baseArg(ld.getPtr()) == a)
            place(ld, /*mayWrite=*/false);
    for (Operation &op : body)
      if (stripOf(&op) == s)
        place(&op, /*mayWrite=*/true);
  }
  for (Operation &op : body)
    place(&op, /*mayWrite=*/true);
  if (!ok)
    return false;
  for (Operation *op : order)
    op->moveBefore(term);
  return true;
}

struct IMLoadClusterPass
    : public triton::im::impl::IMLoadClusterBase<IMLoadClusterPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto a = mod->getAttrOfType<IntegerAttr>(kGateAttr);
    if (!a || a.getInt() < 2 || !mod->hasAttr("im.noalias_args"))
      return;
    int64_t cap = a.getInt();
    bool serpentine = mod->hasAttr(kSerpentineAttr);
    int64_t dqBits = 256;
    if (auto q = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = q.getInt();
    bool persistent = mod->hasAttr("im.persistent");
    SmallVector<std::pair<scf::ForOp, int64_t>> loops;
    mod.walk([&](scf::ForOp f) {
      bool nested = false;
      f.getBody()->walk([&](scf::ForOp) { nested = true; });
      bool tileLoop = persistent && !f->getParentOfType<scf::ForOp>();
      int64_t widest = 0;
      if (!nested && !tileLoop && qualifies(f, dqBits, widest))
        loops.push_back({f, widest});
    });
    auto i64 = IntegerType::get(mod.getContext(), 64);
    for (auto [f, widest] : loops) {
      auto trip = tripCount(f);
      if (!trip)
        continue;
      int64_t g = 1;
      for (int64_t d = std::min(cap / widest, *trip - 1); d >= 2; --d)
        if (*trip % d == 0) {
          g = d;
          break;
        }
      if (g < 2)
        continue;
      int64_t strips = serpentine && *trip % (2 * g) == 0 && 2 * g < *trip ? 2 : 1;
      int64_t factor = g * strips;
      for (Operation &op : f.getBody()->without_terminator())
        op.setAttr(kIterAttr, IntegerAttr::get(i64, 0));
      auto res = loopUnrollByFactor(f, factor, [&](unsigned i, Operation *op, OpBuilder) {
        op->setAttr(kIterAttr, IntegerAttr::get(i64, i));
      });
      if (succeeded(res) && res->mainLoopOp &&
          clusterStrips(*res->mainLoopOp->getBody(), factor, strips))
        (*res->mainLoopOp)->setAttr(kMarkedAttr,
                                    DenseI64ArrayAttr::get(mod.getContext(), {g, strips}));
    }
    mod.walk([&](Operation *op) { op->removeAttr(kIterAttr); });
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMLoadClusterPass() {
  return std::make_unique<IMLoadClusterPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
