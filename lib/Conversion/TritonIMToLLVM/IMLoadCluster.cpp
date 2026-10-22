/*
 * IMLoadCluster -- group a loop's loads by tensor, one strip of G iterations at a time.
 *
 * A loop that loads two tensors per iteration alternates between them. Once each tensor
 * fills its own DRAM row per bank, every load switches rows. DCC's generator walks n_grf
 * columns of one operand before the next (gen_trace_HBMPIM_VA.py:95-112), so a row stays
 * open for G commands. This unrolls such a loop in strips of G and moves each strip's loads
 * up, grouped by the kernel argument they read. A map that loads one tensor alternates
 * between its load and store rows the same way.
 *
 * A strip's first group stays live until its last is loaded, so G iterations of any one
 * argument must fit `im.cluster_loads` register-file entries per lane. That count reads
 * the lane layout, which is why this runs after rewrite-im-layout. As DCC's generator does,
 * G is as many iterations as fit and the trip count's remainder is a shorter strip of its
 * own. A strip that is a loop's whole trip count is straight-line code, clustered in place.
 * The widest strip of a loop that stores is stamped as `im.map_strip_entries`: wider than
 * GRF_B, its results must stay in place over the operand registers, and the runtime checks
 * it against both files.
 *
 * Only loads and ops free of memory effects move, and a load never moves above a store
 * to the argument it reads. Moving a load past a store
 * to another argument needs im.noalias_args, so the pass does nothing without it. A
 * persistent kernel's tile loop is never touched, because im-tile-boundary marks one
 * dispatch per iteration of it.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/ADT/SmallPtrSet.h"

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
/// Stamped on the unrolled loop as G.
constexpr llvm::StringLiteral kMarkedAttr = "im.load-cluster";
/// Marks the body's own ops while this pass runs. Clones copy it and the index arithmetic
/// the unroll adds has none, so that arithmetic is placed only as a dependency or last.
constexpr llvm::StringLiteral kBodyAttr = "im.load-cluster.body";

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

/// Loads read two or more arguments, or one when the body also stores, none of them
/// stored, and nothing else in the body touches memory. `widest` gets the most entries per
/// lane one iteration loads of any one argument.
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
  // A map's one loaded tensor and its stores take turns on two rows, as two loads do.
  if (loaded.size() < 2 && (loaded.empty() || stored.empty()))
    return false;
  widest = 0;
  for (auto &kv : loaded) {
    if (stored.contains(kv.first))
      return false;
    widest = std::max(widest, kv.second);
  }
  return widest > 0;
}

/// Reorder `ops`, one strip in program order: its loads grouped by argument in order of
/// first use, then the rest in program order, all placed before `before`. Returns false,
/// touching nothing, when a load depends on an op that cannot move.
static bool clusterLoads(ArrayRef<Operation *> ops, Operation *before) {
  llvm::SmallPtrSet<Operation *, 32> in(ops.begin(), ops.end());
  SmallVector<Value> args;
  for (Operation *op : ops)
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      if (!llvm::is_contained(args, baseArg(ld.getPtr())))
        args.push_back(baseArg(ld.getPtr()));

  llvm::SetVector<Operation *> order;
  bool ok = true;
  std::function<void(Operation *, bool)> place = [&](Operation *op, bool mayWrite) {
    if (!ok || !in.contains(op) || order.contains(op))
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
  for (Value a : args)
    for (Operation *op : ops)
      if (auto ld = dyn_cast<triton::LoadOp>(op))
        if (ld->hasAttr(kBodyAttr) && baseArg(ld.getPtr()) == a)
          place(ld, /*mayWrite=*/false);
  for (Operation *op : ops)
    if (op->hasAttr(kBodyAttr))
      place(op, /*mayWrite=*/true);
  for (Operation *op : ops)
    place(op, /*mayWrite=*/true);
  if (!ok)
    return false;
  for (Operation *op : order)
    op->moveBefore(before);
  return true;
}

/// Splits `f` after its first `n` iterations into `f` and a loop over the rest, which takes
/// f's results as its initial values. Returns the second loop.
static scf::ForOp splitAfter(scf::ForOp f, int64_t n) {
  OpBuilder b(f);
  int64_t lb = *getConstantIntValue(f.getLowerBound());
  int64_t st = *getConstantIntValue(f.getStep());
  Type t = f.getLowerBound().getType();
  Value mid = t.isIndex()
                  ? arith::ConstantIndexOp::create(b, f.getLoc(), lb + n * st).getResult()
                  : arith::ConstantIntOp::create(b, f.getLoc(), lb + n * st,
                                                 t.getIntOrFloatBitWidth())
                        .getResult();
  b.setInsertionPointAfter(f);
  auto rest = cast<scf::ForOp>(b.clone(*f.getOperation()));
  f.getResults().replaceAllUsesWith(rest.getResults());
  rest.getInitArgsMutable().assign(f.getResults());
  f.getUpperBoundMutable().assign(mid);
  rest.getLowerBoundMutable().assign(mid);
  return rest;
}

/// Unrolls `f`, whose trip count is a multiple of g, in strips of g and clusters each
/// strip. A loop of one strip is promoted to straight-line code, clustered where it lands.
static bool unrollStrips(scf::ForOp f, int64_t g, int64_t trip, MLIRContext *ctx) {
  if (trip > g) {
    auto res = loopUnrollByFactor(f, g);
    if (failed(res) || !res->mainLoopOp)
      return false;
    Block &body = *res->mainLoopOp->getBody();
    SmallVector<Operation *> ops;
    for (Operation &op : body.without_terminator())
      ops.push_back(&op);
    if (!clusterLoads(ops, body.getTerminator()))
      return false;
    (*res->mainLoopOp)->setAttr(kMarkedAttr, IntegerAttr::get(IntegerType::get(ctx, 64), g));
    return true;
  }
  Operation *prev = f->getPrevNode(), *next = f->getNextNode();
  Block *block = f->getBlock();
  if (failed(loopUnrollFull(f)))
    return false;
  SmallVector<Operation *> ops;
  for (Operation *op = prev ? prev->getNextNode() : &block->front(); op != next;
       op = op->getNextNode())
    ops.push_back(op);
  return clusterLoads(ops, next);
}

struct IMLoadClusterPass
    : public triton::im::impl::IMLoadClusterBase<IMLoadClusterPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto a = mod->getAttrOfType<IntegerAttr>(kGateAttr);
    if (!a || a.getInt() < 2 || !mod->hasAttr("im.noalias_args"))
      return;
    int64_t cap = a.getInt();
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
    MLIRContext *ctx = mod.getContext();
    int64_t strip = 0;
    for (auto [f, widest] : loops) {
      auto trip = tripCount(f);
      if (!trip)
        continue;
      int64_t g = std::min(cap / widest, *trip);
      if (g < 2)
        continue;
      for (Operation &op : f.getBody()->without_terminator())
        op.setAttr(kBodyAttr, UnitAttr::get(ctx));
      bool stores = false;
      f.getBody()->walk([&](triton::StoreOp) { stores = true; });
      int64_t whole = *trip - *trip % g, rest = *trip % g;
      scf::ForOp tail = rest ? splitAfter(f, whole) : scf::ForOp();
      bool done = unrollStrips(f, g, whole, ctx);
      if (tail && rest > 1)
        done = unrollStrips(tail, rest, rest, ctx) && done;
      if (done && stores)
        strip = std::max(strip, g * widest);
    }
    mod.walk([&](Operation *op) { op->removeAttr(kBodyAttr); });
    if (strip)
      mod->setAttr("im.map_strip_entries",
                   IntegerAttr::get(IntegerType::get(ctx, 64), strip));
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
