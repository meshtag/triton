/*
 * Scalar replacement on Triton IR: four passes that keep values a kernel moves through
 * memory in registers instead, or move its stores together, which on PIM removes DRAM reads,
 * writes and row switches.
 *
 *   im-register-promotion  a loop that loads and stores one loop-invariant address keeps the
 *                          value in a loop-carried register, loaded once before the loop and
 *                          stored once after it
 *   im-store-forwarding    a load of an address a store in the same block wrote takes the
 *                          stored value
 *   im-dead-stores         a store that a later store in the same block overwrites, with no
 *                          read of it between, is removed
 *   im-store-sinking       a store moves down its block until something may read or write its
 *                          address, or another store, so a block's stores issue together
 *
 * Two accesses are to the same address when their pointers compute the same value, op by op.
 * Two accesses through one kernel argument may overlap at other addresses, and two through
 * different arguments may only without im.noalias_args, Tessera's licence that the caller
 * passes distinct buffers. A loop or branch is seen through to the loads and stores inside
 * it. Masked and volatile accesses are left alone. Each runs before CSE, so it compares
 * pointers by what they compute.
 */
#include "IMContraction.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMREGISTERPROMOTION
#define GEN_PASS_DEF_IMSTOREFORWARDING
#define GEN_PASS_DEF_IMDEADSTORES
#define GEN_PASS_DEF_IMSTORESINKING
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
using mlir::triton::im::contraction::baseArg;

namespace {

constexpr llvm::StringLiteral kLicence = "im.noalias_args";

/// a and b compute the same value: identical, or results of side-effect-free ops with the
/// same name, attributes and types whose operands are the same values in turn.
static bool sameValue(Value a, Value b, int depth = 16) {
  if (a == b)
    return true;
  if (depth == 0)
    return false;
  auto ra = dyn_cast<OpResult>(a), rb = dyn_cast<OpResult>(b);
  if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber())
    return false;
  Operation *da = ra.getOwner(), *db = rb.getOwner();
  if (da->getName() != db->getName() || da->getNumRegions() || db->getNumRegions() ||
      da->getAttrDictionary() != db->getAttrDictionary() ||
      da->getResultTypes() != db->getResultTypes() ||
      da->getNumOperands() != db->getNumOperands() || !isMemoryEffectFree(da))
    return false;
  for (auto [x, y] : llvm::zip(da->getOperands(), db->getOperands()))
    if (!sameValue(x, y, depth - 1))
      return false;
  return true;
}

/// Accesses through p and q may touch the same memory.
static bool mayOverlap(Value p, Value q, bool licence) {
  Value a = baseArg(p), b = baseArg(q);
  return !a || !b || a == b || !licence;
}

static bool plainLoad(triton::LoadOp ld) { return !ld.getMask() && !ld.getIsVolatile(); }
static bool plainStore(triton::StoreOp st) { return !st.getMask(); }

/// What an op reads and writes through pointers, nested ops included, and whether it may
/// touch memory any other way.
struct Effects {
  SmallVector<Value> reads, writes;
  bool unknown = false;
  bool readsFrom(Value p, bool licence) const {
    return unknown || llvm::any_of(reads, [&](Value r) { return mayOverlap(r, p, licence); });
  }
  bool writesTo(Value p, bool licence) const {
    return unknown || llvm::any_of(writes, [&](Value w) { return mayOverlap(w, p, licence); });
  }
};

static Effects effectsOf(Operation &op) {
  Effects e;
  op.walk([&](Operation *in) {
    if (auto ld = dyn_cast<triton::LoadOp>(in))
      e.reads.push_back(ld.getPtr());
    else if (auto st = dyn_cast<triton::StoreOp>(in))
      e.writes.push_back(st.getPtr());
    else if (!in->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
             !in->hasTrait<OpTrait::IsTerminator>() && !isMemoryEffectFree(in))
      e.unknown = true;
  });
  return e;
}

static SmallVector<Block *> allBlocks(ModuleOp mod) {
  SmallVector<Block *> blocks;
  mod.walk([&](Block *b) { blocks.push_back(b); });
  return blocks;
}

// --------------------------------------------------------------------------
// im-store-forwarding
// --------------------------------------------------------------------------

static void forwardInBlock(Block &block, bool licence) {
  SmallVector<triton::StoreOp> live;
  SmallVector<Operation *> dead;
  for (Operation &op : block) {
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      if (!plainLoad(ld))
        continue;
      for (auto st : llvm::reverse(live)) {
        if (sameValue(st.getPtr(), ld.getPtr()) && st.getValue().getType() == ld.getType()) {
          ld.getResult().replaceAllUsesWith(st.getValue());
          dead.push_back(ld);
          break;
        }
      }
    } else if (auto st = dyn_cast<triton::StoreOp>(op)) {
      llvm::erase_if(live, [&](triton::StoreOp s) {
        return mayOverlap(s.getPtr(), st.getPtr(), licence);
      });
      if (plainStore(st))
        live.push_back(st);
    } else if (!isMemoryEffectFree(&op)) {
      Effects e = effectsOf(op);
      llvm::erase_if(live, [&](triton::StoreOp s) { return e.writesTo(s.getPtr(), licence); });
    }
  }
  for (Operation *op : dead)
    op->erase();
}

struct IMStoreForwardingPass
    : public triton::im::impl::IMStoreForwardingBase<IMStoreForwardingPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool licence = mod->hasAttr(kLicence);
    for (Block *b : allBlocks(mod))
      forwardInBlock(*b, licence);
  }
};

// --------------------------------------------------------------------------
// im-dead-stores
// --------------------------------------------------------------------------

static void deadStoresInBlock(Block &block, bool licence) {
  SmallVector<Operation *> dead;
  for (Operation &op : block) {
    auto first = dyn_cast<triton::StoreOp>(op);
    if (!first || !plainStore(first))
      continue;
    Value p = first.getPtr();
    for (Operation *o = op.getNextNode(); o && !o->hasTrait<OpTrait::IsTerminator>();
         o = o->getNextNode()) {
      if (auto st = dyn_cast<triton::StoreOp>(o)) {
        if (plainStore(st) && sameValue(st.getPtr(), p) &&
            st.getValue().getType() == first.getValue().getType()) {
          dead.push_back(&op);
          break;
        }
        continue;
      }
      if (!isMemoryEffectFree(o) && effectsOf(*o).readsFrom(p, licence))
        break;
    }
  }
  for (Operation *op : dead)
    op->erase();
}

struct IMDeadStoresPass : public triton::im::impl::IMDeadStoresBase<IMDeadStoresPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool licence = mod->hasAttr(kLicence);
    for (Block *b : allBlocks(mod))
      deadStoresInBlock(*b, licence);
  }
};

// --------------------------------------------------------------------------
// im-store-sinking
// --------------------------------------------------------------------------

/// Sinks the block's stores, the last first, each to just before the first op after it that
/// may read or write its address, another store, or the terminator, which keeps the stores
/// in their order.
static void sinkInBlock(Block &block, bool licence) {
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : block)
    if (auto st = dyn_cast<triton::StoreOp>(op))
      stores.push_back(st);
  for (triton::StoreOp st : llvm::reverse(stores)) {
    if (!plainStore(st))
      continue;
    Value p = st.getPtr();
    Operation *stop = st->getNextNode();
    while (stop && !stop->hasTrait<OpTrait::IsTerminator>() &&
           !isa<triton::StoreOp>(stop)) {
      if (!isMemoryEffectFree(stop)) {
        Effects e = effectsOf(*stop);
        if (e.readsFrom(p, licence) || e.writesTo(p, licence))
          break;
      }
      stop = stop->getNextNode();
    }
    if (stop && stop != st->getNextNode())
      st->moveBefore(stop);
  }
}

struct IMStoreSinkingPass
    : public triton::im::impl::IMStoreSinkingBase<IMStoreSinkingPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool licence = mod->hasAttr(kLicence);
    for (Block *b : allBlocks(mod))
      sinkInBlock(*b, licence);
  }
};

// --------------------------------------------------------------------------
// im-register-promotion
// --------------------------------------------------------------------------

/// v is the same in every iteration of `loop`: defined outside it, or by a side-effect-free
/// op inside whose operands are.
static bool invariant(Value v, scf::ForOp loop, int depth = 16) {
  if (!loop->isAncestor(v.getParentRegion()->getParentOp()))
    return true;
  Operation *d = v.getDefiningOp();
  if (!d || depth == 0 || d->getNumRegions() || !isMemoryEffectFree(d))
    return false;
  return llvm::all_of(d->getOperands(), [&](Value o) { return invariant(o, loop, depth - 1); });
}

/// v recomputed at the builder's point, outside `loop`, its in-loop ops cloned.
static Value hoisted(OpBuilder &b, Value v, scf::ForOp loop, IRMapping &map) {
  if (map.contains(v))
    return map.lookup(v);
  if (!loop->isAncestor(v.getParentRegion()->getParentOp()))
    return v;
  Operation *d = v.getDefiningOp();
  for (Value o : d->getOperands())
    hoisted(b, o, loop, map);
  Operation *c = b.clone(*d, map);
  return c->getResult(cast<OpResult>(v).getResultNumber());
}

/// Promotes one address of `loop` and returns the new loop, or null if none qualifies.
static scf::ForOp promoteOne(scf::ForOp loop, bool licence) {
  Block *body = loop.getBody();
  SmallVector<Operation *> mem;
  for (Operation &op : body->without_terminator()) {
    if (isa<triton::LoadOp, triton::StoreOp>(op))
      mem.push_back(&op);
    else if (!isMemoryEffectFree(&op))
      return nullptr;
  }
  auto ptrOf = [](Operation *op) {
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      return ld.getPtr();
    return cast<triton::StoreOp>(op).getPtr();
  };
  for (Operation *anchor : mem) {
    auto st = dyn_cast<triton::StoreOp>(anchor);
    if (!st || !plainStore(st) || !invariant(st.getPtr(), loop))
      continue;
    Value p = st.getPtr();
    SmallVector<Operation *> group;
    bool ok = true;
    for (Operation *op : mem) {
      if (sameValue(ptrOf(op), p)) {
        auto ld = dyn_cast<triton::LoadOp>(op);
        if ((ld && (!plainLoad(ld) || ld.getType() != st.getValue().getType())) ||
            (!ld && !plainStore(cast<triton::StoreOp>(op))))
          ok = false;
        group.push_back(op);
      } else if (mayOverlap(ptrOf(op), p, licence)) {
        ok = false;
      }
    }
    // The register starts from memory, so the first access in an iteration is a load.
    if (!ok || !isa<triton::LoadOp>(group.front()))
      continue;

    OpBuilder b(loop);
    IRMapping map;
    Value outside = hoisted(b, p, loop, map);
    IRMapping first;
    first.map(ptrOf(group.front()), outside);
    Value init = b.clone(*group.front(), first)->getResult(0);
    scf::ForOp grown = addIterArgsToLoop(b, loop, {init});
    Value current = grown.getRegionIterArgs().back();
    triton::StoreOp last;
    for (Operation *op : group) {
      if (auto ld = dyn_cast<triton::LoadOp>(op)) {
        ld.getResult().replaceAllUsesWith(current);
      } else {
        last = cast<triton::StoreOp>(op);
        current = last.getValue();
      }
    }
    appendToForOpYield(grown, {current});
    b.setInsertionPointAfter(grown);
    IRMapping after;
    after.map(last.getPtr(), outside);
    after.map(last.getValue(), grown.getResults().back());
    b.clone(*last, after);
    for (Operation *op : llvm::reverse(group))
      op->erase();
    return grown;
  }
  return nullptr;
}

struct IMRegisterPromotionPass
    : public triton::im::impl::IMRegisterPromotionBase<IMRegisterPromotionPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool licence = mod->hasAttr(kLicence);
    // Innermost loops first, so a promoted inner loop leaves its load and store to the loop
    // around it.
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp f) { loops.push_back(f); });
    for (scf::ForOp f : loops)
      while (scf::ForOp grown = promoteOne(f, licence))
        f = grown;
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMRegisterPromotionPass() {
  return std::make_unique<IMRegisterPromotionPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createIMStoreForwardingPass() {
  return std::make_unique<IMStoreForwardingPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createIMDeadStoresPass() {
  return std::make_unique<IMDeadStoresPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> createIMStoreSinkingPass() {
  return std::make_unique<IMStoreSinkingPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
