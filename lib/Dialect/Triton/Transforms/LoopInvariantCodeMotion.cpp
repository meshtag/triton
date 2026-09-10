#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPINVARIANTCODEMOTION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopInvariantCodeMotionPass
    : public impl::TritonLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotionPass> {

  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;
  /// Set when the module asserts its pointer arguments do not alias, which
  /// licenses the relaxed check below. Off by default: without the assertion the
  /// pass keeps its original whole-loop-read-only behaviour exactly.
  bool assumeNoAliasArgs = false;

  /// Is this a pointer, or a tensor of pointers?
  static bool isPtrLike(Type t) {
    if (isa<PointerType>(t))
      return true;
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return isa<PointerType>(rt.getElementType());
    return false;
  }

  /// Which function argument this pointer is built from.
  ///
  /// Follows only pointer-typed operands, so the index arithmetic in
  /// `A + row*K + k` does not drag in the scalar arguments: the walk reaches A
  /// and stops. `unknown` is set when it meets a pointer whose provenance it
  /// cannot name, such as a loop-carried block argument.
  ///
  /// TRAP: `unknown` is a separate flag rather than a sentinel value in the set.
  /// The obvious sentinel, ~0u, is DenseMapInfo<unsigned>'s EMPTY KEY, so
  /// `DenseSet<unsigned>::contains(~0u)` reports it present in an empty set and
  /// every query answers "unknown". That cost a build cycle to find.
  static void collectPtrArgs(Value v, DenseSet<unsigned> &out, bool &unknown,
                             DenseSet<Value> &seen, unsigned depth = 0) {
    if (depth > 64 || !isPtrLike(v.getType()) || !seen.insert(v).second)
      return;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Operation *owner = ba.getOwner()->getParentOp();
      if (owner && isa<FunctionOpInterface>(owner))
        out.insert(ba.getArgNumber());
      else
        unknown = true;
      return;
    }
    if (Operation *def = v.getDefiningOp())
      for (Value o : def->getOperands())
        collectPtrArgs(o, out, unknown, seen, depth + 1);
    else
      unknown = true;
  }

  static bool resolvesToOneArg(Value v, unsigned &arg) {
    DenseSet<unsigned> args;
    DenseSet<Value> seen;
    bool unknown = false;
    collectPtrArgs(v, args, unknown, seen);
    if (unknown || args.size() != 1)
      return false;
    arg = *args.begin();
    return true;
  }

  /// True when no write inside the loop can touch what `load` reads.
  ///
  /// WHY THIS EXISTS. The original guard asks whether the WHOLE loop is
  /// read-only, so one store anywhere in the body pins every load inside it,
  /// however unrelated. That is what stops a persistent matmul from hoisting its
  /// invariant operand out of the tile loop: the loop stores the output tile, so
  /// nothing moves, and the operand is re-read once per tile. Measured on
  /// 128x3072x768: 4,718,592 operand writes against 1,179,648 for the gridded
  /// form, four times worse, exactly the tile count per instance.
  ///
  /// The narrower question is whether any store in the loop aliases THIS load.
  /// Here that is decidable: both sides resolve back to a kernel pointer
  /// argument, and distinct arguments are distinct buffers. Gated on the module
  /// asserting that, because in general two pointer arguments may alias.
  ///
  /// Conservative in every direction it can be: an unresolvable pointer on
  /// either side, or any writing op that is not a triton store, returns false.
  bool noWriteInLoopAliases(LoopLikeOpInterface loopLike, Operation *load) {
    unsigned loadArg;
    if (!resolvesToOneArg(cast<LoadOp>(load).getPtr(), loadArg))
      return false;
    bool safe = true;
    loopLike->walk([&](Operation *op) {
      if (op == load || !safe)
        return;
      if (auto store = dyn_cast<StoreOp>(op)) {
        unsigned storeArg;
        if (!resolvesToOneArg(store.getPtr(), storeArg) || storeArg == loadArg)
          safe = false;
        return;
      }
      // Ops carrying regions are covered by walking their contents, and
      // terminators move no data; neither declares memory effects, so asking
      // would disqualify every loop.
      if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
        return;
      // Anything else that writes, or whose effects are unknown, disqualifies
      // the loop. isMemoryEffectFreeOrOnlyRead covers both cases.
      if (!isMemoryEffectFreeOrOnlyRead(op)) {
        LDBG("noalias: blocked by " << op->getName());
        safe = false;
      }
    });
    return safe;
  }

  bool isMemoryEffectFreeOrOnlyRead(Operation *op) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::all_of(*effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    auto mod = dyn_cast<ModuleOp>(root);
    if (!mod)
      mod = root->getParentOfType<ModuleOp>();
    assumeNoAliasArgs = mod && mod->hasAttr("im.noalias_args");

    // Walk through all loops in a function in innermost-loop-first order.
    // This way, we first LICM from the inner loop, and place the ops in the
    // outer loop, which in turn can be further LICM'ed.
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            if (!isa<LoadOp>(op))
              return isSpeculatable(op) && isMemoryEffectFree(op);
            if (!isMemoryEffectFreeOrOnlyRead(op))
              return false;
            if (!isLoopMemoryEffectFreeOrOnlyRead.contains(loopLike))
              isLoopMemoryEffectFreeOrOnlyRead[loopLike] =
                  isMemoryEffectFreeOrOnlyRead(loopLike);
            if (isLoopMemoryEffectFreeOrOnlyRead[loopLike])
              return true;
            // The loop writes. Under an explicit no-alias assertion, ask the
            // narrower question: does anything it writes alias THIS load?
            return assumeNoAliasArgs && noWriteInLoopAliases(loopLike, op);
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            // Create the new mask for load op.
            if (auto loadOp = dyn_cast<LoadOp>(op)) {
              IRRewriter rewriter(loopLike);
              Location loc = loopLike->getLoc();
              Value cond;
              if (auto forOp = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
                cond = arith::CmpIOp::create(
                    rewriter, loc, arith::CmpIPredicate::slt,
                    forOp.getLowerBound(), forOp.getUpperBound());
              } else if (auto whileOp =
                             dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
                // TODO: Support Load Op hoisting for while loop.
                return;
              } else {
                return;
              }
              Value newMask = getPredMask(rewriter, loadOp.getPtr().getType(),
                                          loadOp.getMask(), cond);
              loadOp.getMaskMutable().assign(newMask);
            }
            loopLike.moveOutOfLoop(op);
          });
    });
  }
};

} // namespace mlir::triton
