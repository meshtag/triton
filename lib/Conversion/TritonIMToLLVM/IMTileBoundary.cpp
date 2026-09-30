/*
 * IMTileBoundary — tell the trace runtime where one tile ends in a persistent
 * kernel.
 *
 * The trace runtime folds the N bank replays of one all-bank PIM command into a
 * single event and scopes that fold to a "dispatch". With one tile per program
 * instance the host marks each dispatch when it sets the program id, and the bank
 * loop sits inside the instance loop, so an instance's accesses are adjacent.
 *
 * A persistent kernel inverts that: the tile loop moves inside the kernel and the
 * bank loop is outside, so bank 0 walks every tile before bank 1 starts its first.
 * With no signal from inside the loop the entire run is one dispatch, and every
 * cross-tile re-read of the same physical address folds to nothing. That looks like
 * a large speedup and nothing warns about it.
 *
 * So the loop tells the runtime. `__pim_tile_boundary()` advances a tile index that
 * resets when the host selects a bank, which makes it an INDEX rather than a running
 * total: bank 0's tile 5 and bank 1's tile 5 get the same dispatch id and still fold,
 * while tile 5 and tile 6 do not.
 *
 * Scope. Runs only when the module carries `im.persistent`, and targets the
 * OUTERMOST scf.for in each function, which in a persistent kernel is the tile loop
 * by construction with the reduction loop nested inside. Marking a specific loop in
 * kernel source would put the decision back in the kernel and out of the compiler.
 *
 * Ordering. Must run BEFORE scf-to-cf. Afterwards the back edge is a branch and
 * finding the tile loop means recovering the structure that was just discarded.
 *
 * `im.tile_prologue_first` skips the call on the first iteration, so work hoisted
 * above the loop shares tile 0 with it. DCC loads an operand after the round's reset,
 * and the runtime's phase order within one tile then emits that order.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMTILEBOUNDARY
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kBoundaryFn = "__pim_tile_boundary";
constexpr llvm::StringLiteral kPersistentAttr = "im.persistent";
constexpr llvm::StringLiteral kPrologueFirstAttr = "im.tile_prologue_first";
/// Stamped on each loop we mark, so a reader of the IR (and the harness) can see
/// that the boundary call is there and how many loops took one.
constexpr llvm::StringLiteral kMarkedAttr = "im.tile-boundary";

/// Declare `void __pim_tile_boundary()` once per module.
static LLVM::LLVMFuncOp getOrDeclareBoundaryFn(ModuleOp mod) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(kBoundaryFn))
    return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto voidTy = LLVM::LLVMVoidType::get(mod.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {}, /*isVarArg=*/false);
  return LLVM::LLVMFuncOp::create(b, mod.getLoc(), kBoundaryFn, fnTy);
}

struct IMTileBoundaryPass
    : public triton::im::impl::IMTileBoundaryBase<IMTileBoundaryPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kPersistentAttr))
      return; // not a persistent kernel: nothing to mark, and marking would be wrong

    // Outermost scf.for per function. walk() is pre-order, so the first one seen on
    // any path is the outermost; skip anything with an scf.for ancestor.
    SmallVector<scf::ForOp> outermost;
    mod.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<scf::ForOp>())
        return;
      outermost.push_back(forOp);
    });
    if (outermost.empty()) {
      mod.emitWarning()
          << "im.persistent is set but no loop was found to mark, so the trace "
             "runtime will see the whole kernel as one dispatch and cross-tile "
             "re-reads will collapse. Either the tile loop was fully unrolled or "
             "this kernel is not persistent.";
      return;
    }

    LLVM::LLVMFuncOp fn = getOrDeclareBoundaryFn(mod);
    bool prologueFirst = mod->hasAttr(kPrologueFirstAttr);
    for (scf::ForOp forOp : outermost) {
      OpBuilder b(forOp.getContext());
      // At the TOP of the body, not before the terminator. Sitting next to the
      // terminator made the membar analysis recurse until the stack ran out while
      // lowering; the tile index counts iterations entered rather than completed,
      // which is an equally valid identity because all that matters is that it
      // differs per tile and repeats across bank replays.
      b.setInsertionPointToStart(forOp.getBody());
      if (prologueFirst) {
        Value later = arith::CmpIOp::create(b, forOp.getLoc(), arith::CmpIPredicate::ne,
                                            forOp.getInductionVar(),
                                            forOp.getLowerBound());
        auto ifOp = scf::IfOp::create(b, forOp.getLoc(), later, /*withElseRegion=*/false);
        b.setInsertionPointToStart(ifOp.thenBlock());
      }
      LLVM::CallOp::create(b, forOp.getLoc(), TypeRange{},
                           SymbolRefAttr::get(fn), ValueRange{});
      forOp->setAttr(kMarkedAttr, b.getUnitAttr());
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMTileBoundaryPass() {
  return std::make_unique<IMTileBoundaryPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
