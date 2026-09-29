/*
 * IMLaneFold -- tell the trace runtime when a reduction folds the lanes of one DRAM
 * column.
 *
 * A reduction along the axis a lane's vector loads made contiguous adds values that
 * sat in one column. The PIM MAC is lane-parallel, so that fold is a tree of
 * log2(values) steps which no memory access represents, and DCC prices each step as a
 * command (gen_trace_HBMPIM_RED.py:89-93). A reduction along any other axis adds
 * across columns, which is the MAC's own accumulation and needs no marker.
 *
 * Runs only when the module carries `im.price_folds`, because only the dcc-parity
 * runtime defines `__pim_trace_fold`. Must run before scf-to-cf, like im-tile-boundary,
 * so a fold inside a loop is marked once per iteration.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMLANEFOLD
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kFoldFn = "__pim_trace_fold";
constexpr llvm::StringLiteral kPriceAttr = "im.price_folds";
/// Stamped on each marked reduce as [steps, outputs per lane], so the IR says what the
/// runtime will be told.
constexpr llvm::StringLiteral kMarkedAttr = "im.lane-fold";

static LLVM::LLVMFuncOp getOrDeclareFoldFn(ModuleOp mod) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(kFoldFn))
    return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto i64 = IntegerType::get(mod.getContext(), 64);
  auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),
                                          {i64, i64}, /*isVarArg=*/false);
  return LLVM::LLVMFuncOp::create(b, mod.getLoc(), kFoldFn, fnTy);
}

struct IMLaneFoldPass : public triton::im::impl::IMLaneFoldBase<IMLaneFoldPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kPriceAttr))
      return;
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();

    struct Mark {
      triton::ReduceOp op;
      int64_t steps, outputs;
    };
    SmallVector<Mark> marks;
    bool unknown = false;
    mod.walk([&](triton::ReduceOp red) {
      auto ty = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!ty)
        return;
      auto enc = dyn_cast_or_null<ttg::BlockedEncodingAttr>(ty.getEncoding());
      if (!enc) {
        unknown = true;
        return;
      }
      unsigned axis = red.getAxis();
      auto spt = enc.getSizePerThread();
      auto order = enc.getOrder();
      auto tpw = enc.getThreadsPerWarp();
      auto wpc = enc.getWarpsPerCTA();
      // In-column when the reduced axis is the lane's register-fastest one, the first
      // in `order` with more than one value per lane. Placement follows access order,
      // so that run of values is what shares a column.
      int fastest = -1;
      for (unsigned d : order)
        if (spt[d] > 1) {
          fastest = (int)d;
          break;
        }
      if (fastest != (int)axis)
        return;
      int64_t bits = ty.getElementType().getIntOrFloatBitWidth();
      int64_t vpc = std::max<int64_t>(1, dqBits / bits);
      int64_t lanes = std::min<int64_t>(spt[axis], vpc);
      int64_t perThread = 1, along = 1;
      for (unsigned d = 0; d < ty.getRank(); ++d) {
        int64_t n = std::max<int64_t>(1, ty.getShape()[d] / ((int64_t)tpw[d] * wpc[d]));
        perThread *= n;
        if (d == axis)
          along = n;
      }
      marks.push_back({red, (int64_t)llvm::Log2_64_Ceil(lanes), perThread / along});
    });
    // A layout this pass cannot read would leave a fold unpriced without a trace of it.
    if (unknown)
      mod.emitWarning() << "im-lane-fold: a reduce operand is not a blocked layout; "
                           "any fold it performs is not priced";
    if (marks.empty())
      return;

    LLVM::LLVMFuncOp fn = getOrDeclareFoldFn(mod);
    for (Mark &m : marks) {
      OpBuilder b(m.op);
      b.setInsertionPointAfter(m.op);
      Location loc = m.op.getLoc();
      Value steps = arith::ConstantIntOp::create(b, loc, m.steps, 64);
      Value outputs = arith::ConstantIntOp::create(b, loc, m.outputs, 64);
      LLVM::CallOp::create(b, loc, TypeRange{}, SymbolRefAttr::get(fn),
                           ValueRange{steps, outputs});
      m.op->setAttr(kMarkedAttr, b.getDenseI64ArrayAttr({m.steps, m.outputs}));
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMLaneFoldPass() {
  return std::make_unique<IMLaneFoldPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
