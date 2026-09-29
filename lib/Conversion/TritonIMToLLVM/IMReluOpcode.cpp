/*
 * IMReluOpcode -- mark the loads a ReLU consumes, so the dcc-parity runtime can issue
 * them as DCC's PIM_RELU instead of a MAC.
 *
 * This adopts DCC's treatment of ReLU on purpose. Their simulator spaces LD, ADD, MUL and
 * MAC_OP on the command bus and not PIM_RELU (HBM3-PIM.cpp:489-510), so that command
 * costs less than the MAC our emitter otherwise issues for the same read. A load
 * qualifies when its only use, through float casts, is a max against zero whose result
 * is only stored, which is DCC's RELU shape (PIM_RELU then PIM_WB_RES). A ReLU feeding
 * further arithmetic keeps its MAC, since that read also carries the arithmetic. The
 * marker calls bracket the load, so every access it lowers to carries the opcode.
 *
 * Runs only when the module carries `im.dcc_relu_opcode`, because only the dcc-parity
 * runtime defines `__pim_trace_op`. Must run before scf-to-cf.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMRELUOPCODE
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kOpFn = "__pim_trace_op";
constexpr llvm::StringLiteral kGateAttr = "im.dcc_relu_opcode";
constexpr llvm::StringLiteral kMarkedAttr = "im.relu-opcode";
constexpr int32_t kOpNone = 0, kOpRelu = 1;

static bool isZero(Value v) {
  if (auto s = v.getDefiningOp<triton::SplatOp>())
    return isZero(s.getSrc());
  auto c = v.getDefiningOp<arith::ConstantOp>();
  if (!c)
    return false;
  if (auto d = dyn_cast<DenseFPElementsAttr>(c.getValue()))
    return d.isSplat() && d.getSplatValue<APFloat>().isZero();
  if (auto f = dyn_cast<FloatAttr>(c.getValue()))
    return f.getValue().isZero();
  return false;
}

/// The value's only use, through float casts, is a tt.store of it.
static bool onlyStored(Value v) {
  if (!v.hasOneUse())
    return false;
  Operation *u = *v.getUsers().begin();
  if (isa<arith::ExtFOp, arith::TruncFOp>(u))
    return onlyStored(u->getResult(0));
  auto st = dyn_cast<triton::StoreOp>(u);
  return st && st.getValue() == v;
}

/// The value's only use, through float casts, is max(value, 0), and that is only stored.
static bool feedsReluOnly(Value v) {
  if (!v.hasOneUse())
    return false;
  Operation *u = *v.getUsers().begin();
  if (isa<arith::ExtFOp, arith::TruncFOp>(u))
    return feedsReluOnly(u->getResult(0));
  if (!isa<arith::MaxNumFOp, arith::MaximumFOp>(u))
    return false;
  return isZero(u->getOperand(0) == v ? u->getOperand(1) : u->getOperand(0)) &&
         onlyStored(u->getResult(0));
}

static LLVM::LLVMFuncOp getOrDeclareOpFn(ModuleOp mod) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(kOpFn))
    return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),
                                          {IntegerType::get(mod.getContext(), 32)},
                                          /*isVarArg=*/false);
  return LLVM::LLVMFuncOp::create(b, mod.getLoc(), kOpFn, fnTy);
}

struct IMReluOpcodePass
    : public triton::im::impl::IMReluOpcodeBase<IMReluOpcodePass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kGateAttr))
      return;
    SmallVector<triton::LoadOp> loads;
    mod.walk([&](triton::LoadOp ld) {
      if (feedsReluOnly(ld.getResult()))
        loads.push_back(ld);
    });
    if (loads.empty())
      return;
    LLVM::LLVMFuncOp fn = getOrDeclareOpFn(mod);
    auto call = [&](OpBuilder &b, Location loc, int32_t op) {
      Value v = arith::ConstantIntOp::create(b, loc, op, 32);
      LLVM::CallOp::create(b, loc, TypeRange{}, SymbolRefAttr::get(fn), ValueRange{v});
    };
    for (triton::LoadOp ld : loads) {
      OpBuilder b(ld);
      call(b, ld.getLoc(), kOpRelu);
      b.setInsertionPointAfter(ld);
      call(b, ld.getLoc(), kOpNone);
      ld->setAttr(kMarkedAttr, b.getUnitAttr());
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMReluOpcodePass() {
  return std::make_unique<IMReluOpcodePass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
