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
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
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
/// The same with a mask of the PCU sides that record it (im.pcu_pair_accumulate). Every
/// lane still calls it, so the lanes of a PCU stay at one access ordinal.
constexpr llvm::StringLiteral kFoldSidesFn = "__pim_trace_fold_sides";
constexpr llvm::StringLiteral kPriceAttr = "im.price_folds";
/// Stamped on each marked reduce as [steps, outputs per lane], so the IR says what the
/// runtime will be told.
constexpr llvm::StringLiteral kMarkedAttr = "im.lane-fold";

static LLVM::LLVMFuncOp getOrDeclareFoldFn(ModuleOp mod, StringRef name, unsigned args) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto i64 = IntegerType::get(mod.getContext(), 64);
  auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),
                                          SmallVector<Type>(args, i64), /*isVarArg=*/false);
  return LLVM::LLVMFuncOp::create(b, mod.getLoc(), name, fnTy);
}

/// A blocked encoding's per-dim fields, or under the pair option a slice of one with the
/// sliced dim dropped, which is what the fold after a PCU pair reduce reads.
struct LaneFields {
  SmallVector<unsigned> spt, order, tpw, wpc;
};
static std::optional<LaneFields> laneFields(Attribute enc, bool slices) {
  if (auto bl = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc))
    return LaneFields{SmallVector<unsigned>(bl.getSizePerThread()),
                      SmallVector<unsigned>(bl.getOrder()),
                      SmallVector<unsigned>(bl.getThreadsPerWarp()),
                      SmallVector<unsigned>(bl.getWarpsPerCTA())};
  auto sl = dyn_cast_or_null<ttg::SliceEncodingAttr>(enc);
  if (!slices || !sl)
    return std::nullopt;
  auto parent = laneFields(sl.getParent(), /*slices=*/false);
  if (!parent)
    return std::nullopt;
  unsigned cut = sl.getDim();
  LaneFields f;
  for (unsigned d = 0; d < parent->spt.size(); ++d)
    if (d != cut) {
      f.spt.push_back(parent->spt[d]);
      f.tpw.push_back(parent->tpw[d]);
      f.wpc.push_back(parent->wpc[d]);
    }
  for (unsigned d : parent->order)
    if (d != cut)
      f.order.push_back(d > cut ? d - 1 : d);
  return f;
}

struct IMLaneFoldPass : public triton::im::impl::IMLaneFoldBase<IMLaneFoldPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kPriceAttr))
      return;
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();

    const bool pair = mod->hasAttr("im.pcu_pair_accumulate");
    struct Mark {
      triton::ReduceOp op;
      int64_t steps, outputs;
      bool oddOnly;
    };
    SmallVector<Mark> marks;
    bool unknown = false;
    mod.walk([&](triton::ReduceOp red) {
      auto ty = dyn_cast<RankedTensorType>(red.getOperands()[0].getType());
      if (!ty)
        return;
      auto fields = laneFields(ty.getEncoding(), pair);
      if (!fields) {
        unknown = true;
        return;
      }
      unsigned axis = red.getAxis();
      ArrayRef<unsigned> spt = fields->spt, order = fields->order, tpw = fields->tpw,
                         wpc = fields->wpc;
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
      // Lane bit 0 free: the pair reduce left the PCU's sum in both lanes, and only the
      // odd bank, which owns the GRF_B entry, folds it.
      bool oddOnly = false;
      if (pair) {
        auto lane = StringAttr::get(mod.getContext(), "lane");
        triton::LinearLayout ll = ttg::toLinearLayout(ty);
        oddOnly = ll.hasInDim(lane) && !ll.getBases().lookup(lane).empty() &&
                  llvm::all_of(ll.getBases().lookup(lane)[0], [](int32_t x) { return x == 0; });
      }
      marks.push_back({red, (int64_t)llvm::Log2_64_Ceil(lanes), perThread / along, oddOnly});
    });
    // A layout this pass cannot read would leave a fold unpriced without a trace of it.
    if (unknown)
      mod.emitWarning() << "im-lane-fold: a reduce operand is not a blocked layout; "
                           "any fold it performs is not priced";
    if (marks.empty())
      return;

    for (Mark &m : marks) {
      OpBuilder b(m.op);
      b.setInsertionPointAfter(m.op);
      Location loc = m.op.getLoc();
      SmallVector<Value> args{arith::ConstantIntOp::create(b, loc, m.steps, 64),
                              arith::ConstantIntOp::create(b, loc, m.outputs, 64)};
      if (m.oddOnly)
        args.push_back(arith::ConstantIntOp::create(b, loc, /*odd side=*/2, 64));
      LLVM::LLVMFuncOp fn = m.oddOnly ? getOrDeclareFoldFn(mod, kFoldSidesFn, 3)
                                      : getOrDeclareFoldFn(mod, kFoldFn, 2);
      LLVM::CallOp::create(b, loc, TypeRange{}, SymbolRefAttr::get(fn), args);
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
