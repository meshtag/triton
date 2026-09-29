/*
 * IMDccTileMac -- select DCC's tile MAC for a contraction whose per-lane tile is one.
 *
 * DCC's PIM_MAC_OP1 multiplies an n_mac x n_mac tile, n_mac k values into the 16 outputs
 * of one column, in one command (gen_trace_HBMPIM_GEMV.py:136-145). Our lane issues one
 * command per column it reads. A reduce over load * broadcast(operand) is that tile when
 * the load's lane-fastest axis holds one column of outputs and the next is the reduce axis
 * with whole tiles of k, so n_mac consecutive column reads of the lane are one tile. The
 * load is stamped with the reads one MAC covers and the runtime issues one command per
 * that many. This adopts DCC's ISA pricing, not Samsung's lane-wise MAC, so it is opt-in
 * (`im.dcc_tile_mac`) and, like im-relu-opcode, a convention of the DCC path. A load of
 * an argument that is also read any other way is refused rather than guessed.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMDCCTILEMAC
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.dcc_tile_mac";
/// Stamped on a selected load as [k per tile, outputs per tile, reads per MAC].
constexpr llvm::StringLiteral kMarkedAttr = "im.dcc-tile-mac";

static Value skipConvert(Value v) {
  while (auto c = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = c.getSrc();
  return v;
}

static bool isBroadcastOperand(Value v) {
  v = skipConvert(v);
  Operation *d = v.getDefiningOp();
  return d && isa<triton::BroadcastOp, triton::ExpandDimsOp>(d);
}

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

/// Reads one DCC MAC covers for this load, or 0 when its lane tile is not DCC's.
static int64_t readsPerMac(triton::LoadOp ld, unsigned axis, int64_t dqBits) {
  auto ty = dyn_cast<RankedTensorType>(ld.getType());
  auto enc = ty ? dyn_cast_or_null<ttg::BlockedEncodingAttr>(ty.getEncoding()) : nullptr;
  if (!enc)
    return 0;
  auto spt = enc.getSizePerThread();
  auto order = enc.getOrder();
  auto tpw = enc.getThreadsPerWarp();
  int64_t vpc = std::max<int64_t>(1, dqBits / ty.getElementType().getIntOrFloatBitWidth());
  SmallVector<unsigned> wide;
  for (unsigned d : order)
    if (spt[d] > 1)
      wide.push_back(d);
  // Lane-fastest axis: one column of outputs. Next: the reduce axis, whole tiles of k.
  if (wide.size() < 2 || wide[0] == axis || spt[wide[0]] != vpc || tpw[wide[0]] != 1)
    return 0;
  if (wide[1] != axis || spt[axis] % vpc != 0 || tpw[axis] != 1)
    return 0;
  return vpc; // n_mac k values x one column of outputs
}

struct IMDccTileMacPass : public triton::im::impl::IMDccTileMacBase<IMDccTileMacPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(kGateAttr))
      return;
    int64_t dqBits = 256;
    if (auto a = mod->getAttrOfType<IntegerAttr>("im.dq_bits"))
      dqBits = a.getInt();

    llvm::DenseMap<Value, SmallVector<std::pair<triton::LoadOp, int64_t>>> picked;
    mod.walk([&](triton::ReduceOp red) {
      if (red.getNumOperands() != 1)
        return;
      bool add = false;
      red.getCombineOp().walk([&](arith::AddFOp) { add = true; });
      auto mul = skipConvert(red.getOperands()[0]).getDefiningOp<arith::MulFOp>();
      if (!add || !mul)
        return;
      for (int side = 0; side < 2; ++side) {
        auto ld = skipConvert(mul->getOperand(side)).getDefiningOp<triton::LoadOp>();
        if (!ld || !isBroadcastOperand(mul->getOperand(1 - side)))
          continue;
        int64_t r = readsPerMac(ld, red.getAxis(), dqBits);
        Value arg = baseArg(ld.getPtr());
        if (r && arg)
          picked[arg].push_back({ld, r});
      }
    });
    // Every load of a selected argument has to be one of the tiles, or the per-tensor
    // count the runtime keeps would mix tiles with plain column reads.
    llvm::DenseMap<Value, int64_t> loadsOf;
    mod.walk([&](triton::LoadOp ld) {
      if (Value a = baseArg(ld.getPtr()))
        loadsOf[a]++;
    });
    for (auto &kv : picked) {
      int64_t r = kv.second.front().second;
      bool uniform = llvm::all_of(kv.second, [&](auto &p) { return p.second == r; });
      if (!uniform || loadsOf[kv.first] != (int64_t)kv.second.size())
        continue;
      for (auto &p : kv.second)
        p.first->setAttr(kMarkedAttr, DenseI64ArrayAttr::get(mod.getContext(), {r, r, r}));
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMDccTileMacPass() {
  return std::make_unique<IMDccTileMacPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
