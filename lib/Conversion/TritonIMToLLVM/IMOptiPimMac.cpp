/*
 * IMOptiPimMac: lower the kernel's one contraction to MACs of the PE OptiPIM's machine
 * declares.
 *
 * Their shared config gives the PE 16-bit values in a 256-bit word, one register and no
 * accumulator, and their codegen gives each iteration point of a step its own slot, 16 to a
 * word (fimdram.cpp:25, 286-299). One MAC therefore covers 16 iteration points, and a lane
 * runs ceil(cells / 16) MACs per contraction, its cells being the product's elements that
 * the factors' masks leave on, lane terms included, so a padded lane runs none. The pass
 * puts a loop of `__pim_opt_mac(j)` calls right after the reduce, so a dispatch that skips
 * the contraction issues none, and states
 *   __pim_opt_macs      MACs per contraction, one entry per lane
 *   __pim_opt_mac_geom  [values per MAC, row from t, row from j, column from t, column
 *                       from j], their alloc new putting MAC j of step t at row t, column j
 *   __pim_opt_step      [x, y, z, tile, constant], step t from the program ids and the
 *                       tile index, read off the tile loop and the program ids' decode
 * Only the optipim-parity runtime defines the call, so it is opt-in (`im.optipim_mac`).
 */
#include "IMContraction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonIMToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>

namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMOPTIPIMMAC
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"
} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::im::contraction;
namespace ttg = mlir::triton::gpu;

namespace {

constexpr llvm::StringLiteral kGateAttr = "im.optipim_mac";
constexpr llvm::StringLiteral kMacFn = "__pim_opt_mac";
/// Stamped on each MAC loop, so the IR shows where the calls are.
constexpr llvm::StringLiteral kMarkedAttr = "im.optipim-mac";
constexpr llvm::StringLiteral kTileLoopAttr = "im.tile-boundary";
/// Their PE's value width. Their word is the module's dq over it (fimdram.cpp:25).
constexpr int64_t kValueBits = 16;

static triton::LoadOp loadOf(Value v) {
  while (Operation *d = v.getDefiningOp()) {
    if (auto ld = dyn_cast<triton::LoadOp>(d))
      return ld;
    if (!isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp,
             ttg::ConvertLayoutOp, arith::ExtFOp, arith::TruncFOp>(d))
      return {};
    v = d->getOperand(0);
  }
  return {};
}

/// The extent a kernel gives one program id: the product of the radices of its decode, a
/// chain of `% R` and `/ R` that ends in a lone `% R`. nullopt when a use reads the id as a
/// number, so the extent is open. Fails on a decode that drops the digits below a radix.
static FailureOr<std::optional<int64_t>> pidExtent(Value pid) {
  int64_t ext = 1;
  for (Value v = pid;;) {
    std::optional<int64_t> radix;
    Value quot;
    bool rem = false;
    for (Operation *u : v.getUsers()) {
      bool isRem = isa<arith::RemSIOp, arith::RemUIOp>(u);
      bool isDiv = isa<arith::DivSIOp, arith::DivUIOp>(u);
      std::optional<int64_t> r;
      if ((isRem || isDiv) && u->getOperand(0) == v)
        r = getConstantIntValue(u->getOperand(1));
      if (!r || *r < 1)
        return std::optional<int64_t>();
      if ((radix && *radix != *r) || (isDiv && quot))
        return failure();
      radix = r;
      rem |= isRem;
      if (isDiv)
        quot = u->getResult(0);
    }
    if (!radix)
      return std::optional<int64_t>(ext);
    if (!rem || ext > INT32_MAX / *radix)
      return failure();
    ext *= *radix;
    if (!quot)
      return std::optional<int64_t>(ext);
    v = quot;
  }
}

static LLVM::LLVMFuncOp getOrDeclareMacFn(ModuleOp mod) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(kMacFn))
    return fn;
  MLIRContext *ctx = mod.getContext();
  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                          {IntegerType::get(ctx, 32)}, /*isVarArg=*/false);
  auto fn = LLVM::LLVMFuncOp::create(b, mod.getLoc(), kMacFn, fnTy);
  // It touches only the runtime's own state, so LLVM treats the kernel's loads and stores
  // around it as it would without the call.
  using MR = LLVM::ModRefInfo;
  fn.setMemoryEffectsAttr(LLVM::MemoryEffectsAttr::get(
      ctx, MR::NoModRef, MR::NoModRef, MR::ModRef, MR::NoModRef, MR::NoModRef, MR::NoModRef));
  fn.setNoUnwind(true);
  fn.setWillReturn(true);
  return fn;
}

static LogicalResult lowerMacs(ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  auto dq = mod->getAttrOfType<IntegerAttr>("im.dq_bits");
  if (!dq || dq.getInt() < kValueBits || dq.getInt() % kValueBits)
    return mod.emitError() << "im.optipim_mac packs their word, dq over pe_bits, and needs "
                              "the module's bus width (im.dq_bits) in multiples of "
                           << kValueBits << " bits";
  // Their SIMDRAM stands each value pe_bits rows tall in one bitline, so a word holds dq.
  const bool vertical = mod->hasAttr("im.optipim_vertical");
  const int64_t kValuesPerMac = vertical ? dq.getInt() : dq.getInt() / kValueBits;

  SmallVector<std::pair<triton::ReduceOp, arith::MulFOp>> found;
  mod.walk([&](triton::ReduceOp red) {
    if (red.getNumOperands() != 1)
      return;
    bool adds = false;
    red.getCombineOp().walk([&](arith::AddFOp) { adds = true; });
    if (auto mul = skipConvert(red.getOperands()[0]).getDefiningOp<arith::MulFOp>();
        mul && adds)
      found.push_back({red, mul});
  });
  if (found.size() != 1)
    return mod.emitError() << "im.optipim_mac lowers exactly one contraction, a sum over a "
                              "product, and this kernel has " << found.size();
  auto [red, mul] = found.front();
  triton::LoadOp in = loadOf(mul.getLhs()), wt = loadOf(mul.getRhs());
  if (!in || !wt)
    return mul.emitError() << "im.optipim_mac needs both factors of the product loaded";
  for (triton::LoadOp ld : {in, wt})
    if (unsigned bits = getElementTypeOrSelf(ld.getType()).getIntOrFloatBitWidth();
        bits != kValueBits)
      return ld.emitError() << "im.optipim_mac: their PE takes " << kValueBits
                            << "-bit values, and this factor loads " << bits << "-bit ones";

  auto prodTy = cast<RankedTensorType>(mul.getType());
  auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(prodTy.getEncoding());
  if (!blocked || llvm::any_of(blocked.getWarpsPerCTA(), [](unsigned w) { return w != 1; }))
    return mul.emitError() << "im.optipim_mac needs a blocked product layout on one warp";
  int64_t rank = prodTy.getRank();
  ArrayRef<int64_t> shape = prodTy.getShape();

  // Each dim's bound from the factors' masks, lane dims included.
  SmallVector<int64_t> bound(shape);
  for (triton::LoadOp ld : {in, wt}) {
    Value mask = ld.getMask();
    if (!mask)
      continue;
    auto ldTy = dyn_cast<RankedTensorType>(ld.getType());
    if (!ldTy || ldTy.getRank() != rank)
      return ld.emitError() << "im.optipim_mac counts a masked factor only when it is "
                               "loaded at the product's rank";
    for (MaskTerm t : maskTerms(mask, rank)) {
      if (t.constant) {
        if (*t.constant == 0)
          bound[0] = 0;
        continue;
      }
      if (t.opaque || t.dim < 0 || !t.pure || !t.bound ||
          t.cmp.getPredicate() != arith::CmpIPredicate::slt)
        return ld.emitError() << "im.optipim_mac counts a lane's cells from mask terms "
                                 "`range < extent` only, on lane dimensions too";
      bound[t.dim] = std::min(bound[t.dim], std::max<int64_t>(*t.bound, 0));
    }
  }

  // Exact cells per lane: the distinct product elements a lane holds inside every bound.
  triton::LinearLayout ll = ttg::toLinearLayout(prodTy);
  auto kReg = StringAttr::get(ctx, "register"), kLane = StringAttr::get(ctx, "lane");
  int32_t nRegs = ll.getInDimSize(kReg), nLanes = ll.getInDimSize(kLane);
  SmallVector<int64_t> cells(nLanes);
  for (int32_t l = 0; l < nLanes; ++l) {
    llvm::DenseSet<int64_t> on;
    for (int32_t r = 0; r < nRegs; ++r) {
      SmallVector<std::pair<StringAttr, int32_t>> ins;
      for (StringAttr d : ll.getInDimNames())
        ins.push_back({d, d == kReg ? r : d == kLane ? l : 0});
      SmallVector<int64_t> x(rank, 0);
      for (auto [name, v] : ll.apply(ins)) {
        StringRef dimName = name.getValue();
        unsigned d = 0;
        if (!dimName.consume_front("dim") || dimName.getAsInteger(10, d) || d >= rank)
          return mul.emitError() << "im.optipim_mac cannot read the product layout's dims";
        x[d] = v;
      }
      bool live = true;
      int64_t flat = 0;
      for (int64_t d = 0; d < rank; ++d) {
        live &= x[d] < bound[d];
        flat = flat * shape[d] + x[d];
      }
      if (live)
        on.insert(flat);
    }
    cells[l] = on.size();
  }

  // The same count as arithmetic on the lane id: a lane dim leaves clamp(bound - base, 0,
  // spt) of its elements on, base being the lane's first element along it. Held to the
  // exact count, so the loop runs what the statement says.
  auto spt = blocked.getSizePerThread(), tpw = blocked.getThreadsPerWarp();
  SmallVector<int64_t> shift(rank, 0);
  int64_t lanes = 1;
  for (unsigned d : blocked.getOrder()) {
    shift[d] = llvm::Log2_64(lanes);
    lanes *= tpw[d];
  }
  auto along = [&](int64_t d, int64_t l) -> int64_t {
    if (tpw[d] == 1)
      return std::min(bound[d], shape[d]);
    int64_t base = ((l >> shift[d]) & (tpw[d] - 1)) * spt[d];
    return std::clamp<int64_t>(bound[d] - base, 0, spt[d]);
  };
  SmallVector<int64_t> laneDims;
  int64_t uniform = 1;
  for (int64_t d = 0; d < rank; ++d) {
    if (tpw[d] > 1 && bound[d] < shape[d])
      laneDims.push_back(d);
    else
      uniform *= along(d, 0);
  }
  if (lanes != nLanes)
    return mul.emitError() << "im.optipim_mac: the product's lane dims hold " << lanes
                           << " lanes, its layout " << nLanes;
  for (int32_t l = 0; l < nLanes; ++l) {
    int64_t c = uniform;
    for (int64_t d : laneDims)
      c *= along(d, l);
    if (c != cells[l])
      return mul.emitError() << "im.optipim_mac: lane " << l << " holds " << cells[l]
                             << " cells, which its dims do not factor into (" << c << ")";
  }

  // t: a tile loop around the contraction counts steps inside a program, and the program
  // ids count programs, each slower one by the extent of the faster ones.
  scf::ForOp tile;
  bool tiled = false;
  mod.walk([&](scf::ForOp f) { tiled |= f->hasAttr(kTileLoopAttr); });
  for (Operation *p = red->getParentOp(); p && !isa<triton::FuncOp>(p);
       p = p->getParentOp()) {
    if (auto f = dyn_cast<scf::ForOp>(p); f && f->hasAttr(kTileLoopAttr) && !tile)
      tile = f;
    else if (isa<LoopLikeOpInterface>(p))
      return red.emitError() << "im.optipim_mac: this contraction sits in a loop inside one "
                                "step, so the step would issue its MACs more than once";
  }
  if (tiled && !tile)
    return red.emitError() << "im.optipim_mac: the kernel loops over tiles and this "
                              "contraction is outside the tile loop, so it has no step";
  int64_t perProgram = 1;
  if (tile) {
    auto lb = getConstantIntValue(tile.getLowerBound());
    auto ub = getConstantIntValue(tile.getUpperBound());
    auto st = getConstantIntValue(tile.getStep());
    if (!lb || !ub || !st || *st < 1)
      return tile.emitError() << "im.optipim_mac needs the tile loop's trip count constant";
    perProgram = *ub > *lb ? llvm::divideCeil(*ub - *lb, *st) : 0;
  }
  // The tile index counts boundary calls, and only a prologue-first loop skips the first.
  std::array<int64_t, 5> step{0, 0, 0, tile ? 1 : 0,
                              tile && !mod->hasAttr("im.tile_prologue_first") ? -1 : 0};
  std::array<triton::GetProgramIdOp, 3> pids{};
  bool twice = false;
  mod.walk([&](triton::GetProgramIdOp op) {
    twice |= (bool)pids[op.getAxisAsInt()];
    pids[op.getAxisAsInt()] = op;
  });
  if (twice)
    return mod.emitError() << "im.optipim_mac reads each program id once";
  int64_t stride = std::max<int64_t>(perProgram, 1);
  bool open = false;
  for (int a = 0; a < 3; ++a) {
    if (!pids[a] || pids[a]->use_empty())
      continue;
    auto ext = pidExtent(pids[a].getResult());
    if (open || failed(ext))
      return pids[a].emitError() << "im.optipim_mac cannot state t: program id " << a
                                 << (open ? " is decoded after a faster one whose extent "
                                            "is open"
                                          : " has a decode that is not a radix chain");
    step[a] = stride;
    if (*ext && stride <= INT32_MAX / **ext)
      stride *= **ext;
    else
      open = true;
  }

  // The loop of MACs where the contraction executes, its trip count from the lane id.
  auto func = red->getParentOfType<triton::FuncOp>();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(&func.getBody().front());
  Location loc = red.getLoc();
  Type i32 = b.getI32Type();
  auto cst = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(b, loc, v, 32); };
  Value cellsOn = cst(uniform);
  if (!laneDims.empty()) {
    Value tid = mlir::gpu::ThreadIdOp::create(b, loc, mlir::gpu::Dimension::x);
    Value lane = arith::IndexCastOp::create(b, loc, i32, tid);
    for (int64_t d : laneDims) {
      Value at = arith::AndIOp::create(
          b, loc, arith::ShRUIOp::create(b, loc, lane, cst(shift[d])), cst(tpw[d] - 1));
      Value left = arith::SubIOp::create(b, loc, cst(bound[d]),
                                         arith::MulIOp::create(b, loc, at, cst(spt[d])));
      Value on = arith::MaxSIOp::create(
          b, loc, arith::MinSIOp::create(b, loc, left, cst(spt[d])), cst(0));
      cellsOn = arith::MulIOp::create(b, loc, cellsOn, on);
    }
  }
  Value macs = arith::DivUIOp::create(
      b, loc, arith::AddIOp::create(b, loc, cellsOn, cst(kValuesPerMac - 1)),
      cst(kValuesPerMac));
  LLVM::LLVMFuncOp fn = getOrDeclareMacFn(mod);
  b.setInsertionPointAfter(red);
  auto loop = scf::ForOp::create(
      b, loc, cst(0), macs, cst(1), ValueRange{},
      [&](OpBuilder &nb, Location l, Value j, ValueRange) {
        LLVM::CallOp::create(nb, l, TypeRange{}, SymbolRefAttr::get(fn), ValueRange{j});
        scf::YieldOp::create(nb, l);
      });
  loop->setAttr(kMarkedAttr, b.getUnitAttr());

  OpBuilder g(mod.getBodyRegion());
  g.setInsertionPointToStart(mod.getBody());
  auto table = [&](StringRef name, ArrayRef<int32_t> v) {
    auto dataTy = RankedTensorType::get({(int64_t)v.size()}, i32);
    LLVM::GlobalOp::create(g, mod.getLoc(), LLVM::LLVMArrayType::get(i32, v.size()),
                           /*isConstant=*/true, LLVM::Linkage::External, name,
                           DenseElementsAttr::get(dataTy, v));
  };
  SmallVector<int32_t> perLane;
  for (int64_t c : cells)
    perLane.push_back((int32_t)llvm::divideCeil(c, kValuesPerMac));
  table("__pim_opt_macs", perLane);
  LLVM::GlobalOp::create(g, mod.getLoc(), i32, /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_macs_count", g.getI32IntegerAttr(nLanes));
  table("__pim_opt_mac_geom", {(int32_t)kValuesPerMac, 1, 0, 0, 1});
  LLVM::GlobalOp::create(g, mod.getLoc(), i32, /*isConstant=*/true, LLVM::Linkage::External,
                         "__pim_opt_vertical", g.getI32IntegerAttr(vertical ? kValueBits : 0));
  SmallVector<int32_t> stepFlat(step.begin(), step.end());
  table("__pim_opt_step", stepFlat);
  return success();
}

struct IMOptiPimMacPass : public triton::im::impl::IMOptiPimMacBase<IMOptiPimMacPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (mod->hasAttr(kGateAttr) && failed(lowerMacs(mod)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace im {
std::unique_ptr<OperationPass<ModuleOp>> createIMOptiPimMacPass() {
  return std::make_unique<IMOptiPimMacPass>();
}
} // namespace im
} // namespace triton
} // namespace mlir
