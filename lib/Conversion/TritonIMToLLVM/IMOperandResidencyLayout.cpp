/// IMOperandResidencyLayout.cpp. Decide per-operand physical layout and residency
/// for the in-memory (IM) backend.
///
/// Target scope: this pass is substrate-agnostic. Its classification is a pure IR
/// analysis of the kernel's SSA and loop structure with no architecture branch, so
/// it runs for every IM target, both near-memory PIM (HBM-PIM) and
/// processing-using-memory PUM (SIMDRAM). Today the stamped attributes are honored
/// only by the HBM-PIM runtime. The SIMDRAM runtime has no residency-as-layout
/// consumer, so on SIMDRAM the pass runs but is inert. A PUM backend could consume
/// the same classification (bank and subarray parallelism and reduction-to-column
/// are meaningful there too), it just does not yet.
///
/// Phase 1 (classification): for each tensor-of-pointer load or store, and each
/// scalar load that feeds a broadcast (the matvec `x[k]` operand), classify the
/// operand's reuse axes and stamp an `im.residency` dictionary attribute that a
/// later runtime increment will honor. Classification is purely an IR analysis. It
/// changes NO simulated behaviour until a consumer reads the attrs.
///
///   reuse_class is decided from two reachability facts about the address:
///     pidMask: does the address depend on tt.get_program_id (output tile)?
///     ivDep:   does the address depend on the reduction loop induction var?
///   In a function containing a reduction (accumulator-carrying scf.for):
///     ivDep & pid   -> ReductionStridedMatrix  (reduction axis -> columns)
///     ivDep & !pid  -> BroadcastReplicate      (invariant in output tile)
///     !ivDep & pid  -> ParallelSpread          (output, spread over banks)
///     otherwise     -> StreamedElementwise
///   With no reduction loop (pure elementwise, e.g. axpy) every op is
///   StreamedElementwise.
///
/// Phase 2 (footprint): the element offsets one program instance touches, as
/// (extent, stride) axes over the tl.arange axes and the reduction induction
/// variable. It stays logical. The register-slot count those offsets need
/// depends on the bank/row/column interleaving, which lives in the runtime, and
/// a stride the pass cannot fold (conv's CI*R*S weight pitch) travels as a
/// product of kernel argument indices for the runtime to resolve. Resolving at
/// all requires a constant trip count, so an untiled reduction gets no
/// footprint and the host allocator still sizes it.

#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

// -----------------------------------------------------------------------
// TableGen pass base class
// -----------------------------------------------------------------------
namespace mlir {
namespace triton {
namespace im {
#define GEN_PASS_DEF_IMOPERANDRESIDENCYLAYOUT
#include "triton/Conversion/TritonIMToLLVM/Passes.h.inc"

} // namespace im
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace ttg_ = mlir::triton::gpu;

/// Lane (bank) axis of the kernel's output tile, or -1.
///
/// The store's pointer type is the one 2-D encoding that is already canonical when
/// this pass runs, so it is the reliable anchor for "which axis carries the banks".
/// The per-operand expand_dims encodings are NOT: at this stage each still puts the
/// lanes on its own axis ([32,1] for the M vector, [1,32] for the N vector), and
/// they only agree on the common tile after later canonicalization.
static int laneAxisOfOutputTile(Operation *memOp) {
  auto fn = memOp->getParentOfType<triton::FuncOp>();
  if (!fn)
    return -1;
  int lane = -1;
  fn.walk([&](triton::StoreOp st) {
    auto tt = dyn_cast<RankedTensorType>(st.getPtr().getType());
    if (!tt || tt.getRank() != 2)
      return;
    auto blocked =
        dyn_cast_or_null<ttg_::BlockedEncodingAttr>(tt.getEncoding());
    if (!blocked)
      return;
    SmallVector<unsigned> tpw(blocked.getThreadsPerWarp());
    if (tpw.size() == 2)
      lane = tpw[0] > 1 ? 0 : 1;
  });
  return lane;
}

/// Does every bank see the SAME elements of this loaded operand?
///
/// Banks-as-threads means the lane axis of the output tile IS the bank axis. A
/// rank-1 operand is expand_dims'd into that tile, so it is replicated across the
/// banks exactly when its own axis is not the lane axis. For matmul with a
/// [BLOCK_M, BLOCK_N] tile and lanes on N: A expands on axis 1, occupies dim 0, and
/// every bank reads all of it; B expands on axis 0, occupies the lane dim, and is
/// partitioned.
///
/// Deriving this instead of letting the host assert it is the whole point. The
/// hard-coded role strings were written for one layout, so a layout change silently
/// inverted the charge: a transposed matmul accumulator measured 31x too fast
/// because the replicated operand landed in the role that is never charged for
/// replication (2026-09-06).
static bool isBankReplicated(Operation *memOp) {
  if (memOp->getNumResults() == 0)
    return false; // a store's role is what it is, not how it is spread
  int lane = laneAxisOfOutputTile(memOp);
  if (lane < 0)
    return false;
  SmallVector<Value, 8> work{memOp->getResult(0)};
  llvm::DenseSet<Value> seen;
  int hops = 0;
  while (!work.empty() && hops++ < 128) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (Operation *u : v.getUsers()) {
      if (auto ed = dyn_cast<triton::ExpandDimsOp>(u))
        return (ed.getAxis() == 0 ? 1 : 0) != lane;
      if (u->getNumResults() == 1)
        work.push_back(u->getResult(0)); // walk convert_layout and friends
    }
  }
  return false;
}

namespace {
// Footprint descriptor limits. The layout table record is fixed width, so
// these must match PIM_MAX_FP_AXES / PIM_MAX_STRIDE_ARGS in pim_runtime.c.
static constexpr int64_t kMaxFpAxes = 3;
static constexpr int64_t kMaxStrideArgs = 3;

// Backward def-chain reachability: bitmask over program-id axes the address
// depends on (bit a set => address derives from tt.get_program_id with axis a).
static unsigned reachesProgramId(Value v, llvm::DenseSet<Value> &visited) {
  if (!v || !visited.insert(v).second)
    return 0;
  Operation *def = v.getDefiningOp();
  if (!def)
    return 0; // block argument (func arg, loop IV, iter arg), not a pid source
  if (auto pid = dyn_cast<triton::GetProgramIdOp>(def))
    return 1u << pid.getAxisAsInt();
  unsigned mask = 0;
  for (Value operand : def->getOperands())
    mask |= reachesProgramId(operand, visited);
  return mask;
}

// Backward def-chain reachability: does the address derive from `target`
// (used with a reduction loop's induction variable)
static bool reachesValue(Value v, Value target, llvm::DenseSet<Value> &visited) {
  if (!v)
    return false;
  if (v == target)
    return true;
  if (!visited.insert(v).second)
    return false;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  for (Value operand : def->getOperands())
    if (reachesValue(operand, target, visited))
      return true;
  return false;
}

// Follow the addptr/splat/broadcast chain of an address back to the kernel
// !tt.ptr argument it bottoms out at. Return its func-argument index, or -1.
// This is the join key the runtime emitter maps to a registered tensor id.
static int64_t resolveOperandArg(Value ptr) {
  Value cur = ptr;
  llvm::DenseSet<Value> guard;
  while (cur && guard.insert(cur).second) {
    if (auto barg = dyn_cast<BlockArgument>(cur)) {
      Operation *parent = barg.getOwner()->getParentOp();
      if (isa<triton::FuncOp>(parent))
        return barg.getArgNumber();
      return -1; // a loop/region block arg, not a kernel pointer argument
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      return -1;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      cur = addptr.getPtr();
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      cur = splat.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      cur = bcast.getSrc();
      continue;
    }
    return -1; // unknown producer of a pointer value
  }
  return -1;
}

// Is `op` a scalar tt.load whose (scalar) result feeds a tt.splat?  This is the
// matvec `x_val = tl.load(x + k)` operand, invisible to the tensor-pointer
// walk because its pointer is a plain !tt.ptr, not a tensor of pointers.
/// Constant integer behind a Value, or nullopt.
static std::optional<int64_t> constIntOf(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Trip count of `f`, if all three bounds are compile-time constants.
static std::optional<int64_t> constTripCount(scf::ForOp f) {
  if (!f)
    return std::nullopt;
  auto lb = constIntOf(f.getLowerBound());
  auto ub = constIntOf(f.getUpperBound());
  auto st = constIntOf(f.getStep());
  if (!lb || !ub || !st || *st <= 0 || *ub <= *lb)
    return std::nullopt;
  return (*ub - *lb + *st - 1) / *st;
}

/// Compile-time value of a scalar (or uniformly splatted) integer, or nullopt.
static std::optional<int64_t> uniformIntOf(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
    if (auto de = dyn_cast<DenseElementsAttr>(c.getValue()))
      if (de.isSplat())
        return de.getSplatValue<APInt>().getSExtValue();
  }
  if (auto sp = v.getDefiningOp<triton::SplatOp>())
    return uniformIntOf(sp.getSrc());
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>())
    return uniformIntOf(ext.getIn());
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
    return uniformIntOf(trunc.getIn());
  return std::nullopt;
}

/// A stride the pass could not fold to a number: `factor` times the product of
/// the kernel scalar arguments named in `args`. Conv needs this, its weight row
/// pitch is CI*R*S and all three are kernel arguments.
struct SymStride {
  int64_t factor = 1;
  SmallVector<int64_t, kMaxStrideArgs> args;
  bool ok = true;
};

/// One footprint axis: the address takes `extent` positions `stride` apart.
struct FpAxis {
  int64_t extent = 0;
  SymStride stride;
};

/// Fold `v` into a SymStride, or give up. Only products of constants and kernel
/// scalar arguments are foldable, which is what index arithmetic on shapes
/// looks like.
static bool foldStride(Value v, SymStride &out, int depth = 0) {
  if (!v || depth > 16)
    return false;
  if (auto k = uniformIntOf(v)) {
    out.factor *= *k;
    return true;
  }
  if (auto sp = v.getDefiningOp<triton::SplatOp>())
    return foldStride(sp.getSrc(), out, depth + 1);
  if (auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>())
    return foldStride(cvt.getSrc(), out, depth + 1);
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>())
    return foldStride(ext.getIn(), out, depth + 1);
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
    return foldStride(trunc.getIn(), out, depth + 1);
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (!isa<triton::FuncOp>(barg.getOwner()->getParentOp()))
      return false;
    if ((int64_t)out.args.size() >= kMaxStrideArgs)
      return false;
    out.args.push_back(barg.getArgNumber());
    return true;
  }
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    return foldStride(mul.getLhs(), out, depth + 1) &&
           foldStride(mul.getRhs(), out, depth + 1);
  return false;
}

/// Accumulator keyed by the varying value (a tl.arange or the reduction IV) so
/// a variable appearing twice in one address adds its strides instead of
/// counting as two axes.
using FpAcc = llvm::MapVector<Value, FpAxis>;

/// Decompose an element offset into `const + sum stride_i * v_i` and record the
/// varying part. `scale` carries the enclosing multiplications down.
/// Report why an address was not modelled. Giving up here is silent otherwise:
/// the operand just falls back to the host budget, which looks like success.
/// IM_FP_DEBUG=1 to see it.
static bool fpBail(Value v, const char *why) {
  static int on = -1;
  if (on < 0)
    on = std::getenv("IM_FP_DEBUG") ? 1 : 0;
  if (on) {
    llvm::errs() << "[fp] bail " << why << " : ";
    if (Operation *d = v ? v.getDefiningOp() : nullptr)
      llvm::errs() << d->getName();
    else
      llvm::errs() << "<block arg>";
    llvm::errs() << "\n";
  }
  return false;
}
/// `key` names the axis. It is the outermost layout-changing op the walk went
/// through, because layout assignment CSEs one tl.arange into several tile
/// dimensions and the range value alone cannot tell them apart. Null until the
/// walk passes such an op, so a range used twice in one index expression still
/// merges into a single axis.
static bool collectAxes(Value v, Value iv, int64_t ivTrip, int64_t ivStep,
                        const SymStride &scale, FpAcc &acc, int depth,
                        Value key) {
  if (!v || depth > 32)
    return fpBail(v, "depth");
  if (uniformIntOf(v))
    return true; // shifts the base, does not spread it
  auto note = [&](Value var, int64_t extent, int64_t stepScale) {
    SymStride s = scale;
    s.factor *= stepScale;
    if (key)
      var = key;
    auto it = acc.find(var);
    if (it == acc.end()) {
      acc.insert({var, FpAxis{extent, s}});
      return;
    }
    // Same variable twice. Adding symbolic strides is not representable, so
    // keep the axis and mark it unusable rather than silently under-counting.
    if (it->second.stride.args == s.args && it->second.stride.args.empty())
      it->second.stride.factor += s.factor;
    else
      it->second.stride.ok = false;
  };
  if (iv && v == iv) {
    if (ivTrip <= 0)
      return false;
    note(v, ivTrip, ivStep);
    return true;
  }
  if (auto barg = dyn_cast<BlockArgument>(v))
    // A kernel scalar argument is fixed for the launch. Any other block arg is
    // loop-carried, which this does not model.
    return isa<triton::FuncOp>(barg.getOwner()->getParentOp());
  Operation *def = v.getDefiningOp();
  if (!def)
    return fpBail(v, "no-def");
  if (isa<triton::GetProgramIdOp>(def))
    return true; // fixed within one program instance
  if (auto mr = dyn_cast<triton::MakeRangeOp>(def)) {
    int64_t n = (int64_t)mr.getEnd() - (int64_t)mr.getStart();
    if (n <= 0)
      return false;
    note(v, n, 1);
    return true;
  }
  // ttg.convert_layout only moves values between layouts, so it is a
  // pass-through here. Missing it hid every 2D accumulator address.
  if (isa<triton::BroadcastOp, triton::ExpandDimsOp,
          triton::gpu::ConvertLayoutOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key ? key : v);
  if (isa<triton::SplatOp, arith::ExtSIOp, arith::TruncIOp,
          arith::IndexCastOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key);
  if (isa<arith::AddIOp>(def))
    return collectAxes(def->getOperand(0), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key) &&
           collectAxes(def->getOperand(1), iv, ivTrip, ivStep, scale, acc,
                       depth + 1, key);
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    for (int i = 0; i < 2; ++i) {
      SymStride sub = scale;
      if (foldStride(def->getOperand(i), sub))
        return collectAxes(def->getOperand(1 - i), iv, ivTrip, ivStep, sub, acc,
                           depth + 1, key);
    }
    return fpBail(v, "mul-nonconst-stride");
  }
  return fpBail(v, "unhandled-op");
}

/// Footprint of the whole address: walk the addptr chain back to the kernel
/// pointer argument, summing every offset on the way.
///
/// The result is the set of element offsets one program instance touches, which
/// is what the runtime turns into a register-slot count. The pass deliberately
/// stops at the logical description: how many physical slots those offsets need
/// depends on the bank/row/column interleaving, which lives in the runtime.
static bool operandFootprint(Value ptr, scf::ForOp redFor,
                             SmallVectorImpl<FpAxis> &out) {
  Value iv = redFor ? redFor.getInductionVar() : Value();
  int64_t ivTrip = 0, ivStep = 1;
  if (redFor) {
    auto t = constTripCount(redFor);
    auto s = constIntOf(redFor.getStep());
    if (!t || !s)
      return fpBail(ptr, "runtime-trip-count");
    ivTrip = *t;
    ivStep = *s;
  }
  FpAcc acc;
  Value cur = ptr;
  llvm::DenseSet<Value> guard;
  while (cur && guard.insert(cur).second) {
    if (isa<BlockArgument>(cur))
      break;
    Operation *def = cur.getDefiningOp();
    if (!def)
      return false;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      SymStride unit;
      if (!collectAxes(addptr.getOffset(), iv, ivTrip, ivStep, unit, acc, 0,
                       Value()))
        return false;
      cur = addptr.getPtr();
      continue;
    }
    if (isa<triton::SplatOp, triton::BroadcastOp>(def)) {
      cur = def->getOperand(0);
      continue;
    }
    return fpBail(cur, "ptr-chain");
  }
  for (const auto &kv : acc) {
    const FpAxis &ax = kv.second;
    if (!ax.stride.ok)
      return fpBail(kv.first, "stride-collision");
    if (ax.extent > 1 && (ax.stride.factor != 0 || !ax.stride.args.empty()))
      out.push_back(ax);
  }
  if ((int64_t)out.size() > kMaxFpAxes)
    return fpBail(ptr, "too-many-axes");
  return true;
}

static bool isScalarSplatLoad(Operation *op) {
  auto load = dyn_cast<triton::LoadOp>(op);
  if (!load)
    return false;
  if (isa<RankedTensorType>(load.getType()))
    return false; // tensor load handled by the tensor-pointer path
  for (Operation *user : load->getUsers())
    if (isa<triton::SplatOp>(user))
      return true;
  return false;
}

// -----------------------------------------------------------------------
// Pass implementation
// -----------------------------------------------------------------------

struct IMOperandResidencyLayoutPass
    : public triton::im::impl::IMOperandResidencyLayoutBase<
          IMOperandResidencyLayoutPass> {
  using IMOperandResidencyLayoutBase::IMOperandResidencyLayoutBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    // Compiler-side ablation levers (docs/ablation-levers-plan.md).
    // IM_LAYOUT_ABLATE is a comma/space list of lever names to DISABLE. A
    // disabled lever makes the pass emit the naive (no-reuse) layout for that
    // axis. Read here, inside the compiler pass, so ablation is a compiler-side
    // control rather than a simulator knob. NOTE: Triton caches compiled
    // kernels and does not key on env read inside a pass, so the ablation
    // runner must use a fresh TRITON_CACHE_DIR per level.
    const char *ablateEnv = std::getenv("IM_LAYOUT_ABLATE");
    std::string ablate = ablateEnv ? ablateEnv : "";
    auto disabled = [&](const char *lever) -> bool {
      if (ablate.empty())
        return false;
      std::string needle(lever);
      size_t pos = 0;
      while ((pos = ablate.find(needle, pos)) != std::string::npos) {
        bool lb = (pos == 0) || ablate[pos - 1] == ',' || ablate[pos - 1] == ' ';
        size_t end = pos + needle.size();
        bool rb = (end == ablate.size()) || ablate[end] == ',' ||
                  ablate[end] == ' ';
        if (lb && rb)
          return true;
        pos = end;
      }
      return false;
    };
    const bool dReductionCol = disabled("reduction-col");
    const bool dBankSpread = disabled("bank-spread");
    const bool dOperandScope = disabled("operand-scope");
    const bool dBroadcast = disabled("broadcast");
    const bool dAccResident = disabled("acc-resident");

    // Pass A, reduction-loop detection.  A reduction loop is an scf.for that
    // carries iter_args (the accumulator).  Record the count and which
    // functions contain one.
    int64_t numReductionLoops = 0;
    llvm::DenseSet<Operation *> funcsWithReduction;
    mod.walk([&](scf::ForOp forOp) {
      if (forOp.getNumRegionIterArgs() > 0) {
        ++numReductionLoops;
        if (auto fn = forOp->getParentOfType<triton::FuncOp>())
          funcsWithReduction.insert(fn.getOperation());
      }
    });

    // Pass B, classify each memory-access op and stamp im.residency.
    int64_t numAnalyzed = 0, numClassified = 0;
    mod.walk([&](Operation *op) {
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      bool isTensorPtr =
          tensorTy && isa<triton::PointerType>(tensorTy.getElementType());
      bool scalarSplat = !isTensorPtr && isScalarSplatLoad(op);
      if (!isTensorPtr && !scalarSplat)
        return;
      ++numAnalyzed;

      // Innermost enclosing reduction loop (for the induction-var test).
      scf::ForOp redFor;
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
        if (auto f = dyn_cast<scf::ForOp>(p)) {
          if (f.getNumRegionIterArgs() > 0) {
            redFor = f;
            break;
          }
        }
      }

      unsigned pidMask = 0;
      {
        llvm::DenseSet<Value> vis;
        pidMask = reachesProgramId(ptr, vis);
      }
      bool ivDep = false;
      if (redFor) {
        llvm::DenseSet<Value> vis;
        ivDep = reachesValue(ptr, redFor.getInductionVar(), vis);
      }
      int64_t operandArg = resolveOperandArg(ptr);

      auto fn = op->getParentOfType<triton::FuncOp>();
      bool funcReduction =
          fn && funcsWithReduction.contains(fn.getOperation());

      // A scalar load splatted across the lanes is a bank-broadcast: the value
      // is identical across all banks for a given access, exactly what per-BG
      // replication models, independent of pid/reduction dependence.
      StringRef cls;
      if (scalarSplat && !dBroadcast)
        cls = "BroadcastReplicate";
      else if (!funcReduction)
        cls = "StreamedElementwise";
      else if (ivDep && pidMask != 0)
        cls = "ReductionStridedMatrix";
      else if (ivDep && pidMask == 0 && !dBroadcast)
        cls = "BroadcastReplicate";
      else if (!ivDep && pidMask != 0)
        cls = "ParallelSpread";
      else
        cls = "StreamedElementwise";

      const bool isStore = isa<triton::StoreOp>(op);

      // DERIVED ROLE. Replication is a property of the layout, so read it off the
      // encoding rather than letting the host assert it.
      const bool bankReplicated = scalarSplat || isBankReplicated(op);

      SmallVector<NamedAttribute> fields;
      fields.push_back(b.getNamedAttr("reuse_class", b.getStringAttr(cls)));
      fields.push_back(b.getNamedAttr("bank_replicated",
                                      b.getBoolAttr(bankReplicated)));
      fields.push_back(
          b.getNamedAttr("operand_arg", b.getI64IntegerAttr(operandArg)));
      fields.push_back(
          b.getNamedAttr("axis_deps", b.getI64IntegerAttr((int64_t)pidMask)));
      if (ivDep)
        fields.push_back(b.getNamedAttr("reduction_dep", b.getUnitAttr()));
      // reduction-col lever. Place the contraction axis on the column-low bits.
      if (cls == "ReductionStridedMatrix" && !dReductionCol)
        fields.push_back(
            b.getNamedAttr("reduction_to_column", b.getUnitAttr()));
      // operand-scope lever. The pid axes the operand is INVARIANT in (the
      // complement of axis_deps over x/y/z). This migrates MemTracePass's
      // __pim_load_persistent{,_yz,_xz,_xy} scope decision into the layout pass.
      // The simulator dedups at this scope. 0 when the lever is disabled.
      int64_t scope = dOperandScope ? 0 : (int64_t)((~pidMask) & 0x7u);
      fields.push_back(b.getNamedAttr("residency_scope",
                                      b.getI64IntegerAttr(scope)));
      if (pidMask == 0 && funcReduction && !dOperandScope)
        fields.push_back(b.getNamedAttr("resident", b.getUnitAttr()));
      // broadcast lever. Per-BG replication of a bank-broadcast operand.
      if (cls == "BroadcastReplicate") {
        fields.push_back(
            b.getNamedAttr("replicate_mode", b.getStringAttr("per_bg")));
        if (scalarSplat)
          fields.push_back(b.getNamedAttr("scalar_splat", b.getUnitAttr()));
      }
      // acc-resident lever. Keep the accumulated output psum in the PE across
      // the reduction (the output store of a reduction kernel).
      if (isStore && funcReduction && cls == "ParallelSpread" && !dAccResident)
        fields.push_back(
            b.getNamedAttr("accumulator_resident", b.getUnitAttr()));
      // bank-spread lever. Spread the parallel output dim across the banks.
      if (isTensorPtr && !dBankSpread) {
        int64_t spreadAxis[] = {0};
        fields.push_back(b.getNamedAttr("bank_spread_axes",
                                        b.getDenseI64ArrayAttr(spreadAxis)));
      }
      // Address footprint of one program instance, flattened five words per
      // axis as [extent, stride factor, arg, arg, arg] with -1 padding. A
      // stride the pass cannot fold to a number is left as a product of kernel
      // argument indices for the runtime to resolve, which is how conv's
      // CI*R*S weight pitch survives. Absent when the address is not affine in
      // the ranges and the reduction IV.
      SmallVector<FpAxis> fp;
      if (operandFootprint(ptr, redFor, fp) && !fp.empty()) {
        SmallVector<int64_t> flat;
        for (const FpAxis &ax : fp) {
          flat.push_back(ax.extent);
          flat.push_back(ax.stride.factor);
          for (int64_t i = 0; i < kMaxStrideArgs; ++i)
            flat.push_back(i < (int64_t)ax.stride.args.size()
                               ? ax.stride.args[i]
                               : -1);
        }
        fields.push_back(
            b.getNamedAttr("footprint", b.getDenseI64ArrayAttr(flat)));
      }

      op->setDiscardableAttr("im.residency", b.getDictionaryAttr(fields));
      ++numClassified;
    });

    // llvm::outs() << "\n\n\n I was here meshtag \n\n\n";

    mod->setAttr("im.residency-analyzed-mem-ops",
                 b.getI64IntegerAttr(numAnalyzed));
    mod->setAttr("im.residency-classified-mem-ops",
                 b.getI64IntegerAttr(numClassified));
    mod->setAttr("im.residency-reduction-loops",
                 b.getI64IntegerAttr(numReductionLoops));
  }
};

} // anonymous namespace

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

namespace mlir {
namespace triton {
namespace im {

std::unique_ptr<OperationPass<ModuleOp>> createIMOperandResidencyLayoutPass() {
  return std::make_unique<IMOperandResidencyLayoutPass>();
}

} // namespace im
} // namespace triton
} // namespace mlir
