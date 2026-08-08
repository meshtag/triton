/// IMOperandResidencyLayout.cpp. Decide per-operand physical layout and residency
/// for the in-memory (IM) backend. This is the explicit compiler-side analog of
/// OptiPIM's DataLayout/DetailLayout MILP, heuristic rather than a solver.
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

#include "triton/Conversion/TritonIMToLLVM/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
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

namespace {
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
// (used with a reduction loop's induction variable)?
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
      // replication models, independent of pid/reduction dependence. Covers
      // matvec x (pid-invariant) and matmul row-tiled A (pid-dependent). The
      // `broadcast` lever (when disabled) downgrades it to its plain class.
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

      SmallVector<NamedAttribute> fields;
      fields.push_back(b.getNamedAttr("reuse_class", b.getStringAttr(cls)));
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

      op->setDiscardableAttr("im.residency", b.getDictionaryAttr(fields));
      ++numClassified;
    });

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
