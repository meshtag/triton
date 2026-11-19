#include "TargetInfo.h"
#include "IMArtifact.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton::im;

// ---------------------------------------------------------------------------
// Feature queries
// ---------------------------------------------------------------------------

bool TargetInfo::supportMaximumMinimum() const { return false; }

bool TargetInfo::supportVectorizedAtomics() const { return false; }

// ---------------------------------------------------------------------------
// Cluster / CTA helpers
// ---------------------------------------------------------------------------

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // Single execution unit – always CTA 0.
  return LLVM::createConstantI32(loc, rewriter, 0);
}

// ---------------------------------------------------------------------------
// Barriers  (the lanes run in lockstep and share no memory, so none is needed)
// ---------------------------------------------------------------------------

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace /*targets*/) const {
  // No-op.
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter) const {
  // No-op.
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  // No-op.
}

// ---------------------------------------------------------------------------
// Shared / distributed memory  (not available on IM targets)
// ---------------------------------------------------------------------------

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  llvm_unreachable("IM targets do not have shared memory");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation * /*localLoadOp*/) const {
  llvm_unreachable("IM targets do not have shared memory");
}

// ---------------------------------------------------------------------------
// Cross-lane primitives
// ---------------------------------------------------------------------------

// Lanes are banks, and no lane can read another's registers. These return their input, and
// refuseCrossBank rejects every op that would need a real exchange.

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  return LLVM::ZExtOp::create(rewriter, loc, type, cmp);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int /*i*/) const {
  return val;
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int /*i*/) const {
  return val;
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int /*i*/) const {
  return val;
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value /*i*/) const {
  return val;
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value /*b*/, Value /*selector*/) const {
  return a;
}

// ---------------------------------------------------------------------------
// Program ID
// ---------------------------------------------------------------------------

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  // On HBM-PIM the "program id" identifies which tile of the input arrays
  // this invocation processes.  The host driver iterates over tiles and
  // sets the id via the IM runtime before each call.
  // Multi-axis support: X (axis 0), Y (axis 1), Z (axis 2).
  const char *fnName;
  switch (axis) {
  case ProgramIDDim::X:
    fnName = "__pim_get_program_id";
    break;
  case ProgramIDDim::Y:
    fnName = "__pim_get_program_id_y";
    break;
  case ProgramIDDim::Z:
    fnName = "__pim_get_program_id_z";
    break;
  }

  Type i32 = IntegerType::get(rewriter.getContext(), 32);
  auto funcOp = getOrDeclareRuntimeFn(rewriter, moduleOp, loc, fnName,
                                      LLVM::LLVMFunctionType::get(i32, {}));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

// ---------------------------------------------------------------------------
// Warp-level reduce
// ---------------------------------------------------------------------------

bool TargetInfo::warpReduce(RewriterBase & /*rewriter*/, Location /*loc*/,
                            SmallVector<Value> & /*acc*/,
                            triton::ReduceOp /*op*/,
                            unsigned /*reduceLaneIdMask*/) const {
  // Never reached. The shared lowering calls this only for a reduce across lanes, and
  // refuseCrossBank rejects those before lowering starts.
  return false;
}

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  if (resultElementTy.isInteger(32))
    return "__mulhi_i32";
  if (resultElementTy.isInteger(64))
    return "__mulhi_i64";
  llvm_unreachable("unsupported type for mulhi");
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> /*isSigned*/) const {
  auto loc = formatStrStart.getLoc();
  auto *ctx = rewriter.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  Type ptr = LLVM::LLVMPointerType::get(ctx);

  auto moduleOp = formatStrStart.getParentRegion()->getParentOfType<ModuleOp>();
  auto funcOp = getOrDeclareRuntimeFn(
      rewriter, moduleOp, loc, "printf",
      LLVM::LLVMFunctionType::get(i32, {ptr}, /*isVarArg=*/true));

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  operands.append(args.begin(), args.end());
  LLVM::CallOp::create(rewriter, loc, funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "empty message");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto moduleOp =
      rewriter.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  Type voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  auto abortFn = getOrDeclareRuntimeFn(rewriter, moduleOp, loc, "abort",
                                       LLVM::LLVMFunctionType::get(voidTy, {}));
  LLVM::CallOp::create(rewriter, loc, abortFn, ValueRange{});
}

// ---------------------------------------------------------------------------
// Address spaces
// ---------------------------------------------------------------------------

int TargetInfo::getSharedAddressSpace() const {
  // IM targets use a flat address space.  Return 0 (global) so that
  // any residual shared-memory references fall into global memory.
  return 0;
}

int TargetInfo::getAddressSpace(Attribute /*addressSpace*/) const {
  // Everything maps to address space 0 (flat / global).
  return 0;
}
