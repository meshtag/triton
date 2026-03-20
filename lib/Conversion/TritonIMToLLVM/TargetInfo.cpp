#include "TargetInfo.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

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
// Warp-level primitives  (no warp hardware on IM targets)
// ---------------------------------------------------------------------------

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  // Single-lane "ballot" – return cmp zero-extended to the requested type.
  return LLVM::ZExtOp::create(rewriter, loc, type, cmp);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace /*targets*/) const {
  // No-op: single-threaded execution needs no barriers.
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
// Shuffle primitives  (not available on IM targets)
// ---------------------------------------------------------------------------

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int /*i*/) const {
  // Single lane – shuffle is identity.
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
  // Single lane – permute is identity on a.
  return a;
}

// ---------------------------------------------------------------------------
// Program ID
// ---------------------------------------------------------------------------

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp /*moduleOp*/,
                            ProgramIDDim /*axis*/) const {
  // Single PIM unit – program id is always 0.
  return LLVM::createConstantI32(loc, rewriter, 0);
}

// ---------------------------------------------------------------------------
// Warp-level reduce
// ---------------------------------------------------------------------------

bool TargetInfo::warpReduce(RewriterBase & /*rewriter*/, Location /*loc*/,
                            SmallVector<Value> & /*acc*/,
                            triton::ReduceOp /*op*/,
                            unsigned /*reduceLaneIdMask*/) const {
  // No hardware reduce; fall back to the generic tree reduction.
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

  // Build declaration of printf if it does not exist.
  auto moduleOp = formatStrStart.getParentRegion()->getParentOfType<ModuleOp>();
  auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("printf");
  if (!funcOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(i32, {ptr}, /*isVarArg=*/true);
    funcOp = LLVM::LLVMFuncOp::create(rewriter, loc, "printf", fnType);
  }

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
  auto *ctx = rewriter.getContext();
  Type voidTy = LLVM::LLVMVoidType::get(ctx);

  auto abortFn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("abort");
  if (!abortFn) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(voidTy, {});
    abortFn = LLVM::LLVMFuncOp::create(rewriter, loc, "abort", fnType);
  }
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
