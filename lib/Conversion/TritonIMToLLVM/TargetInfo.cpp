#include "TargetInfo.h"

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

  auto *ctx = rewriter.getContext();
  Type i32 = IntegerType::get(ctx, 32);

  auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(fnName);
  if (!funcOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(i32, {});
    funcOp = LLVM::LLVMFuncOp::create(rewriter, loc, fnName, fnType);
  }

  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

// ---------------------------------------------------------------------------
// Warp-level reduce
// ---------------------------------------------------------------------------

// A reduce rewrite-im-layout stamped im.pcu-pair sums the two banks of one PCU over lane
// bit 0. The PCU does it by accumulating both passes into one GRF_B entry, which the
// runtime models with an untraced carry: the even bank deposits each cell and gets 0, the
// odd bank receives it. Anything else has no hardware reduce, and refuseCrossBank has
// refused it before lowering, so the identity-shuffle fallback is never reached.
bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  if (reduceLaneIdMask != 1 || !op->hasAttr("im.pcu-pair"))
    return false;
  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  auto fn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__pim_pcu_carry");
  if (!fn) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    fn = LLVM::LLVMFuncOp::create(rewriter, loc, "__pim_pcu_carry",
                                  LLVM::LLVMFunctionType::get(i32, {i32}));
  }
  for (Value &v : acc) {
    Type ty = v.getType();
    unsigned bits = ty.getIntOrFloatBitWidth();
    if (bits > 32)
      return false;
    Type ity = IntegerType::get(ctx, bits);
    Value raw = isa<FloatType>(ty) ? LLVM::BitcastOp::create(rewriter, loc, ity, v).getResult()
                                   : v;
    if (bits < 32)
      raw = LLVM::ZExtOp::create(rewriter, loc, i32, raw);
    Value got = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{raw}).getResult();
    if (bits < 32)
      got = LLVM::TruncOp::create(rewriter, loc, ity, got);
    if (isa<FloatType>(ty))
      got = LLVM::BitcastOp::create(rewriter, loc, ty, got);
    v = isa<FloatType>(ty) ? LLVM::FAddOp::create(rewriter, loc, v, got).getResult()
                           : LLVM::AddOp::create(rewriter, loc, v, got).getResult();
  }
  return true;
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
