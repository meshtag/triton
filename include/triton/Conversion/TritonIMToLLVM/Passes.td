#ifndef TRITONIM_CONVERSION_PASSES
#define TRITONIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// Names. A lane is one Triton thread (threads_per_warp = num_banks) and, on the device, one
// of the paper's bank-PE pairs. Option and runtime names keep the word bank for a lane, as in
// num_banks and __pim_get_bank_id().

// Conversion pass: lowers TritonGPU IR → LLVM IR for In-Memory (IM)
// processing targets such as HBM-PIM.  No GPU-specific constructs
// (shared memory, warps, barriers) are emitted; the resulting LLVM IR
// is suitable for instrumentation (e.g. MemTracePass) and PIM
// simulation in Ramulator 2.
def ConvertTritonIMToLLVM
    : Pass<"convert-triton-im-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonGPU IR to LLVM for In-Memory processing";
  let description = [{
    Lowers Triton GPU dialect operations to the LLVM dialect targeting
    In-Memory processing architectures such as HBM-PIM.  The pass
    reuses the shared TritonGPUToLLVM lowering patterns (elementwise,
    memory, control-flow, SPMD, etc.) with an IM-specific TargetInfo
    that models bank-level SIMD parallelism: each PIM bank is one
    Triton "thread" (threads_per_warp = numBanks).  The bank index is
    obtained at runtime via the extern function __pim_get_bank_id(). Before
    converting, it states the tables the trace runtimes read as module globals
    (IMArtifact.cpp).
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect", "mlir::scf::SCFDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Places the banks on the axis the module states. Each PIM bank owns a
// consecutive chunk of elements (sizePerThread = shape / threadsPerWarp), so
// bank 0 holds {0, 1, 2, 3} rather than the GPU-coalesced {0, 16, 32, 48}.
//
// This pass replaces tritongpu-coalesce in the IM pipeline.
def RewriteIMLayout : Pass<"rewrite-im-layout", "mlir::ModuleOp"> {
  let summary = "Rewrite tensor layout for bank-consecutive PIM access";
  let description = [{
    For im: targets convert-triton-to-tritongpu already emits the bank-consecutive
    encoding with the lanes on the innermost axis. This pass rebuilds it with the
    banks on the axis the module states (im.bank_axis, im.bank_split, or one derived
    from a reduce or from im.derive_bank_axis) and refuses a choice that would put
    lanes on a reduced axis. ConvertLayoutOps are inserted around load/store
    operations, and the downstream tritongpu-remove-layout-conversions pass
    propagates the new encoding through the rest of the IR.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Stamps what the runtimes need to know about each operand, read back through
// emitPimLayoutTable.
def IMOperandResidencyLayout
    : Pass<"im-operand-residency-layout", "mlir::ModuleOp"> {
  let summary = "Stamp each memory access with its replication and footprint";
  let description = [{
    Stamps each tensor-of-pointer load or store, and each scalar load splatted
    across the lanes, with an im.residency dictionary. Its fields include the reuse
    class, how many banks receive a copy (bank_replicated), the kernel argument read
    and the footprint of one program instance. im.schedule can override
    bank_replicated. Under im.optipim_conv, im.optipim_roles and
    im.optipim_hold_input it also stamps OptiPIM's conventions on the module as
    im.opt_<x>, which the lowering states as globals.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

// Marks tile boundaries in a persistent kernel so the trace runtime can scope its
// per-dispatch collapse correctly.
def IMTileBoundary : Pass<"im-tile-boundary", "mlir::ModuleOp"> {
  let summary = "Call __pim_tile_boundary() at each tile-loop back edge";
  let description = [{
    Only runs when the module carries `im.persistent`. Inserts a call to the
    runtime's `__pim_tile_boundary()` at the top of the OUTERMOST `scf.for` body
    in every function.

    Why it is needed. The trace runtime folds the N bank replays of one all-bank
    command into one event, and scopes that fold to a dispatch. With one tile per
    program instance the host's program-id setter marks the dispatch. A persistent
    kernel puts the tile loop inside the kernel and the bank loop outside, so bank 0
    walks every tile before bank 1 starts; without a signal from inside the loop the
    whole run looks like one dispatch and every cross-tile re-read of the same
    address collapses to nothing. That reads as a large speedup and nothing warns.

    Why the outermost loop. In a persistent kernel the tile loop is outermost by
    construction and the reduction loop is nested inside it. Marking the specific
    loop in kernel source instead would put the decision back in the kernel.

    Ordering. Must run BEFORE scf-to-cf: afterwards the back edge is a branch and
    identifying the tile loop means recovering the structure just discarded.

    Under `im.tile_prologue_first` the first iteration skips the call, so work
    hoisted above the loop shares tile 0 with it.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
}

// Marks reductions that fold the lanes of one DRAM column, so the trace runtime can
// price a cost no memory access represents.
def IMLaneFold : Pass<"im-lane-fold", "mlir::ModuleOp"> {
  let summary = "Call __pim_trace_fold(steps, outputs) after each in-column reduce";
  let description = [{
    Only runs when the module carries `im.price_folds`. A reduce whose axis is the
    layout's contiguous axis within a lane adds values that shared a DRAM column,
    which the lane-parallel PIM MAC cannot do: it is a tree of log2(values) steps.
    The call after the reduce reports the steps and the outputs per lane. A reduce
    along any other axis is MAC accumulation across columns and is not marked.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect"];
}

// Marks loads a ReLU consumes, so the dcc-parity runtime issues them as PIM_RELU.
def IMReluOpcode : Pass<"im-relu-opcode", "mlir::ModuleOp"> {
  let summary = "Bracket each ReLU-only load with __pim_trace_op calls";
  let description = [{
    Only runs when the module carries `im.dcc_relu_opcode`. A load whose only use,
    through float casts, is a max against zero is bracketed by
    `__pim_trace_op(1)` and `__pim_trace_op(0)`, so the runtime issues its reads as
    DCC's PIM_RELU. This adopts DCC's pricing of ReLU, which their simulator does not
    space on the command bus. A stored ReLU of a value already in the register file
    has its store bracketed with a register-ReLU opcode, and the runtime issues
    HBM-PIM's GRF-to-GRF MOV with the ReLU bit before the write-back.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect"];
}

// Scalar replacement on Triton IR (IMScalarReplacement.cpp).
def IMRegisterPromotion : Pass<"im-register-promotion", "mlir::ModuleOp"> {
  let summary = "Keep a loop's loads and stores of one invariant address in a register";
  let description = [{
    A loop whose body loads a loop-invariant address before storing it, and touches
    nothing that may overlap that address otherwise, carries the value in an iter_arg
    instead: one load before the loop, one store after it. Different kernel arguments
    may overlap unless the module carries `im.noalias_args`.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::triton::TritonDialect"];
}

def IMStoreForwarding : Pass<"im-store-forwarding", "mlir::ModuleOp"> {
  let summary = "Give a load the value a store in the same block just wrote there";
  let description = [{
    A load of an address that a store earlier in the same block wrote, with no store
    or opaque op between that may overlap it, takes the stored value. Different kernel
    arguments may overlap unless the module carries `im.noalias_args`.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def IMDeadStores : Pass<"im-dead-stores", "mlir::ModuleOp"> {
  let summary = "Remove a store that a later store in the same block overwrites";
  let description = [{
    A store to an address that a later store in the same block writes again, with no
    load or opaque op between that may read it, is removed. Different kernel arguments
    may overlap unless the module carries `im.noalias_args`.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def IMStoreSinking : Pass<"im-store-sinking", "mlir::ModuleOp"> {
  let summary = "Move each store down its block so the block's stores issue together";
  let description = [{
    A store moves down its block, the last first, until an op that may read or write
    its address, another store or the terminator, so stores keep their order and a
    row of inputs stays open while they wait. Different kernel arguments may overlap
    unless the module carries `im.noalias_args`.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

// Groups a loop's loads by tensor, n unrolled iterations at a time.
def IMLoadCluster : Pass<"im-load-cluster", "mlir::ModuleOp"> {
  let summary = "Unroll and jam a loop, grouping each strip's loads by argument";
  let description = [{
    Only runs when the module carries `im.cluster_loads = n`, the register-file entries
    per lane. An innermost scf.for with a constant trip count whose body loads two or
    more kernel arguments (or one, when it also stores), stores none of them and touches
    memory no other way is unrolled in strips of G, as many iterations as fit n entries
    for any one argument, and the remainder of the trip count is a shorter strip of its
    own. Each strip's loads are grouped by argument, so a DRAM row stays open for G
    loads. Reads the lane layout. Stamps the widest strip of a storing loop as
    `im.map_strip_entries`.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::triton::TritonDialect"];
}

def IMDccTileMac : Pass<"im-dcc-tile-mac", "mlir::ModuleOp"> {
  let summary = "Select DCC's tile MAC for a contraction whose per-lane tile is one";
  let description = [{
    Only runs when the module carries `im.dcc_tile_mac`. A reduce over load * broadcast
    whose load keeps one column of outputs lane-fastest and whole tiles of the reduce axis
    next is stamped with the column reads one DCC MAC_OP1 covers. Every load of that
    argument must qualify. Adopts DCC's ISA pricing, a convention of the DCC path.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::triton::TritonDialect"];
}

def IMOptiPimMac : Pass<"im-optipim-mac", "mlir::ModuleOp"> {
  let summary = "Lower the contraction to MACs of the PE OptiPIM's machine declares";
  let description = [{
    Only runs when the module carries `im.optipim_mac`. Their PE takes one word of
    `im.dq_bits` per MAC, n values of 16 bits, and has one register and no accumulator, so a
    lane runs ceil(cells / n) MACs per contraction, its cells being the product's elements
    its operand masks leave on, lane terms included. Under `im.optipim_vertical`, their
    SIMDRAM, a word is `im.dq_bits` bitlines each holding one value 16 rows tall. A loop of
    `__pim_opt_mac(j)` calls goes where the contraction executes. The per-lane counts, the
    arranged slot of MAC j in step t and t as an affine map of the program ids and the tile
    index are stamped on the module as im.opt_<x>, which the lowering states as globals.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect"];
}

#endif // TRITONIM_CONVERSION_PASSES
