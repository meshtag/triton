from dataclasses import dataclass, fields
from types import ModuleType
from typing import Any, Dict
import os
import re

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia


# ---------------------------------------------------------------------------
# TRITON_IM_DEBUG=1 writes each stage's IR to im_debug/ in the working directory.
# ---------------------------------------------------------------------------

def _im_debug() -> bool:
    return os.environ.get("TRITON_IM_DEBUG", "") == "1"


def _dump_stage(mod_or_str, stage: str, filename: str) -> str:
    """Write an IR snapshot to im_debug/<filename> and return its path."""
    outdir = os.path.join(os.getcwd(), "im_debug")
    os.makedirs(outdir, exist_ok=True)
    ir_text = mod_or_str if isinstance(mod_or_str, str) else str(mod_or_str)
    path = os.path.join(outdir, filename)
    with open(path, "w") as f:
        f.write(ir_text)
    return path


def _ints(text) -> list[int]:
    """The integers of a comma-separated option such as "4,8", empty items skipped."""
    return [int(v) for v in str(text).split(",") if v != ""]


def _memo(text) -> list[int]:
    """im_optipim_memo's "bound:weight,..." pairs as one list, empty when an item is not an
    integer."""
    try:
        return [int(v) for v in re.split("[,:]", str(text))]
    except ValueError:
        return []


# An option that needs another, with what it does with it.
_NEEDS = {
    "im_dcc_mac_addressing": ("im_dcc_tile_mac", "addresses DCC's tile MAC"),
    "im_optipim_roles": ("im_optipim_conv", "states residency for OptiPIM's contraction"),
    "im_optipim_mac": ("im_optipim_roles", "lowers the contraction im_optipim_roles states"),
    "im_optipim_hold_input": ("im_optipim_roles", "holds the input im_optipim_roles names"),
    "im_optipim_vertical": ("im_optipim_mac", "lays out im_optipim_mac's words"),
    "im_optipim_memo": ("im_optipim_hold_input",
                        "hashes the input im_optipim_hold_input holds"),
    "im_bank_active": ("im_bank_split", "names the active lanes of im_bank_split"),
}

# The numbers the runtimes read as __pim_placement_align.
_ALIGNS = {"dq": 1, "row-pack": 3}

# Options stamped on the module as `1 : i64` when set, each im_x as im.x.
_FLAGS = ("im_persistent", "im_tile_prologue_first", "im_price_folds", "im_derive_bank_axis",
          "im_dcc_relu_opcode", "im_dcc_tile_mac", "im_dcc_acc_grf", "im_dcc_mac_addressing",
          "im_dcc_return_from_input", "im_optipim_conv", "im_optipim_roles", "im_optipim_mac",
          "im_optipim_hold_input", "im_optipim_vertical")


@dataclass(frozen=True)
class IMOptions:
    num_banks: int = 16   # lanes in lockstep, overridden by the target's warp_size
    num_warps: int = 1    # always 1 for IM (no warp hierarchy)
    num_ctas: int = 1
    num_stages: int = 2
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    sanitize_overflow: bool = True
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple[str, ...] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    debug: bool = False
    backend_name: str = "im"
    instrumentation_mode: str = ""
    # HBM-PIM and SIMDRAM kernels both compile for this arch, and differ in im_dq_bits.
    arch: str = "hbm-pim"
    # Only "triton-licm" can be skipped, so __post_init__ refuses any other name.
    skip_passes: tuple[str, ...] = ()
    # User schedule for im-operand-residency-layout, as MLIR dictionary text
    # keyed by kernel operand index, e.g. '{arg1 = {bank_replicated = true}}'.
    # Empty means the classifier decides everything.
    im_schedule: str = ""
    # Which tensor axis carries the banks (rewrite-im-layout). -1 keeps the
    # upstream choice. Bounds operand amortisation: emitted traffic is
    # MACs*banks/BLOCK[bank_axis], so a shape whose bank axis is small can never
    # amortise however the tile is picked.
    im_bank_axis: int = -1
    # Persistent kernel: the grid stops being the work decomposition and one program
    # instance loops over several tiles. Marks the module so im-tile-boundary knows
    # which loop to signal; without it the trace runtime sees the whole replay as one
    # dispatch and cross-tile re-reads collapse to nothing. Also sets im.noalias_args, which
    # the optipim-parity runtime does not check, so its caller must pass distinct buffers.
    im_persistent: bool = False
    # Work hoisted above the tile loop belongs to the first tile (im-tile-boundary), so an
    # operand hoisted out of the loop loads after that tile's accumulator reset, as DCC
    # orders a round, and OptiPIM's parity kernel issues its input on step 0 only, not in
    # a step of its own. Off unless a harness asks, since the other paths keep the prologue
    # apart.
    im_tile_prologue_first: bool = False
    # Where each tensor starts. "dq" packs tensors into shared rows on different banks,
    # and "row-pack" gives each lane its exact share of one column space. The dcc-parity
    # runtime takes only "row-pack", and the default and optipim-parity runtimes only "dq".
    im_placement_align: str = "dq"
    # Bits one column command moves, which is what a vector access is sized against.
    # Differs by target: HBM-PIM moves prefetch*dq = 256, SIMDRAM's column is a
    # bit-serial slot of dq = 128. 0 = unstated, and the lowering keeps its default.
    im_dq_bits: int = 0
    # Lanes per tensor axis, e.g. "4,8" = 4 banks on axis 0 and 8 on axis 1. Empty
    # keeps the all-lanes-on-one-axis placement. The product must be num_banks. A
    # split over an axis the kernel reduces is refused by rewrite-im-layout: that is
    # a cross-bank reduction, and the partials have to leave the kernel instead.
    im_bank_split: str = ""
    # Price in-column reduction folds (im-lane-fold). Only the dcc-parity runtime defines
    # the call it inserts, so it is off unless a DCC harness asks for it.
    im_price_folds: bool = False
    # With no im_bank_split or im_bank_axis, let rewrite-im-layout put the banks on the
    # output's outermost axis instead of keeping the upstream layout. Off unless a harness
    # asks, since the other paths keep the upstream layout.
    im_derive_bank_axis: bool = False
    # Loads a ReLU consumes issue as DCC's PIM_RELU (im-relu-opcode). Adopts DCC's pricing,
    # which their simulator does not space on the command bus. Only the dcc-parity runtime
    # defines the call, so it is off unless a DCC harness asks for it.
    im_dcc_relu_opcode: bool = False
    # Unroll and jam a loop, its loads grouped by tensor (im-load-cluster), in strips that
    # fit this many register-file entries per lane. 0 leaves loop bodies as written.
    im_cluster_loads: int = 0
    # DCC's tile MAC (im-dcc-tile-mac): a contraction whose per-lane tile is one of theirs
    # issues one command per tile, their ISA's pricing, a convention of the DCC path.
    im_dcc_tile_mac: bool = False
    # DCC's accumulator convention per stored tensor and its GRF_A operand, derived from
    # the residency stamps and written into the artifact, for the dcc-parity runtime.
    im_dcc_acc_grf: bool = False
    # With the tile MAC, state the matrix geometry DCC's generator addresses each tile by,
    # so the runtime can issue a MAC at their address as a declared convention.
    im_dcc_mac_addressing: bool = False
    # A lane-folded partial's return stage is sized from its one input, as DCC's RED does,
    # for the dcc-parity runtime's RET bit.
    im_dcc_return_from_input: bool = False
    # OptiPIM's codegen_new conventions for the kernel's one contraction: the operand roles
    # their tensor index fixes, written into the artifact as __pim_opt_role, which
    # check_roles verifies and the optipim-parity runtime's counters read back.
    im_optipim_conv: bool = False
    # Residency by place in that contraction (im-operand-residency-layout): the first
    # factor is delivered to the PE, the second is resident in the bank and the stored sum
    # is drained by read-out. The optipim-parity runtime numbers a replicated resident
    # tensor's slots per lane.
    im_optipim_roles: bool = False
    # The contraction lowered to MACs of the PE their config declares, one im_dq_bits word
    # each, a loop of calls per lane where it executes (im-optipim-mac). Only the
    # optipim-parity runtime defines the call. Needs im_optipim_roles.
    im_optipim_mac: bool = False
    # OptiPIM's unbounded input memo, THEIR bug, reproduced and disclosed: the input held
    # across the program ids it ignores (conv), or wherever im_optipim_memo's hash repeats,
    # though their PE declares reg_per_pe 1. Stated by im-operand-residency-layout, needs
    # im_optipim_roles.
    im_optipim_hold_input: bool = False
    # Their SIMDRAM's vertical layout for im_optipim_mac: a word is im_dq_bits bitlines, one value
    # each, pe_bits rows tall, not im_dq_bits / 16 values side by side.
    im_optipim_vertical: bool = False
    # Their memo's hash of a step, as their simulator reads the MILP's coefficients: "b:w,..."
    # per temporal loop, slowest first. THEIR bug, reproduced and disclosed: on a window axis
    # it reads the temporal bound as the stride. Needs im_optipim_hold_input.
    im_optipim_memo: str = ""
    # Active lanes per tensor axis where their mapping leaves lanes of im_bank_split idle, as
    # "1,3,4,3,1"; the runtime gives the active lanes banks 0..S-1, their codegen's PU order.
    im_bank_active: str = ""

    def __post_init__(self):
        bad = set(self.skip_passes) - {"triton-licm"}
        if bad:
            raise ValueError(f"skip_passes={sorted(bad)!r}: only 'triton-licm' can be skipped")
        for option, (need, does) in _NEEDS.items():
            if getattr(self, option) and not getattr(self, need):
                raise ValueError(f"{option} {does}, set {need} too")
        if self.im_optipim_hold_input and self.im_persistent and not self.im_optipim_memo:
            raise ValueError("im_optipim_hold_input on a persistent kernel such as the gemm and "
                             "matvec parity kernels needs im_optipim_memo: the hoist out of the "
                             "tile loop already holds the input, and only their hash holds more")
        if self.im_optipim_memo:
            memo = _memo(self.im_optipim_memo)
            if (not memo or len(memo) % 2 or min(memo[0::2]) < 1 or min(memo[1::2]) < 0
                    or max(memo) >= 2 ** 31):
                raise ValueError(f"im_optipim_memo={self.im_optipim_memo!r}; want "
                                 "bound:weight pairs of 32-bit integers, bounds positive")
        if self.im_bank_active:
            act, split = _ints(self.im_bank_active), _ints(self.im_bank_split)
            if len(act) != len(split) or any(not 1 <= a <= s for a, s in zip(act, split)):
                raise ValueError(f"im_bank_active={self.im_bank_active!r} must give "
                                 f"1..lanes per axis of im_bank_split={self.im_bank_split!r}")
        if self.im_placement_align not in _ALIGNS:
            raise ValueError(
                f"im_placement_align={self.im_placement_align!r}; "
                f"expected one of {sorted(_ALIGNS)}")
        if self.im_dq_bits and int(self.im_dq_bits) < 1:
            raise ValueError(f"im_dq_bits={self.im_dq_bits!r}; must be positive")
        if self.im_bank_split:
            parts = _ints(self.im_bank_split)
            if not parts or any(v < 1 for v in parts):
                raise ValueError(
                    f"im_bank_split={self.im_bank_split!r}; want comma-separated "
                    f"positive lane counts, e.g. \"4,8\"")

    def hash(self):
        # Every field keys the cache, so no option can be left out of it.
        return ",".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))


def _stamp_module(mod, options: IMOptions) -> None:
    """State the im_* options on the module as the IM passes read them."""
    builder = ir.builder(mod.context)

    def stamp(name: str, text: str) -> None:
        mod.set_attr(name, builder.parse_attr(text))

    def array(values: list[int]) -> str:
        return "array<i64: " + ", ".join(map(str, values)) + ">"

    # The residency pass refuses a malformed or unmatched schedule, so a typo fails the
    # compile instead of quietly measuring the classifier's default.
    if options.im_schedule:
        stamp("im.schedule", options.im_schedule)
    for option in _FLAGS:
        if getattr(options, option):
            stamp("im." + option[len("im_"):], "1 : i64")
    if options.im_cluster_loads:
        stamp("im.cluster_loads", f"{int(options.im_cluster_loads)} : i64")
    if options.im_optipim_memo:
        stamp("im.optipim_memo", array(_memo(options.im_optipim_memo)))
    if options.im_bank_active:
        stamp("im.bank_active", array(_ints(options.im_bank_active)))
    # Interleaved, the only placement anything compiles for, is still stated so that a runtime
    # can refuse a kernel placed for another.
    stamp("im.layout_scheme", "2 : i64")
    stamp("im.placement_align", f"{_ALIGNS[options.im_placement_align]} : i64")
    if options.im_dq_bits:
        stamp("im.dq_bits", f"{int(options.im_dq_bits)} : i64")
    if options.im_bank_split:
        stamp("im.bank_split", array(_ints(options.im_bank_split)))
    if options.im_bank_axis >= 0:
        stamp("im.bank_axis", f"{options.im_bank_axis} : i64")


class IMBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "im"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "llir"

    def hash(self) -> str:
        return f"im-{self.target.arch}"

    def parse_options(self, opts) -> Any:
        args = {
            "arch": str(self.target.arch),
            # GPUTarget.warp_size carries num_banks (the IM-meaningful field).
            "num_banks": self.target.warp_size,
            "num_warps": 1,
            "num_ctas": int(opts.get("num_ctas", 1)),
            "num_stages": int(opts.get("num_stages", 2)),
        }
        for key in IMOptions.__dataclass_fields__.keys():
            if key in opts and opts[key] is not None:
                args[key] = opts[key]
        return IMOptions(**args)

    def pack_metadata(self, metadata):
        return (
            getattr(metadata, "num_warps", 1),
            getattr(metadata, "num_ctas", 1),
            getattr(metadata, "shared", 0),
        )

    def get_codegen_implementation(self, options):
        return {}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # The shared FuncOpConversion (TritonGPUToLLVM) references NVVM dialect
        # types (nvvm.kernel attr).  Loading NVIDIA dialects is required until
        # IM has its own FuncOpConversion that avoids NVVM attributes.
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options):
        skip = set(options.skip_passes)

        if options.im_persistent or options.im_cluster_loads:
            # Set before this stage's passes, since triton-licm runs here and is the only
            # pass in the pipeline that hoists loads.
            #
            # Licenses triton-licm, and im-load-cluster's reordering, to move a load
            # past a write that lands on a different kernel pointer argument. The
            # dcc-parity runtime's pim_register_tensor refuses overlapping tensors, so
            # there two pointer arguments are distinct buffers. The default runtime does
            # not check, and a caller passing aliased buffers gets wrong results. Named
            # separately from im.persistent so the assumption is visible in the IR.
            mod.set_attr("im.noalias_args",
                         ir.builder(mod.context).parse_attr("1 : i64"))

        if _im_debug():
            _dump_stage(mod, "ttir_input", "ttir_input.mlir")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        # triton-combine is left out of this stage. Its CombineBroadcastMulReducePattern
        # folds reduce_add(broadcast(a)*broadcast(b)) into tt.dot, which IM cannot lower.
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        if "triton-licm" not in skip:
            passes.ttir.add_triton_licm(pm)
            passes.ttir.add_loop_aware_cse(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")

        _stamp_module(mod, options)

        if _im_debug():
            _dump_stage(mod, "ttir", "ttir.mlir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"im:{options.arch}",
            options.num_warps,     # always 1
            options.num_banks,     # threads_per_warp ← num_banks
            options.num_ctas,
        )
        passes.convert.add_rewrite_im_layout(pm)
        # It places against the lane layout, so it follows rewrite-im-layout.
        passes.convert.add_im_load_cluster(pm)
        passes.convert.add_im_operand_residency_layout(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttgir")

        if _im_debug():
            _dump_stage(mod, "ttgir", "ttgir.mlir")
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        metadata["name"] = mod.get_entry_func_name()
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_licm(pm)
        # Before scf-to-cf, since afterwards the tile loop's back edge is a branch and
        # finding it means recovering the structure that conversion discards. No-op
        # unless the module is marked persistent.
        passes.convert.add_im_tile_boundary(pm)
        passes.convert.add_im_lane_fold(pm)
        passes.convert.add_im_dcc_tile_mac(pm)
        # After im-tile-boundary, which marks the tile loop that the step index reads.
        passes.convert.add_im_optipim_mac(pm)
        passes.convert.add_im_relu_opcode(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_convert_triton_im_to_llvm(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_llir")

        if _im_debug():
            _dump_stage(mod, "llvm_dialect", "llvm_dialect.mlir")

        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        metadata["shared"] = 0
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1
        llir = str(llvm_mod)
        # Strip NVIDIA artifacts injected by the shared FuncOpConversion.
        llir = re.sub(r"\bptx_kernel\s+void\b", "void", llir)
        llir = re.sub(
            r'^attributes\s*#\d+\s*=\s*\{[^}]*nvvm[^}]*\}\s*$', "",
            llir, flags=re.MULTILINE,
        )
        llir = re.sub(r'\)\s*#\d+\s*\{', ") {", llir)

        if _im_debug():
            _dump_stage(llir, "llir", "llir.ll")
        return llir

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
