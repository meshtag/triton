"""
IM kernel launcher — compile and execute IM kernels on CPU.

Provides the multi-bank execution loop that emulates HBM-PIM's bank-level
SIMD parallelism.  On real hardware all banks execute in lockstep; here
banks are iterated sequentially.  The result is identical.

Two-level parallelism model
---------------------------
  Level 1 — inter-bank (SIMT-like):
      32 banks (one PE per bank) execute the same instruction stream on
      different data.  Modeled by ``threads_per_warp = num_banks`` in
      BlockedEncodingAttr.  Each bank is identified by ``__pim_get_bank_id()``.

  Level 2 — intra-bank (SIMD-like):
      Within each bank the PE processes ``sizePerThread`` elements.
      With ``BLOCK = 64`` and ``num_banks = 32`` → ``sizePerThread = 2``,
      so each bank handles 2 elements (unrolled scalar ops in the IR).

Execution loop::

    for pid_z in range(num_programs_z):   # optional 3D
        for pid_y in range(num_programs_y):   # optional 2D
            for pid in range(num_programs):       # tiles / blocks
                for bank in range(num_banks):     # parallel on real HW
                    set_program_id(pid)
                    set_program_id_y(pid_y)
                    set_program_id_z(pid_z)
                    set_bank_id(bank)
                    kernel(args...)

Usage
-----
From another script::

    from triton.backends.im.launcher import compile_im_kernel, launch_im_kernel

    lib = compile_im_kernel(llir, kernel_name="axpy_kernel")
    launch_im_kernel(lib, num_banks=32, num_programs=4,
                     arg_types=[P, P, c_int32, c_int32],
                     arg_values=[X_ptr, Y_ptr, A, N])
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# IR post-processing
# ---------------------------------------------------------------------------

def strip_for_cpu(llir: str) -> str:
    """Remove target / NVVM / addrspace artifacts so clang can compile for CPU.

    The IM backend emits ``ptr addrspace(1)`` for global pointers (inherited
    from the shared TritonGPU load/store lowering).  For native CPU execution
    we strip address-space qualifiers since the host flat address space is 0.
    """
    llir = re.sub(
        r"^target\s+triple\s*=\s*\"[^\"]*\"\s*$", "", llir, flags=re.MULTILINE
    )
    llir = re.sub(
        r"^target\s+datalayout\s*=\s*\"[^\"]*\"\s*$", "", llir, flags=re.MULTILINE
    )
    # Remove address-space qualifiers (addrspace(N) → flat)
    llir = re.sub(r"\s*addrspace\(\d+\)\s*", " ", llir)
    # Residual NVVM metadata
    llir = re.sub(
        r"^![a-zA-Z_]+\s*=\s*!\{.*nvvm.*\}\s*$", "", llir, flags=re.MULTILINE
    )
    # Strip LLVM 19+ attributes unknown to older opt (e.g. system LLVM 18)
    llir = re.sub(r"\bnocreateundeforpoison\b\s*", "", llir)
    return llir


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_IM_RUNTIME_C = str(
    Path(__file__).resolve().parents[6]          # triton repo root
    / ".." / ".." / "ramulator2"                 # ../../ramulator2
    / "llvm-tracer" / "runtime" / "im_runtime.c"
)


def compile_im_kernel(
    llir: str,
    kernel_name: str = "axpy_kernel",
    *,
    runtime_c: str | None = None,
    output_dir: str | None = None,
    extra_clang_flags: Sequence[str] = (),
) -> ctypes.CDLL:
    """Compile Triton-emitted LLVM IR + im_runtime.c into a shared library.

    Parameters
    ----------
    llir : str
        Raw LLVM IR from the IM backend (``ccinfo.asm["llir"]``).
    kernel_name : str
        Entry-point symbol (used only for error messages).
    runtime_c : str | None
        Path to ``im_runtime.c``.  Auto-detected from the repo layout when
        ``None`` (works from the TritonPIM workspace).
    output_dir : str | None
        Directory for the shared library.  A temp directory is used if ``None``.
    extra_clang_flags : sequence of str
        Additional flags passed to ``clang``.

    Returns
    -------
    ctypes.CDLL
        Loaded shared library with the kernel and runtime symbols.
    """
    if runtime_c is None:
        runtime_c = _IM_RUNTIME_C
    if not os.path.isfile(runtime_c):
        # Fall back: look relative to CWD (common when running from repo root)
        alt = os.path.join(
            "third_party", "ramulator2", "llvm-tracer", "runtime", "im_runtime.c"
        )
        if os.path.isfile(alt):
            runtime_c = alt
        else:
            raise FileNotFoundError(
                f"im_runtime.c not found at {runtime_c} or {alt}. "
                "Pass runtime_c= explicitly."
            )

    llir = strip_for_cpu(llir)

    if output_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="im_kernel_")
        output_dir = tmpdir

    ir_path = os.path.join(output_dir, "kernel.ll")
    with open(ir_path, "w") as f:
        f.write(llir)

    ext = "dylib" if platform.system() == "Darwin" else "so"
    lib_path = os.path.join(output_dir, f"libkernel.{ext}")

    cmd = [
        "clang", "-shared", "-O2", "-fPIC",
        "-Wno-override-module",
        *extra_clang_flags,
        ir_path, runtime_c,
        "-o", lib_path,
    ]
    subprocess.check_call(cmd)
    return ctypes.CDLL(lib_path)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def launch_im_kernel(
    lib: ctypes.CDLL,
    *,
    kernel_name: str = "axpy_kernel",
    num_banks: int = 32,
    num_programs: int = 1,
    num_programs_y: int = 1,
    num_programs_z: int = 1,
    arg_types: Sequence[Any] | None = None,
    arg_values: Sequence[Any] | None = None,
) -> None:
    """Execute an IM kernel across all ``(program_id, bank_id)`` pairs.

    Parameters
    ----------
    lib : ctypes.CDLL
        Shared library produced by :func:`compile_im_kernel`.
    kernel_name : str
        Symbol name of the kernel entry point.
    num_banks : int
        Number of PIM banks (= ``threads_per_warp``).
    num_programs : int
        Number of program invocations along axis 0 (X).
    num_programs_y : int
        Number of program invocations along axis 1 (Y).  Default 1 (1D).
    num_programs_z : int
        Number of program invocations along axis 2 (Z).  Default 1 (1D).
    arg_types : list
        ``ctypes`` types for each kernel argument.  The IM backend always
        appends two extra ``ptr`` arguments (``global_scratch`` and
        ``profile_scratch``) — pass ``ctypes.c_void_p`` / ``None`` for them.
    arg_values : list
        Values matching *arg_types*.
    """
    kernel_fn = getattr(lib, kernel_name)
    if arg_types is not None:
        kernel_fn.argtypes = list(arg_types)
    kernel_fn.restype = None

    set_bank = lib.__pim_set_bank_id
    set_bank.argtypes = [ctypes.c_int32]
    set_bank.restype = None

    set_pid = lib.__pim_set_program_id
    set_pid.argtypes = [ctypes.c_int32]
    set_pid.restype = None

    set_pid_y = lib.__pim_set_program_id_y
    set_pid_y.argtypes = [ctypes.c_int32]
    set_pid_y.restype = None

    set_pid_z = lib.__pim_set_program_id_z
    set_pid_z.argtypes = [ctypes.c_int32]
    set_pid_z.restype = None

    args = list(arg_values) if arg_values is not None else []

    for pid_z in range(num_programs_z):
        set_pid_z(pid_z)
        for pid_y in range(num_programs_y):
            set_pid_y(pid_y)
            for pid in range(num_programs):
                set_pid(pid)
                for bank in range(num_banks):       # parallel on real HBM-PIM hardware
                    set_bank(bank)
                    kernel_fn(*args)
