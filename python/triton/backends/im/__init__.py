import os

from triton.backends.compiler import GPUTarget
from triton.backends.im.launcher import compile_im_kernel, launch_im_kernel


def IMTarget(arch: str = "hbm-pim", num_banks: int = 16,
             debug: bool = False) -> GPUTarget:
    """Create a compile target for the In-Memory (IM) backend.

    Args:
        arch: Architecture identifier.  ``"hbm-pim"`` for Samsung HBM-PIM
              (the only supported architecture so far).
        num_banks: Number of PIM banks that execute in SIMD lock-step.
              For HBM-PIM this equals ``bank_groups × banks_per_group``
              from the Ramulator2 config (default 4 × 4 = 16).
        debug: When ``True`` (or when ``TRITON_IM_DEBUG=1`` is set),
              the compiler dumps IR after every pass-pipeline stage
              (TTIR, TTGIR, LLIR) to ``im_debug/`` in the current
              working directory and prints the pass pipeline for each
              stage to stderr.

    Returns:
        A ``GPUTarget("im", arch, num_banks)`` suitable for
        ``triton.compile()``.  Internally, ``GPUTarget.warp_size``
        carries the *num_banks* value; the IM compiler maps it to
        ``im.num-banks`` on the MLIR module and uses
        ``threads_per_warp = num_banks`` as TritonGPU plumbing so that
        the ``BlockedEncodingAttr`` distributes BLOCK elements across
        banks automatically.
    """
    if debug or os.environ.get("TRITON_IM_DEBUG", "") == "1":
        os.environ["TRITON_IM_DEBUG"] = "1"
    return GPUTarget("im", arch, num_banks)


__all__ = ["IMTarget", "compile_im_kernel", "launch_im_kernel"]
