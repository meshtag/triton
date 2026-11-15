from triton.backends.compiler import GPUTarget


def IMTarget(arch: str = "hbm-pim", num_banks: int = 16) -> GPUTarget:
    """Create a compile target for the In-Memory (IM) backend.

    Args:
        arch: Architecture identifier.  ``"hbm-pim"`` is the only one. SIMDRAM
              kernels compile with it too, and the harness states that
              machine's column width as the ``im_dq_bits`` option.
        num_banks: Number of PIM lanes that execute in SIMD lock-step. The
              harnesses always pass it (pass_ablation defaults to 32).

    Set ``TRITON_IM_DEBUG=1`` to dump the IR after every pass-pipeline stage
    (TTIR, TTGIR, LLIR) to ``im_debug/`` in the current working directory.

    Returns:
        A ``GPUTarget("im", arch, num_banks)`` suitable for
        ``triton.compile()``.  ``GPUTarget.warp_size`` carries *num_banks*,
        which the IM compiler uses as ``threads_per_warp`` so that the
        ``BlockedEncodingAttr`` distributes BLOCK elements across banks.
    """
    return GPUTarget("im", arch, num_banks)


__all__ = ["IMTarget"]
