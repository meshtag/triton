"""
IR post-processing for running an IM kernel's LLIR on the host.

pipeline compiles the stripped module against a trace runtime. Banks are
lanes of ``threads_per_warp = num_banks`` in BlockedEncodingAttr, each lane
names itself through ``__pim_get_bank_id()``, and within a lane the
``sizePerThread`` elements become unrolled scalar ops.
"""

from __future__ import annotations

import re


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
