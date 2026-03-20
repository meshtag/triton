# Triton IM Backend Changes: Design and Implementation Notes

This document explains the Triton-side changes made to enable lowering Triton kernels
for a generic **In-Memory (IM)** backend and producing LLVM IR suitable for downstream
PIM tracing/simulation.

## Goals

- Add a new Triton conversion pass: `convert-triton-im-to-llvm`
- Reuse Triton's shared GPU->LLVM lowering infrastructure where possible
- Model IM execution as single-threaded, flat-memory (no warp/shared-memory primitives)
- Expose the pass through Python bindings and `triton-opt`
- Fix legalization blockers encountered during real kernel compilation (`make_range`, `load`, `store`)

---

## Files Added / Modified

### New IM conversion pass files

- `include/triton/Conversion/TritonIMToLLVM/CMakeLists.txt`
- `include/triton/Conversion/TritonIMToLLVM/Passes.h`
- `include/triton/Conversion/TritonIMToLLVM/Passes.td`
- `lib/Conversion/TritonIMToLLVM/CMakeLists.txt`
- `lib/Conversion/TritonIMToLLVM/TargetInfo.h`
- `lib/Conversion/TritonIMToLLVM/TargetInfo.cpp`
- `lib/Conversion/TritonIMToLLVM/TritonIMToLLVM.cpp`

### Existing files updated to wire IM pass

- `include/triton/Conversion/CMakeLists.txt`
- `lib/Conversion/CMakeLists.txt`
- `python/src/passes.cc`
- `bin/RegisterTritonDialects.h`

---

## Architecture

## 1) `TargetInfo` for IM (`TargetInfo.h/.cpp`)

`mlir::triton::im::TargetInfo` implements `TargetInfoBase` with IM semantics:

- **Single program id:** `programId(...) -> 0`
- **No barriers / sync:** barrier methods are no-ops
- **No shared memory:** `storeDShared` / `loadDShared` are unsupported (`llvm_unreachable`)
- **No warp data movement:** shuffle/permute act as identity
- **Flat memory model:** address spaces map to `0`
- **No warp-level reduction hardware:** returns `false` and falls back to generic lowering
- **Host-like runtime helpers:** `printf` and `abort` declarations emitted in LLVM dialect

This lets shared conversion patterns run while enforcing IM constraints.

## 2) Conversion pass (`TritonIMToLLVM.cpp`)

The pass is a 3-phase partial conversion:

1. **Function signature conversion**
2. **Bulk TritonGPU/Triton lowering to LLVM** (shared Triton populate* helpers + IM overrides)
3. **Residual control-flow lowering**

### Legal/illegal dialect policy

- Legal: `LLVM`, `cf`
- Illegal: `triton`, `triton.gpu`, `mlir::gpu`

Marking `mlir::gpu` illegal is crucial because shared utilities create `gpu.thread_id`.

---

## Critical bug fixes made during bring-up

## A) `tt.make_range` legalization failure

### Symptom

- Conversion failed: `failed to legalize operation 'tt.make_range'`

### Root cause

- Shared helper `emitIndices` creates `gpu::ThreadIdOp`
- IM pass originally had no conversion for `gpu::ThreadIdOp`

### Fix

- Added `IMThreadIdOpConversion` that rewrites `gpu.thread_id` to constant `0` (`index`)
- Marked `mlir::gpu::GPUDialect` illegal so conversions are explicit and complete

This unblocks `tt.make_range` lowering under single-thread IM assumptions.

## B) `tt.load` / `tt.store` legalization failure

### Symptom

- After fixing `make_range`, conversion failed on `tt.load` (and by extension `tt.store`)

### Root cause

- Core `populateMemoryOpToLLVMPatterns` does not include global `tt.load`/`tt.store`
- NVIDIA/AMD backends provide backend-specific load/store conversion patterns separately

### Fix

Added IM-specific patterns:

- `IMLoadOpConversion`
- `IMStoreOpConversion`

Behavior:

- Unpack per-element pointers/masks/values using shared helpers
- Emit scalar `LLVM::LoadOp` / `LLVM::StoreOp`
- Support masked operations via explicit conditional branches
- Repack tensor results for `tt.load`

This made AXPY-style kernels lower fully to LLVM IR in IM mode.

---

## Tooling exposure

## Python binding

In `python/src/passes.cc`, IM pass is exposed as:

- `passes.convert.add_convert_triton_im_to_llvm`

## `triton-opt` registration

In `bin/RegisterTritonDialects.h`:

- Include IM pass header
- Register pass in `registerTritonDialects(...)`

CLI pass name:

- `--convert-triton-im-to-llvm`

---

## Build / runtime notes

- In-tree backend discovery for IM requires:
  - `TRITON_BACKENDS_IN_TREE=1`
- The IM backend now uses its own target string (`"im:hbm-pim"`) instead of the
  previous `"cuda:80"` hack.  The shared `FuncOpConversion` still sets `nvvm.kernel`
  attributes, which are stripped in the Python `make_llir` step.
- HBM-PIM bank-level parallelism is modeled via `threads_per_warp = 16` (one
  "thread" per PIM bank).  The bank index is provided at runtime by the extern
  function `__pim_get_bank_id()`.

---

## Validation commands

From repo root (`/Users/meshtag/TritonPIM`):

```bash
source third_party/triton/.venv/bin/activate
TRITON_BACKENDS_IN_TREE=1 python test_im_debug.py
```

Expected: prints LLVM IR preview and `[OK] ... bytes of LLVM IR generated`.

Direct pass check (optional):

```bash
third_party/triton/build/cmake.macosx-12.1-arm64-cpython-3.13/bin/triton-opt \
  --convert-triton-im-to-llvm <input.mlir>
```

---

## Known follow-up work

- Replace/abstract CUDA-specific target metadata in IM frontend path
- Add direct Triton-kernel-LLVM-to-tracer bridge helper for one-command flow
- Expand IM lowering coverage for more Triton ops (atomics/special ops as needed)
