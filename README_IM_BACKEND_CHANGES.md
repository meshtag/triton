# Triton IM Backend Changes: Design and Implementation Notes

This document explains the Triton-side changes made to enable lowering Triton kernels
for a generic **In-Memory (IM)** backend and producing LLVM IR suitable for downstream
PIM tracing/simulation and CPU-based functional verification.

## Goals

- Add a new Triton conversion pass: `convert-triton-im-to-llvm`
- Reuse Triton's shared GPU→LLVM lowering infrastructure where possible
- Model HBM-PIM two-level parallelism: bank-level SIMT + intra-bank SIMD
- Provide a CPU launcher that executes compiled kernels for functional verification
- Expose the pass through Python bindings and `triton-opt`

---

## Two-Level Parallelism Model

HBM-PIM has two levels of parallelism, analogous to GPU SIMT+SIMD:

| Level | GPU analogy | HBM-PIM mechanism | Triton mapping |
|-------|-------------|--------------------|----------------|
| **Level 1** — inter-bank (SIMT-like) | Threads in a warp | 16 banks execute the same instruction in lock-step on different data | `threads_per_warp = num_banks` in `BlockedEncodingAttr` |
| **Level 2** — intra-bank (SIMD-like) | Elements per thread | Each bank's PE processes multiple elements from its row buffer | `sizePerThread = BLOCK / num_banks` |

With `BLOCK=64` and `num_banks=16`:
- `sizePerThread = 64 / 16 = 4` → each bank handles 4 elements
- Elements are interleaved: bank 0 gets `{0, 16, 32, 48}`, bank 1 gets `{1, 17, 33, 49}`, etc.
- This interleaving maps naturally to HBM physical address interleaving across banks

The generated LLVM IR for one bank shows 4 unrolled load/compute/store sequences —
that is the Level 2 vectorization.

---

## Module Attributes

The IM compiler sets these attributes on the MLIR module:

| Attribute | Type | Description |
|-----------|------|-------------|
| `im.num-banks` | `i32` | Canonical IM bank count (read by `ConvertTritonIMToLLVM`) |
| `im.arch` | `string` | Architecture identifier (e.g. `"hbm-pim"`) |
| `ttg.threads-per-warp` | `i32` | Set to `num_banks` — internal plumbing for `BlockedEncodingAttr` |
| `ttg.num-warps` | `i32` | Always 1 (no warp hierarchy on PIM) |

`ConvertTritonIMToLLVM` reads `im.num-banks` (not `ttg.threads-per-warp`).

---

## Runtime Functions

The compiled kernel calls two extern functions:

| Function | Returns | Set by |
|----------|---------|--------|
| `__pim_get_bank_id()` | `i32` bank index | `__pim_set_bank_id()` (host driver) |
| `__pim_get_program_id()` | `i32` tile index | `__pim_set_program_id()` (host driver) |

These are defined in `im_runtime.c` (in `ramulator2/llvm-tracer/runtime/`).
On real HBM-PIM hardware, these would map to hardware registers.

---

## CPU Launcher

The `triton.backends.im.launcher` module provides:

- `compile_im_kernel(llir)` — post-processes IR (strips addrspace, target triple) +
  compiles with `clang -shared` alongside `im_runtime.c` → returns `ctypes.CDLL`
- `launch_im_kernel(lib, num_banks, num_programs, ...)` — iterates over
  `(program_id, bank_id)` setting the runtime state before each kernel call

This is the **main execution path** for CPU-based verification and trace generation.
On real hardware the bank loop is hardware-parallel; the CPU emulation is sequential
but produces identical results.

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

- **Program ID:** `programId()` calls extern `__pim_get_program_id()` (host driver sets it per tile)
- **Bank ID:** `IMThreadIdOpConversion` rewrites `gpu.thread_id` → call `__pim_get_bank_id()`
- **No barriers / sync:** barrier methods are no-ops (banks execute in lock-step)
- **No shared memory:** `storeDShared` / `loadDShared` are unsupported (`llvm_unreachable`)
- **No warp data movement:** shuffle/permute act as identity
- **Flat memory model:** address spaces map to `0`
- **No warp-level reduction hardware:** returns `false` and falls back to generic lowering

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

- Added `IMThreadIdOpConversion` that rewrites `gpu.thread_id` to a call to
  `__pim_get_bank_id()` — returns the current bank index as `i32`
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
- The IM backend uses its own target string (`"im:hbm-pim"`) and target factory:
  ```python
  from triton.backends.im import IMTarget
  target = IMTarget("hbm-pim", 16)   # arch, num_banks
  ```
- The shared `FuncOpConversion` still sets `nvvm.kernel` attributes, which are
  stripped in the Python `make_llir` step (regex removal of attribute groups and refs).
- HBM-PIM bank-level parallelism is modeled via `threads_per_warp = num_banks`
  (one "thread" per PIM bank).  The bank index is provided at runtime by the extern
  function `__pim_get_bank_id()`.
- With `BLOCK > num_banks`, each bank processes `sizePerThread = BLOCK / num_banks`
  elements — this is the intra-bank SIMD-like vectorization.

---

## Validation commands

### Quick compilation check

```bash
cd /Users/meshtag/TritonPIM
source third_party/triton/.venv/bin/activate
TRITON_BACKENDS_IN_TREE=1 python test_im_debug.py
```

Expected: prints LLVM IR preview with `__pim_get_bank_id()` and
`__pim_get_program_id()` calls, ends with `[OK] ... bytes of LLVM IR generated`.

### Functional verification (CPU)

```bash
cd /Users/meshtag/TritonPIM
source third_party/triton/.venv/bin/activate
TRITON_BACKENDS_IN_TREE=1 python scripts/verify_im_cpu.py
```

Expected: compiles AXPY (BLOCK=64, 16 banks, sizePerThread=4), executes across
4 programs × 16 banks, compares against NumPy reference → `[PASS]`.

### Inspect intermediate IR stages

```bash
cd /Users/meshtag/TritonPIM
source third_party/triton/.venv/bin/activate
TRITON_BACKENDS_IN_TREE=1 python examples/axpy_im.py --out artifacts/axpy.ll
cat artifacts/axpy.ll
```

### Direct pass check (optional)

```bash
third_party/triton/build/cmake.macosx-12.1-arm64-cpython-3.13/bin/triton-opt \
  --convert-triton-im-to-llvm <input.mlir>
```

---

## Known follow-up work

- Replace/abstract CUDA-specific target metadata in IM frontend path
  (FuncOpConversion still injects `nvvm.kernel` — need IM-specific version)
- Add direct Triton-kernel-LLVM-to-tracer bridge for one-command trace generation
- Expand IM lowering coverage for more Triton ops (atomics, special ops)
- Row-buffer-aware data placement: ensure sizePerThread elements per bank
  map to contiguous physical row-buffer addresses (currently interleaved)
