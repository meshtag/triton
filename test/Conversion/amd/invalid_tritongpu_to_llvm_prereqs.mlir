// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950 -verify-diagnostics

// Missing TritonGPU layout encoding (convert-triton-to-tritongpu required).
// expected-error@+1 {{missing TritonGPU layout encoding on tensor type}}
module {
  tt.func @missing_encoding(%arg0: tensor<64xi32>) -> tensor<64xi32> {
    tt.return %arg0 : tensor<64xi32>
  }
}

// -----

// Missing global scratch allocation offsets (tritongpu-global-scratch-memory-allocation required).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx950"} {
  tt.func @missing_global_scratch_offset() -> !tt.ptr<i8> {
    // expected-error@+2 {{missing 'ttg.global_scratch_memory_offset' attribute; run tritongpu-global-scratch-memory-allocation first}}
    // expected-error@+1 {{failed to legalize operation 'ttg.global_scratch_alloc' that was explicitly marked illegal}}
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    tt.return %0 : !tt.ptr<i8>
  }
}
