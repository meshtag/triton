module {
  tt.func public @bwd_kv_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: f32) attributes {noinline = false} {
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
    %cst_1 = arith.constant dense<0> : tensor<64x1xi64>
    %cst_2 = arith.constant dense<0> : tensor<64xi32>
    %cst_3 = arith.constant dense<0> : tensor<64xi64>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16>
    %cst_5 = arith.constant dense<0> : tensor<1x64xi64>
    %c0_i64 = arith.constant 0 : i64
    %cst_6 = arith.constant dense<0> : tensor<64x64xi32>
    %cst_7 = arith.constant dense<1> : tensor<1x64xi32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %cst_9 = arith.constant dense<1.44269502> : tensor<64x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c32_i64 = arith.constant 32 : i64
    %c32_i32 = arith.constant 32 : i32
    %cst_11 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %arg12, %arg13 : i32
    %4 = arith.divsi %1, %3 : i32
    %5 = tt.addptr %arg11, %2 : !tt.ptr<i32>, i32
    %6 = tt.load %5 : !tt.ptr<i32>
    %7 = tt.addptr %5, %c1_i32 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = arith.subi %8, %6 : i32
    %10 = arith.muli %0, %c64_i32 : i32
    %11 = arith.cmpi sge, %10, %9 : i32
    cf.cond_br %11, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %12 = tt.addptr %arg10, %2 : !tt.ptr<i32>, i32
    %13 = tt.load %12 : !tt.ptr<i32>
    %14 = tt.addptr %12, %c1_i32 : !tt.ptr<i32>, i32
    %15 = tt.load %14 : !tt.ptr<i32>
    %16 = arith.subi %15, %13 : i32
    %17 = arith.mulf %arg14, %cst_11 : f32
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = tt.splat %10 : i32 -> tensor<64xi32>
    %20 = arith.addi %19, %18 : tensor<64xi32>
    %21 = arith.extsi %13 : i32 to i64
    %22 = arith.extsi %6 : i32 to i64
    %23 = arith.extsi %arg12 : i32 to i64
    %24 = arith.muli %21, %23 : i64
    %25 = arith.muli %24, %c32_i64 : i64
    %26 = tt.addptr %arg0, %25 : !tt.ptr<bf16>, i64
    %27 = arith.muli %1, %c32_i32 : i32
    %28 = tt.addptr %26, %27 : !tt.ptr<bf16>, i32
    %29 = arith.muli %arg12, %c32_i32 : i32
    %30 = arith.extsi %16 : i32 to i64
    %31 = arith.extsi %29 : i32 to i64
    %32 = arith.extsi %arg13 : i32 to i64
    %33 = arith.muli %22, %32 : i64
    %34 = arith.muli %33, %c32_i64 : i64
    %35 = tt.addptr %arg1, %34 : !tt.ptr<bf16>, i64
    %36 = arith.muli %4, %c32_i32 : i32
    %37 = tt.addptr %35, %36 : !tt.ptr<bf16>, i32
    %38 = arith.muli %arg13, %c32_i32 : i32
    %39 = arith.extsi %9 : i32 to i64
    %40 = arith.extsi %38 : i32 to i64
    %41 = arith.extsi %10 : i32 to i64
    %42 = tt.addptr %arg2, %34 : !tt.ptr<bf16>, i64
    %43 = tt.addptr %42, %36 : !tt.ptr<bf16>, i32
    %44 = arith.muli %22, %23 : i64
    %45 = arith.muli %44, %c32_i64 : i64
    %46 = tt.addptr %arg3, %45 : !tt.ptr<bf16>, i64
    %47 = tt.addptr %46, %27 : !tt.ptr<bf16>, i32
    %48 = tt.addptr %arg4, %45 : !tt.ptr<bf16>, i64
    %49 = tt.addptr %48, %27 : !tt.ptr<bf16>, i32
    %50 = tt.addptr %arg5, %25 : !tt.ptr<bf16>, i64
    %51 = tt.addptr %50, %27 : !tt.ptr<bf16>, i32
    %52 = tt.addptr %arg6, %24 : !tt.ptr<f32>, i64
    %53 = tt.addptr %52, %1 : !tt.ptr<f32>, i32
    %54 = tt.addptr %arg7, %24 : !tt.ptr<f32>, i64
    %55 = tt.addptr %54, %1 : !tt.ptr<f32>, i32
    %56 = tt.addptr %arg8, %21 : !tt.ptr<i32>, i64
    %57 = tt.addptr %arg9, %22 : !tt.ptr<i32>, i64
    %58 = tt.splat %37 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    %59 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %60 = arith.extsi %59 : tensor<32xi32> to tensor<32xi64>
    %61 = tt.expand_dims %60 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
    %62 = tt.broadcast %61 : tensor<32x1xi64> -> tensor<32x64xi64>
    %63 = tt.splat %41 : i64 -> tensor<64xi64>
    %64 = arith.extsi %18 : tensor<64xi32> to tensor<64xi64>
    %65 = arith.addi %63, %64 : tensor<64xi64>
    %66 = tt.expand_dims %65 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
    %67 = tt.splat %40 : i64 -> tensor<1x64xi64>
    %68 = arith.muli %66, %67 : tensor<1x64xi64>
    %69 = tt.broadcast %68 : tensor<1x64xi64> -> tensor<32x64xi64>
    %70 = arith.addi %62, %69 : tensor<32x64xi64>
    %71 = tt.addptr %58, %70 : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi64>
    %72 = arith.cmpi sge, %66, %cst_5 : tensor<1x64xi64>
    %73 = tt.splat %39 : i64 -> tensor<1x64xi64>
    %74 = arith.cmpi slt, %66, %73 : tensor<1x64xi64>
    %75 = arith.andi %72, %74 : tensor<1x64xi1>
    %76 = tt.broadcast %75 : tensor<1x64xi1> -> tensor<32x64xi1>
    %77 = tt.load %71, %76, %cst_4 : tensor<32x64x!tt.ptr<bf16>>
    %78 = tt.splat %43 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    %79 = tt.addptr %78, %70 : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi64>
    %80 = tt.load %79, %76, %cst_4 : tensor<32x64x!tt.ptr<bf16>>
    %81 = tt.splat %57 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %82 = tt.addptr %81, %65 : tensor<64x!tt.ptr<i32>>, tensor<64xi64>
    %83 = arith.cmpi sge, %65, %cst_3 : tensor<64xi64>
    %84 = tt.splat %39 : i64 -> tensor<64xi64>
    %85 = arith.cmpi slt, %65, %84 : tensor<64xi64>
    %86 = arith.andi %83, %85 : tensor<64xi1>
    %87 = tt.load %82, %86, %cst_2 : tensor<64x!tt.ptr<i32>>
    %88:7 = scf.for %arg15 = %c0_i32 to %16 step %c64_i32 iter_args(%arg16 = %c0_i64, %arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64, %arg20 = %c0_i64, %arg21 = %cst_8, %arg22 = %cst_8) -> (i64, i64, i64, i64, i64, tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
      %109 = tt.splat %56 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
      %110 = tt.splat %arg20 : i64 -> tensor<64xi64>
      %111 = arith.addi %110, %64 : tensor<64xi64>
      %112 = tt.addptr %109, %111 : tensor<64x!tt.ptr<i32>>, tensor<64xi64>
      %113 = arith.cmpi sge, %111, %cst_3 : tensor<64xi64>
      %114 = tt.splat %30 : i64 -> tensor<64xi64>
      %115 = arith.cmpi slt, %111, %114 : tensor<64xi64>
      %116 = arith.andi %113, %115 : tensor<64xi1>
      %117 = tt.load %112, %116, %cst_2 : tensor<64x!tt.ptr<i32>>
      %118 = tt.expand_dims %117 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %119 = tt.expand_dims %20 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %120 = tt.broadcast %118 : tensor<64x1xi32> -> tensor<64x64xi32>
      %121 = tt.broadcast %119 : tensor<1x64xi32> -> tensor<64x64xi32>
      %122 = arith.subi %120, %121 : tensor<64x64xi32>
      %123 = arith.cmpi sgt, %122, %cst_6 : tensor<64x64xi32>
      %124 = tt.expand_dims %87 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %125 = arith.cmpi eq, %124, %cst_7 : tensor<1x64xi32>
      %126 = tt.broadcast %125 : tensor<1x64xi1> -> tensor<64x64xi1>
      %127 = arith.andi %123, %126 : tensor<64x64xi1>
      %128 = arith.extui %127 : tensor<64x64xi1> to tensor<64x64xi32>
      %129 = tt.reshape %128 allow_reorder : tensor<64x64xi32> -> tensor<4096xi32>
      %130 = "tt.reduce"(%129) <{axis = 0 : i32}> ({
      ^bb0(%arg23: i32, %arg24: i32):
        %138 = arith.addi %arg23, %arg24 : i32
        tt.reduce.return %138 : i32
      }) : (tensor<4096xi32>) -> i32
      %131 = arith.cmpi ne, %130, %c0_i32 : i32
      %132:2 = scf.if %131 -> (tensor<64x32xf32>, tensor<64x32xf32>) {
        %138 = tt.splat %28 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
        %139 = tt.splat %arg16 : i64 -> tensor<64xi64>
        %140 = arith.addi %139, %64 : tensor<64xi64>
        %141 = tt.expand_dims %140 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64>
        %142 = tt.splat %31 : i64 -> tensor<64x1xi64>
        %143 = arith.muli %141, %142 : tensor<64x1xi64>
        %144 = tt.broadcast %143 : tensor<64x1xi64> -> tensor<64x32xi64>
        %145 = tt.expand_dims %60 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
        %146 = tt.broadcast %145 : tensor<1x32xi64> -> tensor<64x32xi64>
        %147 = arith.addi %144, %146 : tensor<64x32xi64>
        %148 = tt.addptr %138, %147 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi64>
        %149 = arith.cmpi sge, %141, %cst_1 : tensor<64x1xi64>
        %150 = tt.splat %30 : i64 -> tensor<64x1xi64>
        %151 = arith.cmpi slt, %141, %150 : tensor<64x1xi64>
        %152 = arith.andi %149, %151 : tensor<64x1xi1>
        %153 = tt.broadcast %152 : tensor<64x1xi1> -> tensor<64x32xi1>
        %154 = tt.load %148, %153, %cst_0 : tensor<64x32x!tt.ptr<bf16>>
        %155 = tt.dot %154, %77, %cst_10 : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
        %156 = tt.splat %53 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %157 = tt.splat %arg18 : i64 -> tensor<64xi64>
        %158 = arith.addi %157, %64 : tensor<64xi64>
        %159 = tt.splat %23 : i64 -> tensor<64xi64>
        %160 = arith.muli %158, %159 : tensor<64xi64>
        %161 = tt.addptr %156, %160 : tensor<64x!tt.ptr<f32>>, tensor<64xi64>
        %162 = arith.cmpi sge, %158, %cst_3 : tensor<64xi64>
        %163 = arith.cmpi slt, %158, %114 : tensor<64xi64>
        %164 = arith.andi %162, %163 : tensor<64xi1>
        %165 = tt.load %161, %164, %cst : tensor<64x!tt.ptr<f32>>
        %166 = tt.splat %17 : f32 -> tensor<64x64xf32>
        %167 = arith.mulf %155, %166 : tensor<64x64xf32>
        %168 = tt.expand_dims %165 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %169 = arith.mulf %168, %cst_9 : tensor<64x1xf32>
        %170 = tt.broadcast %169 : tensor<64x1xf32> -> tensor<64x64xf32>
        %171 = arith.subf %167, %170 : tensor<64x64xf32>
        %172 = math.exp2 %171 : tensor<64x64xf32>
        %173 = arith.select %127, %172, %cst_10 : tensor<64x64xi1>, tensor<64x64xf32>
        %174 = tt.splat %51 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
        %175 = tt.splat %arg17 : i64 -> tensor<64xi64>
        %176 = arith.addi %175, %64 : tensor<64xi64>
        %177 = tt.expand_dims %176 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64>
        %178 = arith.muli %177, %142 : tensor<64x1xi64>
        %179 = tt.broadcast %178 : tensor<64x1xi64> -> tensor<64x32xi64>
        %180 = arith.addi %179, %146 : tensor<64x32xi64>
        %181 = tt.addptr %174, %180 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi64>
        %182 = arith.cmpi sge, %177, %cst_1 : tensor<64x1xi64>
        %183 = arith.cmpi slt, %177, %150 : tensor<64x1xi64>
        %184 = arith.andi %182, %183 : tensor<64x1xi1>
        %185 = tt.broadcast %184 : tensor<64x1xi1> -> tensor<64x32xi1>
        %186 = tt.load %181, %185, %cst_0 : tensor<64x32x!tt.ptr<bf16>>
        %187 = tt.trans %173 {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
        %188 = arith.truncf %187 : tensor<64x64xf32> to tensor<64x64xbf16>
        %189 = tt.dot %188, %186, %arg22 : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
        %190 = tt.splat %55 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
        %191 = tt.splat %arg19 : i64 -> tensor<64xi64>
        %192 = arith.addi %191, %64 : tensor<64xi64>
        %193 = arith.muli %192, %159 : tensor<64xi64>
        %194 = tt.addptr %190, %193 : tensor<64x!tt.ptr<f32>>, tensor<64xi64>
        %195 = arith.cmpi sge, %192, %cst_3 : tensor<64xi64>
        %196 = arith.cmpi slt, %192, %114 : tensor<64xi64>
        %197 = arith.andi %195, %196 : tensor<64xi1>
        %198 = tt.load %194, %197, %cst : tensor<64x!tt.ptr<f32>>
        %199 = tt.dot %186, %80, %cst_10 : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
        %200 = tt.expand_dims %198 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %201 = tt.broadcast %200 : tensor<64x1xf32> -> tensor<64x64xf32>
        %202 = arith.subf %199, %201 : tensor<64x64xf32>
        %203 = arith.mulf %173, %202 : tensor<64x64xf32>
        %204 = arith.select %127, %203, %cst_10 : tensor<64x64xi1>, tensor<64x64xf32>
        %205 = tt.trans %204 {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
        %206 = arith.truncf %205 : tensor<64x64xf32> to tensor<64x64xbf16>
        %207 = tt.dot %206, %154, %arg21 : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
        scf.yield %207, %189 : tensor<64x32xf32>, tensor<64x32xf32>
      } else {
        scf.yield %arg21, %arg22 : tensor<64x32xf32>, tensor<64x32xf32>
      }
      %133 = arith.addi %arg16, %c64_i64 : i64
      %134 = arith.addi %arg17, %c64_i64 : i64
      %135 = arith.addi %arg18, %c64_i64 : i64
      %136 = arith.addi %arg19, %c64_i64 : i64
      %137 = arith.addi %arg20, %c64_i64 : i64
      scf.yield %133, %134, %135, %136, %137, %132#0, %132#1 : i64, i64, i64, i64, i64, tensor<64x32xf32>, tensor<64x32xf32>
    }
    %89 = tt.splat %arg14 : f32 -> tensor<64x32xf32>
    %90 = arith.mulf %88#5, %89 : tensor<64x32xf32>
    %91 = arith.truncf %90 : tensor<64x32xf32> to tensor<64x32xbf16>
    %92 = tt.splat %47 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %93 = tt.expand_dims %65 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64>
    %94 = tt.splat %31 : i64 -> tensor<64x1xi64>
    %95 = arith.muli %93, %94 : tensor<64x1xi64>
    %96 = tt.broadcast %95 : tensor<64x1xi64> -> tensor<64x32xi64>
    %97 = tt.expand_dims %60 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
    %98 = tt.broadcast %97 : tensor<1x32xi64> -> tensor<64x32xi64>
    %99 = arith.addi %96, %98 : tensor<64x32xi64>
    %100 = tt.addptr %92, %99 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi64>
    %101 = arith.cmpi sge, %93, %cst_1 : tensor<64x1xi64>
    %102 = tt.splat %39 : i64 -> tensor<64x1xi64>
    %103 = arith.cmpi slt, %93, %102 : tensor<64x1xi64>
    %104 = arith.andi %101, %103 : tensor<64x1xi1>
    %105 = tt.broadcast %104 : tensor<64x1xi1> -> tensor<64x32xi1>
    tt.store %100, %91, %105 : tensor<64x32x!tt.ptr<bf16>>
    %106 = arith.truncf %88#6 : tensor<64x32xf32> to tensor<64x32xbf16>
    %107 = tt.splat %49 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %108 = tt.addptr %107, %99 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi64>
    tt.store %108, %106, %105 : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(convert-triton-to-tritongpu{enable-source-remat=false num-ctas=1 num-warps=4 target=cuda:100 threads-per-warp=32}, tritongpu-coalesce, tritongpu-F32DotTC, triton-nvidia-gpu-plan-cta, tritongpu-remove-layout-conversions, tritongpu-optimize-thread-locality, tritongpu-accelerate-matmul, tritongpu-remove-layout-conversions, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, triton-nvidia-optimize-descriptor-encoding, triton-loop-aware-cse, tritongpu-fuse-nested-loops, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-licm, tritongpu-optimize-accumulator-init, tritongpu-hoist-tmem-alloc, tritongpu-promote-lhs-to-tmem, tritongpu-assign-latencies{num-stages=1}, tritongpu-schedule-loops, tritongpu-automatic-warp-specialization{num-stages=1}, tritongpu-pipeline{dump-intermediate-steps=false num-stages=1}, tritongpu-combine-tensor-select-and-if, triton-nvidia-gpu-remove-tmem-tokens, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-loop-aware-cse, tritongpu-prefetch, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tritongpu-coalesce-async-copy, triton-nvidia-optimize-tmem-layouts, tritongpu-remove-layout-conversions, triton-nvidia-interleave-tmem, tritongpu-reduce-data-duplication, tritongpu-reorder-instructions, triton-loop-aware-cse, symbol-dce, triton-nvidia-tma-lowering, triton-nvidia-gpu-fence-insertion{compute-capability=100}, sccp, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
