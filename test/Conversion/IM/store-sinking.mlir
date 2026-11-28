// RUN: triton-opt --split-input-file %s -im-store-sinking | FileCheck %s

// The store to %x moves down past the load of %a, which with the licence reads another
// buffer, and stops at the next store, so the two stores issue together and in order.
module attributes {im.noalias_args = 1 : i64} {
  // CHECK-LABEL: @sink
  tt.func @sink(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %y: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK: tt.load %arg1
    // CHECK: arith.addi
    // CHECK: tt.store %arg0, %arg3
    // CHECK: tt.store %arg2
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    %w = tt.load %a : tensor<8x!tt.ptr<i32>>
    %s = arith.addi %w, %w : tensor<8xi32>
    tt.store %y, %s : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// Without the licence the load of %a may read %x, so the store stays before it.
module {
  // CHECK-LABEL: @no_licence
  tt.func @no_licence(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %y: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK: tt.store %arg0, %arg3
    // CHECK: tt.load %arg1
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    %w = tt.load %a : tensor<8x!tt.ptr<i32>>
    %s = arith.addi %w, %w : tensor<8xi32>
    tt.store %y, %s : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
