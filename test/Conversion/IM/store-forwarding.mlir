// RUN: triton-opt --split-input-file %s -im-store-forwarding | FileCheck %s

// The load of %x takes the value just stored there. With the licence the store to %y in
// between cannot change %x.
module attributes {im.noalias_args = 1 : i64} {
  // CHECK-LABEL: @forward
  tt.func @forward(%x: tensor<8x!tt.ptr<i32>>, %y: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK-NOT: tt.load
    // CHECK: tt.store %arg1, %arg2
    // CHECK: tt.store %arg1, %arg2
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    tt.store %y, %v : tensor<8x!tt.ptr<i32>>
    %r = tt.load %x : tensor<8x!tt.ptr<i32>>
    tt.store %y, %r : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// Without the licence the store to %y may overwrite %x, so the load stays.
module {
  // CHECK-LABEL: @no_licence
  tt.func @no_licence(%x: tensor<8x!tt.ptr<i32>>, %y: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK: tt.load %arg0
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    tt.store %y, %v : tensor<8x!tt.ptr<i32>>
    %r = tt.load %x : tensor<8x!tt.ptr<i32>>
    tt.store %y, %r : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
