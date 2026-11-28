// RUN: triton-opt --split-input-file %s -im-dead-stores | FileCheck %s

// The first store to %x is overwritten before anything reads %x. With the licence the
// load of %a in between reads another buffer.
module attributes {im.noalias_args = 1 : i64} {
  // CHECK-LABEL: @dead
  tt.func @dead(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK-NOT: tt.store %arg0, %arg2
    // CHECK: tt.load %arg1
    // CHECK: tt.store %arg0
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    %w = tt.load %a : tensor<8x!tt.ptr<i32>>
    tt.store %x, %w : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// A read of %x between the two stores keeps the first.
module attributes {im.noalias_args = 1 : i64} {
  // CHECK-LABEL: @read_between
  tt.func @read_between(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK: tt.store %arg0, %arg2
    // CHECK: tt.load %arg0
    // CHECK: tt.store %arg0
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    %w = tt.load %x : tensor<8x!tt.ptr<i32>>
    tt.store %x, %w : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// Without the licence the load of %a may read %x, which keeps the first store.
module {
  // CHECK-LABEL: @no_licence
  tt.func @no_licence(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %v: tensor<8xi32>) {
    // CHECK: tt.store %arg0, %arg2
    tt.store %x, %v : tensor<8x!tt.ptr<i32>>
    %w = tt.load %a : tensor<8x!tt.ptr<i32>>
    tt.store %x, %w : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
