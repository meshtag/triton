// RUN: triton-opt --split-input-file %s -im-register-promotion | FileCheck %s

// %x is accumulated in memory across the loop. With the licence %a is another buffer, so
// %x is loaded once before the loop, carried in a register and stored once after it.
module attributes {im.noalias_args = 1 : i64} {
  // CHECK-LABEL: @promote
  tt.func @promote(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: %[[INIT:.*]] = tt.load %arg0
    // CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]])
    // CHECK-NOT: tt.load %arg0
    // CHECK-NOT: tt.store
    // CHECK: scf.yield
    // CHECK: tt.store %arg0, %[[R]]
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %x : tensor<8x!tt.ptr<i32>>
      %w = tt.load %a : tensor<8x!tt.ptr<i32>>
      %s = arith.addi %v, %w : tensor<8xi32>
      tt.store %x, %s : tensor<8x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// Without the licence %a may be %x, so the loop keeps its load and store of %x.
module {
  // CHECK-LABEL: @no_licence
  tt.func @no_licence(%x: tensor<8x!tt.ptr<i32>>, %a: tensor<8x!tt.ptr<i32>>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: tt.load %arg0
    // CHECK: tt.store %arg0
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = tt.load %x : tensor<8x!tt.ptr<i32>>
      %w = tt.load %a : tensor<8x!tt.ptr<i32>>
      %s = arith.addi %v, %w : tensor<8xi32>
      tt.store %x, %s : tensor<8x!tt.ptr<i32>>
    }
    tt.return
  }
}
