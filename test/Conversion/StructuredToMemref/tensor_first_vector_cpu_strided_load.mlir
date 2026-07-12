// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

// A stride-2 source must be rejected by masked elementwise fusion before that
// pattern mutates IR. The regular load/store conversion can then lower it.

// CHECK-LABEL: func.func @strided_load(
// CHECK: %[[SRC:.*]] = memref.reinterpret_cast {{.*}} strides: [2]
// CHECK: %[[SLICE:.*]] = memref.subview %[[SRC]]
// CHECK: scf.for
// CHECK: memref.load %[[SLICE]]
// CHECK: linalg.generic
// CHECK: tensor.extract
// CHECK: memref.store

module {
  tt.func public @strided_load(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %two = arith.constant dense<2> : tensor<4xi32>
    %zero = arith.constant dense<0.0> : tensor<4xf32>
    %strided = arith.muli %range, %two : tensor<4xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %strided : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %mask = arith.cmpi slt, %range, %two : tensor<4xi32>
    %values = tt.load %src_ptrs, %mask, %zero : tensor<4x!tt.ptr<f32>>
    %squared = arith.mulf %values, %values : tensor<4xf32>
    tt.store %dst_ptrs, %squared, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
