// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @scalar_mask(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %index: i32, %mask: i1, %other: f32) {
    %in = tt.addptr %input, %index : !tt.ptr<f32>, i32
    %value = tt.load %in, %mask, %other : !tt.ptr<f32>
    %out = tt.addptr %output, %index : !tt.ptr<f32>, i32
    tt.store %out, %value, %mask : !tt.ptr<f32>
    tt.return
  }

  tt.func public @scalar_mask_no_other(%input: !tt.ptr<f32>, %index: i32, %mask: i1) -> f32 {
    %in = tt.addptr %input, %index : !tt.ptr<f32>, i32
    %value = tt.load %in, %mask : !tt.ptr<f32>
    tt.return %value : f32
  }
}

// CHECK-LABEL: tt.func public @scalar_mask
// CHECK-SAME: %[[MASK:arg[0-9]+]]: i1, %[[OTHER:arg[0-9]+]]: f32
// CHECK: %[[VALUE:.*]] = scf.if %[[MASK]] -> (f32) {
// CHECK: %[[LOADED:.*]] = memref.load
// CHECK: scf.yield %[[LOADED]] : f32
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[OTHER]] : f32
// CHECK: scf.if %[[MASK]] {
// CHECK-NEXT: affine.store %[[VALUE]]
// CHECK: tt.return

// CHECK-LABEL: tt.func public @scalar_mask_no_other
// CHECK-SAME: %[[MASK2:arg[0-9]+]]: i1
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VALUE2:.*]] = scf.if %[[MASK2]] -> (f32) {
// CHECK: memref.load
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[ZERO]] : f32
// CHECK: tt.return %[[VALUE2]] : f32
