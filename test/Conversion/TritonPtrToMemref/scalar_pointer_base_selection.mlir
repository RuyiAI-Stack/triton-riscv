// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func @select_pointer_base(
      %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>,
      %cond0: i1, %cond1: i1, %value: f32) {
    %0 = scf.if %cond0 -> (!tt.ptr<f32>) {
      scf.yield %arg0 : !tt.ptr<f32>
    } else {
      %1 = arith.select %cond1, %arg1, %arg2 : !tt.ptr<f32>
      scf.yield %1 : !tt.ptr<f32>
    }
    tt.store %0, %value : !tt.ptr<f32>
    return
  }

  func.func @mixed_pointer_results(
      %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %cond: i1,
      %value: f32, %values: tensor<4xf32>) {
    %selected, %ptrs = scf.if %cond -> (!tt.ptr<f32>, tensor<4x!tt.ptr<f32>>) {
      %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      scf.yield %arg0, %0 : !tt.ptr<f32>, tensor<4x!tt.ptr<f32>>
    } else {
      %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      scf.yield %arg1, %0 : !tt.ptr<f32>, tensor<4x!tt.ptr<f32>>
    }
    tt.store %selected, %value : !tt.ptr<f32>
    tt.store %ptrs, %values : tensor<4x!tt.ptr<f32>>
    return
  }
}

// CHECK-LABEL: func.func @select_pointer_base(
// CHECK-SAME: %[[ARG0:.*]]: memref<*xf32>, %[[ARG1:.*]]: memref<*xf32>, %[[ARG2:.*]]: memref<*xf32>
// CHECK: %[[SELECTED:.*]] = scf.if %{{.*}} -> (memref<*xf32>) {
// CHECK: scf.yield %[[ARG0]] : memref<*xf32>
// CHECK: } else {
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[ARG1]], %[[ARG2]] : memref<*xf32>
// CHECK: scf.yield %[[SELECT]] : memref<*xf32>

// CHECK-LABEL: func.func @mixed_pointer_results(
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<f32>
// CHECK: scf.if %{{.*}} -> (tensor<4x!tt.ptr<f32>>) {
