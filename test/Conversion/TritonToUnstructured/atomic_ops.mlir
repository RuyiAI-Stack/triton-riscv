// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @atomic_add_tensor(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>, %arg2: tensor<4xf32>, %arg3: tensor<4xi1>) -> tensor<4xf32> {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %1 = tt.addptr %0, %arg1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %2 = tt.atomic_rmw fadd, relaxed, gpu, %1, %arg2, %arg3 : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
    tt.return %2 : tensor<4xf32>
  }

  tt.func public @atomic_cas_scalar(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
    %0 = tt.addptr %arg0, %arg1 : !tt.ptr<i32>, i32
    %1 = tt.atomic_cas acq_rel, gpu, %0, %arg2, %arg3 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return %1 : i32
  }
}

// CHECK-LABEL:   tt.func public @atomic_add_tensor(
// CHECK:           [[ATOM:%.+]] = tts.atomic_rmw fadd, relaxed, gpu, %arg2 into %arg0[%arg1] mask = %arg3 : tensor<4xf32> into (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:           tt.return [[ATOM]] : tensor<4xf32>

// CHECK-LABEL:   tt.func public @atomic_cas_scalar(
// CHECK:           [[CAS:%.+]] = tts.atomic_cas acq_rel, gpu, %arg2, %arg3 into %arg0[%arg1] : i32, i32 into (<i32>, i32) -> i32
// CHECK:           tt.return [[CAS]] : i32
