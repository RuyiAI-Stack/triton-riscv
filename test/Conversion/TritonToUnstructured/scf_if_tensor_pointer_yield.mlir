// RUN: triton-shared-opt --split-input-file --triton-to-unstructured --verify-each %s 2>&1 | FileCheck %s

// CHECK:         warning: Cannot transform tensor of pointers into a single base pointer with tensor of offsets
// CHECK:         note: see current operation:
// CHECK:         tt.func public @live_tensor_pointer_if_rolls_back(

module {
  tt.func public @dead_tensor_pointer_if_does_not_block_lowering(
      %tensor_ptrs: tensor<4x!tt.ptr<f32>>, %base: !tt.ptr<f32>,
      %offsets: tensor<4xi32>, %cond: i1) -> tensor<4xf32> {
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      scf.yield %tensor_ptrs : tensor<4x!tt.ptr<f32>>
    } else {
      scf.yield %tensor_ptrs : tensor<4x!tt.ptr<f32>>
    }
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %shifted = tt.addptr %ptrs, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %shifted : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }
}

// CHECK-LABEL:   tt.func public @dead_tensor_pointer_if_does_not_block_lowering(
// CHECK:         %[[VALUE:.*]] = tts.gather %{{.*}}[%{{.*}}] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:         tt.return %[[VALUE]] : tensor<4xf32>
// CHECK-NOT:     tt.load

// -----

module {
  tt.func public @live_tensor_pointer_if_rolls_back(
      %tensor_ptrs_a: tensor<4x!tt.ptr<f32>>,
      %tensor_ptrs_b: tensor<4x!tt.ptr<f32>>, %cond: i1) -> tensor<4xf32> {
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      scf.yield %tensor_ptrs_a : tensor<4x!tt.ptr<f32>>
    } else {
      scf.yield %tensor_ptrs_b : tensor<4x!tt.ptr<f32>>
    }
    %value = tt.load %selected : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }
}

// CHECK-LABEL:   tt.func public @live_tensor_pointer_if_rolls_back(
// CHECK:         %[[SELECTED:.*]] = scf.if %{{.*}} -> (tensor<4x!tt.ptr<f32>>) {
// CHECK:           scf.yield %arg0 : tensor<4x!tt.ptr<f32>>
// CHECK:         } else {
// CHECK:           scf.yield %arg1 : tensor<4x!tt.ptr<f32>>
// CHECK:         }
// CHECK:         %[[VALUE:.*]] = tt.load %[[SELECTED]] : tensor<4x!tt.ptr<f32>>
// CHECK:         tt.return %[[VALUE]] : tensor<4xf32>
