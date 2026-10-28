// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s
//
// Regression: a non-affine row mask AND a structured one-column bound AND
// a broadcast scalar condition must not discard the one-column bound.
// An incomplete MaskState must preserve the complete tt.load mask by falling
// back, rather than emit tts.load with static_mask_dims = [0, 4].
//
// CHECK-LABEL: tt.func public @masked_broadcast_conjunction
// CHECK-NOT: static_mask_dims = array<i64: 0, 4>
// CHECK: tt.load
//
module {
  tt.func public @masked_broadcast_conjunction(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %enable: i1) {
    %zero = arith.constant dense<0.0> : tensor<4x4xf32>
    %one = arith.constant dense<1> : tensor<4xi32>
    %two = arith.constant dense<2> : tensor<4x1xi32>
    %four = arith.constant dense<4> : tensor<4x1xi32>
    %row = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %col = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %rows = tt.expand_dims %row {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %cols = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_offset = arith.muli %rows, %four : tensor<4x1xi32>
    %row_grid = tt.broadcast %row_offset : tensor<4x1xi32> -> tensor<4x4xi32>
    %col_grid = tt.broadcast %cols : tensor<1x4xi32> -> tensor<4x4xi32>
    %offsets = arith.addi %row_grid, %col_grid : tensor<4x4xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %input_ptrs = tt.addptr %input_base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %row_mod = arith.remsi %rows, %two : tensor<4x1xi32>
    %row_valid = arith.cmpi slt, %row_mod, %two : tensor<4x1xi32>
    %row_mask = tt.broadcast %row_valid : tensor<4x1xi1> -> tensor<4x4xi1>
    %col_valid = arith.cmpi slt, %col, %one : tensor<4xi32>
    %col_expand = tt.expand_dims %col_valid {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %col_mask = tt.broadcast %col_expand : tensor<1x4xi1> -> tensor<4x4xi1>
    %matrix_mask = arith.andi %col_mask, %row_mask : tensor<4x4xi1>
    %scalar_mask = tt.splat %enable : i1 -> tensor<4x4xi1>
    %mask = arith.andi %matrix_mask, %scalar_mask : tensor<4x4xi1>
    %value = tt.load %input_ptrs, %mask, %zero : tensor<4x4x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %output_ptrs = tt.addptr %output_base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %output_ptrs, %value : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}
