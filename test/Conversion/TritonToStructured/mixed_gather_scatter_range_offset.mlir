// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s

module {
// Preserve the lane contribution when a structured range and loaded index are added.
// CHECK-LABEL: tt.func @mixed_range_offset
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[SCALED:.*]] = arith.muli %[[RANGE]], {{.*}} : tensor<8xi32>
// CHECK: %[[INDEX:.*]] = "tts.load"
// CHECK: %[[MIXED:.*]] = arith.addi %[[SCALED]], %[[INDEX]] : tensor<8xi32>
// CHECK: tts.make_gather_scatter_tptr {{.*}} gather_scatter_offset: %[[MIXED]], strides: [1], offsets: [0]
  tt.func @mixed_range_offset(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>, %output: !tt.ptr<f32>) {
    %four = arith.constant dense<4> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %scaled = arith.muli %range, %four : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %offset = arith.addi %scaled, %idx : tensor<8xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = tt.load %input_ptr : tensor<8x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %output_ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// A base in the non-gather dimension must not be counted twice.
// Keep the original expression for the existing unstructured lowering.
// CHECK-LABEL: tt.func @grouped_non_gather_offset_reverse
// CHECK-DAG: %[[BASE_R:.*]] = arith.constant dense<13> : tensor<1x8xi32>
// CHECK-DAG: %[[STRUCTURED_R:.*]] = arith.addi {{.*}}, %[[BASE_R]] : tensor<1x8xi32>
// CHECK-DAG: %[[INDEX_R:.*]] = "tts.load"
// CHECK: %[[OFFSET_R:.*]] = arith.addi %[[INDEX_R]], %[[STRUCTURED_R]] : tensor<1x8xi32>
// CHECK-NOT: tts.make_gather_scatter_tptr
// CHECK: %[[PTR_R:.*]] = tt.addptr {{.*}}, %[[OFFSET_R]] : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
// CHECK: tt.store %[[PTR_R]],
  tt.func @grouped_non_gather_offset_reverse(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %four = arith.constant dense<4> : tensor<1x8xi32>
    %thirteen = arith.constant dense<13> : tensor<1x8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %columns = tt.expand_dims %range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %scaled = arith.muli %columns, %four : tensor<1x8xi32>
    %structured = arith.addi %scaled, %thirteen : tensor<1x8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<1x8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %columns : tensor<1x8x!tt.ptr<i32>>, tensor<1x8xi32>
    %idx = tt.load %idx_ptr : tensor<1x8x!tt.ptr<i32>>
    %offset = arith.addi %idx, %structured : tensor<1x8xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
    %value = arith.constant dense<1.0> : tensor<1x8xf32>
    tt.store %ptr, %value : tensor<1x8x!tt.ptr<f32>>
    tt.return
  }
}
