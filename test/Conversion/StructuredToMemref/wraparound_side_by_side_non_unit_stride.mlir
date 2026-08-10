// RUN: triton-shared-opt --split-input-file --structured-to-memref %s | FileCheck %s
// RUN: triton-shared-opt --split-input-file --structured-to-memref="enable-tensor-first-vector-cpu=true" %s | FileCheck %s --check-prefix=TENSOR-FIRST

module {
  tt.func public @wrap_side_by_side_load_non_unit_stride(%arg0: !tt.ptr<f32>, %offset: index, %modulo: index) -> tensor<2x4xf32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %0 = tts.make_tptr %arg0 to sizes: [2, 4], strides: [%c32, %c2], offsets: [%c0, %offset], shape: [0, %modulo], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<f32>>) -> tensor<2x4xf32>
    tt.return %1 : tensor<2x4xf32>
  }

  tt.func public @wrap_side_by_side_load_zero_row_stride(%arg0: !tt.ptr<f32>, %offset: index, %modulo: index) -> tensor<1x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tts.make_tptr %arg0 to sizes: [1, 4], strides: [%c0, %c1], offsets: [%c0, %offset], shape: [0, %modulo], order: [] : <f32> to tensor<1x4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1x4x!tt.ptr<f32>>) -> tensor<1x4xf32>
    tt.return %1 : tensor<1x4xf32>
  }

  tt.func public @wrap_side_by_side_load_runtime_zero_row_stride(
      %arg0: !tt.ptr<f32>, %offset: index, %modulo: index,
      %row_stride: index) -> tensor<1x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tts.make_tptr %arg0 to sizes: [1, 4],
      strides: [%row_stride, %c1], offsets: [%c0, %offset],
      shape: [0, %modulo], order: [] : <f32>
      to tensor<1x4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>,
      static_mask_dims = array<i64>}> :
      (tensor<1x4x!tt.ptr<f32>>) -> tensor<1x4xf32>
    tt.return %1 : tensor<1x4xf32>
  }

  tt.func public @wrap_side_by_side_load_zero_col_stride(
      %arg0: !tt.ptr<f32>, %offset: index, %modulo: index)
      -> tensor<1x4xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = tts.make_tptr %arg0 to sizes: [1, 4], strides: [%c8, %c0],
      offsets: [%c0, %offset], shape: [0, %modulo], order: [] : <f32>
      to tensor<1x4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>,
      static_mask_dims = array<i64>}> :
      (tensor<1x4x!tt.ptr<f32>>) -> tensor<1x4xf32>
    tt.return %1 : tensor<1x4xf32>
  }
}

// CHECK-LABEL:   tt.func public @wrap_side_by_side_load_non_unit_stride(
// CHECK:           %9 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %10 = arith.addi %c0, %arg1 : index
// CHECK:           %11 = arith.remsi %10, %c32 : index
// CHECK:           %12 = arith.subi %10, %11 : index
// CHECK:           %13 = arith.divsi %11, %c2 : index
// CHECK:           %14 = arith.addi %13, %c4_5 : index
// CHECK:           %15 = arith.minsi %14, %arg2 : index
// CHECK:           %16 = arith.subi %15, %13 : index
// CHECK:           %17 = arith.subi %c4_5, %16 : index
// CHECK:           %reinterpret_cast_6 = memref.reinterpret_cast %9 to offset: [%10], sizes: [%c2_4, %16], strides: [%c32_2, %c2_3] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           %reinterpret_cast_7 = memref.reinterpret_cast %9 to offset: [%12], sizes: [%c2_4, %17], strides: [%c32_2, %c2_3] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>

// TENSOR-FIRST-LABEL:   tt.func public @wrap_side_by_side_load_zero_row_stride(
// TENSOR-FIRST-SAME:        %[[PTR:.*]]: !tt.ptr<f32>, %[[OFFSET:.*]]: index, %[[MODULO:.*]]: index)
// TENSOR-FIRST:           %[[C1:.*]] = arith.constant 1 : index
// TENSOR-FIRST:           %[[ROW_SPAN:.*]] = arith.muli %[[MODULO]], %[[C1]] : index
// TENSOR-FIRST:           %[[INTRA_ROW:.*]] = arith.remsi {{.*}}, %[[ROW_SPAN]] : index

// TENSOR-FIRST-LABEL:   tt.func public @wrap_side_by_side_load_runtime_zero_row_stride(
// TENSOR-FIRST-SAME:        %[[PTR:.*]]: !tt.ptr<f32>, %[[OFFSET:.*]]: index, %[[MODULO:.*]]: index, %[[ROW_STRIDE:.*]]: index)
// TENSOR-FIRST:           %[[ROW_SPAN:.*]] = arith.muli %[[MODULO]], {{.*}} : index
// TENSOR-FIRST:           %[[C0_SAFE:.*]] = arith.constant 0 : index
// TENSOR-FIRST:           %[[ROW_STRIDE_ZERO:.*]] = arith.cmpi eq, %[[ROW_STRIDE]], %[[C0_SAFE]] : index
// TENSOR-FIRST:           %[[SELECTED_ROW_SPAN:.*]] = arith.select %[[ROW_STRIDE_ZERO]], %[[ROW_SPAN]], %[[ROW_STRIDE]] : index
// TENSOR-FIRST:           arith.remsi {{.*}}, %[[SELECTED_ROW_SPAN]] : index

// TENSOR-FIRST-LABEL:   tt.func public @wrap_side_by_side_load_zero_col_stride(
// TENSOR-FIRST-NOT:       arith.divsi
// TENSOR-FIRST:           builtin.unrealized_conversion_cast {{.*}} {wrap_side_by_side}
