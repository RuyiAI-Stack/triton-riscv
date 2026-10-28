// RUN: triton-shared-opt --structured-to-memref %s | FileCheck %s
// RUN: triton-shared-opt --structured-to-memref="enable-tensor-first-vector-cpu=true" %s | FileCheck %s

module {
  tt.func public @wrap_stacked_runtime_zero_stride(
      %base: !tt.ptr<f32>, %offset: index, %modulo: index,
      %row_stride: index, %col_stride: index) -> tensor<4x4xf32> {
    %ptr = tts.make_tptr %base to sizes: [4, 4],
      strides: [%row_stride, %col_stride], offsets: [%offset, 0],
      shape: [%modulo, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>
    %value = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>,
      static_mask_dims = array<i64>}> :
      (tensor<4x4x!tt.ptr<f32>>) -> tensor<4x4xf32>
    tt.return %value : tensor<4x4xf32>
  }
}

// Both MakeTensorPtrConverter and load-side materialization must protect their
// divisors. Selecting the final row count alone would still execute division
// by zero. The actual memory descriptor keeps the original broadcast strides.
// CHECK-LABEL: tt.func public @wrap_stacked_runtime_zero_stride(
// CHECK-SAME: %[[BASE:.*]]: !tt.ptr<f32>, %[[OFFSET:.*]]: index, %[[MODULO:.*]]: index, %[[ROW:.*]]: index, %[[COL:.*]]: index)
// CHECK: %[[TILE_ROWS0:.*]] = arith.constant 4 : index
// CHECK: %[[TILE_COLS0:.*]] = arith.constant 4 : index
// CHECK: %[[ZERO0:.*]] = arith.constant 0 : index
// CHECK: %[[ONE0:.*]] = arith.constant 1 : index
// CHECK: %[[IS_ZERO0:.*]] = arith.cmpi eq, %[[ROW]], %[[ZERO0]] : index
// CHECK: %[[SAFE0:.*]] = arith.select %[[IS_ZERO0]], %[[ONE0]], %[[ROW]] : index
// CHECK: arith.remsi {{.*}}, %[[SAFE0]] : index
// CHECK: arith.divsi {{.*}}, %[[SAFE0]] : index
// CHECK: %[[COUNT0:.*]] = arith.select %[[IS_ZERO0]], %[[TILE_ROWS0]], %{{.*}} : index
// CHECK: memref.reinterpret_cast {{.*}} sizes: [%[[COUNT0]], %[[TILE_COLS0]]], strides: [%[[ROW]], %[[COL]]]
// CHECK: builtin.unrealized_conversion_cast {{.*}} {wrap_stacked}
// CHECK: %[[TILE_ROWS1:.*]] = arith.constant 4 : index
// CHECK: %[[TILE_COLS1:.*]] = arith.constant 4 : index
// CHECK: %[[ZERO1:.*]] = arith.constant 0 : index
// CHECK: %[[ONE1:.*]] = arith.constant 1 : index
// CHECK: %[[IS_ZERO1:.*]] = arith.cmpi eq, %[[ROW]], %[[ZERO1]] : index
// CHECK: %[[SAFE1:.*]] = arith.select %[[IS_ZERO1]], %[[ONE1]], %[[ROW]] : index
// CHECK: arith.remsi {{.*}}, %[[SAFE1]] : index
// CHECK: arith.divsi {{.*}}, %[[SAFE1]] : index
// CHECK: %[[COUNT1:.*]] = arith.select %[[IS_ZERO1]], %[[TILE_ROWS1]], %{{.*}} : index
// CHECK: memref.reinterpret_cast {{.*}} sizes: [%[[COUNT1]], %[[TILE_COLS1]]], strides: [%[[ROW]], %[[COL]]]
