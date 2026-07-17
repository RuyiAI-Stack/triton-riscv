// RUN: triton-shared-opt --split-input-file --structured-to-memref %s | FileCheck %s
// RUN: triton-shared-opt --split-input-file --structured-to-memref --canonicalize %s | FileCheck %s --check-prefix=CHECK-CANON
// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s --check-prefix=CHECK-LINALG

module {
  tt.func public @wrap_linear_load_non_unit_stride(%arg0: !tt.ptr<f32>, %offset: index, %modulo: index, %stride: index) -> tensor<4xf32> {
    %0 = tts.make_tptr %arg0 to sizes: [4], strides: [%stride], offsets: [%offset], shape: [%modulo], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return %1 : tensor<4xf32>
  }

  // Access pattern is 8, 1, 4, 7, 0, 3, 6, 9 and wraps more than once.
  tt.func public @wrap_linear_load_non_unit_stride_repeated_wrap(%arg0: !tt.ptr<f32>) -> tensor<8xf32> {
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %0 = tts.make_tptr %arg0 to sizes: [8], strides: [%c3], offsets: [%c8], shape: [%c10], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<f32>>) -> tensor<8xf32>
    tt.return %1 : tensor<8xf32>
  }

  tt.func public @wrap_linear_store_non_unit_stride_repeated_wrap(%arg0: !tt.ptr<f32>, %value: tensor<8xf32>) {
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %0 = tts.make_tptr %arg0 to sizes: [8], strides: [%c3], offsets: [%c8], shape: [%c10], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    "tts.store"(%0, %value) <{static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) -> ()
    tt.return
  }

  // Access pattern is 1, 8, 5, 2 and must not produce negative memref offsets.
  tt.func public @wrap_linear_load_negative_stride(%arg0: !tt.ptr<f32>) -> tensor<4xf32> {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %cm3 = arith.constant -3 : index
    %0 = tts.make_tptr %arg0 to sizes: [4], strides: [%cm3], offsets: [%c1], shape: [%c10], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return %1 : tensor<4xf32>
  }

  // Access pattern is 9, 2, 5, 8 and must normalize the negative start.
  tt.func public @wrap_linear_store_negative_offset(%arg0: !tt.ptr<f32>, %value: tensor<4xf32>) {
    %cm1 = arith.constant -1 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %0 = tts.make_tptr %arg0 to sizes: [4], strides: [%c3], offsets: [%cm1], shape: [%c10], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    "tts.store"(%0, %value) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @wrap_linear_load_non_unit_stride(
// CHECK-SAME:        %[[PTR:.*]]: !tt.ptr<f32>, %[[OFFSET:.*]]: index, %[[MODULO:.*]]: index, %[[STRIDE:.*]]: index)
// CHECK:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[START:.*]] = arith.remsi %[[OFFSET]], %[[MODULO]] : index
// CHECK:           builtin.unrealized_conversion_cast %[[BASE]], %[[START]], %[[MODULO]], %[[STRIDE]] : memref<*xf32>, index, index, index to tensor<4x!tt.ptr<f32>> {wrap_linear}

// CHECK-CANON-LABEL:   tt.func public @wrap_linear_load_non_unit_stride_repeated_wrap(
// CHECK-CANON-DAG:       %[[BASE:.*]] = builtin.unrealized_conversion_cast
// CHECK-CANON-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-CANON-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-CANON-DAG:       %[[C10:.*]] = arith.constant 10 : index
// CHECK-CANON:           scf.for %[[IV:.*]] = %{{.*}} to %[[C8]] step %{{.*}} {
// CHECK-CANON:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %[[C3]] : index
// CHECK-CANON:             %[[LINEAR:.*]] = arith.addi %[[DISTANCE]], %[[C8]] : index
// CHECK-CANON:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[C10]] : index
// CHECK-CANON:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK-CANON:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[C10]] : index
// CHECK-CANON:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK-CANON:             %[[ELEMENT:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[WRAPPED]]], sizes: [1], strides: [1]
// CHECK-CANON:             memref.load %[[ELEMENT]]
// CHECK-CANON-NOT:       memref.copy

// CHECK-CANON-LABEL:   tt.func public @wrap_linear_store_non_unit_stride_repeated_wrap(
// CHECK-CANON:           scf.for %[[IV:.*]] = %{{.*}} to %[[C8:.*]] step %{{.*}} {
// CHECK-CANON:             %[[VALUE:.*]] = tensor.extract %{{.*}}[%[[IV]]]
// CHECK-CANON:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %[[C3:.*]] : index
// CHECK-CANON:             %[[LINEAR:.*]] = arith.addi %[[DISTANCE]], %[[C8]] : index
// CHECK-CANON:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[C10:.*]] : index
// CHECK-CANON:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK-CANON:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[C10]] : index
// CHECK-CANON:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK-CANON:             %[[ELEMENT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[WRAPPED]]], sizes: [1], strides: [1]
// CHECK-CANON:             memref.store %[[VALUE]], %[[ELEMENT]]

// CHECK-CANON-LABEL:   tt.func public @wrap_linear_load_negative_stride(
// CHECK-CANON-DAG:       %[[NEG3:.*]] = arith.constant -3 : index
// CHECK-CANON-DAG:       %[[C10:.*]] = arith.constant 10 : index
// CHECK-CANON-DAG:       %[[ONE:.*]] = arith.constant 1 : index
// CHECK-CANON:           scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-CANON:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %[[NEG3]] : index
// CHECK-CANON:             %[[LINEAR:.*]] = arith.addi %[[DISTANCE]], %[[ONE]] : index
// CHECK-CANON:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[C10]] : index
// CHECK-CANON:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK-CANON:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[C10]] : index
// CHECK-CANON:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK-CANON:             %[[ELEMENT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[WRAPPED]]], sizes: [1], strides: [1]
// CHECK-CANON:             memref.load %[[ELEMENT]]

// CHECK-CANON-LABEL:   tt.func public @wrap_linear_store_negative_offset(
// CHECK-CANON-DAG:       %[[NEG1:.*]] = arith.constant -1 : index
// CHECK-CANON-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-CANON-DAG:       %[[C10:.*]] = arith.constant 10 : index
// CHECK-CANON:           scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-CANON:             %[[VALUE:.*]] = tensor.extract %{{.*}}[%[[IV]]]
// CHECK-CANON:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %[[C3]] : index
// CHECK-CANON:             %[[LINEAR:.*]] = arith.addi %[[DISTANCE]], %[[NEG1]] : index
// CHECK-CANON:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[C10]] : index
// CHECK-CANON:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK-CANON:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[C10]] : index
// CHECK-CANON:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK-CANON:             %[[ELEMENT:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%[[WRAPPED]]], sizes: [1], strides: [1]
// CHECK-CANON:             memref.store %[[VALUE]], %[[ELEMENT]]

// CHECK-LINALG-LABEL:   func.func @wrap_linear_load_non_unit_stride_repeated_wrap(
// CHECK-LINALG:           scf.for %[[IV:.*]] = %{{.*}} to %[[C8:.*]] step %{{.*}} {
// CHECK-LINALG:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %[[C3:.*]] : index
// CHECK-LINALG:             %[[LINEAR:.*]] = arith.addi %[[DISTANCE]], %[[C8]] : index
// CHECK-LINALG:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[C10:.*]] : index
// CHECK-LINALG:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK-LINALG:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[C10]] : index
// CHECK-LINALG:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK-LINALG:             memref.reinterpret_cast %{{.*}} to offset: [%[[WRAPPED]]], sizes: [1], strides: [1]
