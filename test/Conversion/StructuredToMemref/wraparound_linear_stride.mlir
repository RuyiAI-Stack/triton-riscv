// RUN: triton-shared-opt --split-input-file --structured-to-memref %s | FileCheck %s

module {
  tt.func public @wrap_linear_load_non_unit_stride(%arg0: !tt.ptr<f32>, %offset: index, %modulo: index, %stride: index) -> tensor<4xf32> {
    %0 = tts.make_tptr %arg0 to sizes: [4], strides: [%stride], offsets: [%offset], shape: [%modulo], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return %1 : tensor<4xf32>
  }
}

// CHECK-LABEL:   tt.func public @wrap_linear_load_non_unit_stride(
// CHECK-SAME:        %[[PTR:.*]]: !tt.ptr<f32>, %[[OFFSET:.*]]: index, %[[MODULO:.*]]: index, %[[STRIDE:.*]]: index)
// CHECK:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[START:.*]] = arith.remsi %[[OFFSET]], %[[MODULO]] : index
// CHECK:           %[[REMAINING:.*]] = arith.subi %[[MODULO]], %[[START]] : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[STRIDE_MINUS_ONE:.*]] = arith.subi %[[STRIDE]], %[[C1]] : index
// CHECK:           %[[NUMERATOR:.*]] = arith.addi %[[REMAINING]], %[[STRIDE_MINUS_ONE]] : index
// CHECK:           %[[ELEMENTS_UNTIL_WRAP:.*]] = arith.divsi %[[NUMERATOR]], %[[STRIDE]] : index
// CHECK:           %[[D1:.*]] = arith.minsi %[[C4]], %[[ELEMENTS_UNTIL_WRAP]] : index
// CHECK:           %[[D2:.*]] = arith.subi %[[C4]], %[[D1]] : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[CAST0:.*]] = memref.reinterpret_cast %[[BASE]] to offset: {{\[}}%[[START]]], sizes: {{\[}}%[[D1]]], strides: {{\[}}%[[STRIDE]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:           %[[CAST1:.*]] = memref.reinterpret_cast %[[BASE]] to offset: {{\[}}%[[ZERO]]], sizes: {{\[}}%[[D2]]], strides: {{\[}}%[[STRIDE]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:           builtin.unrealized_conversion_cast %[[CAST0]], %[[CAST1]] : memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>> to tensor<4x!tt.ptr<f32>> {wrap_linear}
