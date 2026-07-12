// RUN: triton-shared-opt --structured-to-memref %s | FileCheck %s

module {
  tt.func public @wrap_linear_load_runtime_zero_stride(
      %arg0: !tt.ptr<f32>, %offset: index, %modulo: index,
      %stride: index) -> tensor<4xf32> {
    %0 = tts.make_tptr %arg0 to sizes: [4], strides: [%stride],
      offsets: [%offset], shape: [%modulo], order: [] : <f32>
      to tensor<4x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>,
      static_mask_dims = array<i64>}> :
      (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return %1 : tensor<4xf32>
  }
}

// CHECK-LABEL:   tt.func public @wrap_linear_load_runtime_zero_stride(
// CHECK:           scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[DISTANCE:.*]] = arith.muli %[[IV]], %{{.*}} : index
// CHECK:             %[[LINEAR:.*]] = arith.addi %{{.*}}, %[[DISTANCE]] : index
// CHECK:             %[[REMAINDER:.*]] = arith.remsi %[[LINEAR]], %[[MODULO:.*]] : index
// CHECK:             %[[IS_NEGATIVE:.*]] = arith.cmpi slt, %[[REMAINDER]], %{{.*}} : index
// CHECK:             %[[ADJUSTED:.*]] = arith.addi %[[REMAINDER]], %[[MODULO]] : index
// CHECK:             %[[WRAPPED:.*]] = arith.select %[[IS_NEGATIVE]], %[[ADJUSTED]], %[[REMAINDER]] : index
// CHECK:             memref.reinterpret_cast %{{.*}} to offset: [%[[WRAPPED]]]
