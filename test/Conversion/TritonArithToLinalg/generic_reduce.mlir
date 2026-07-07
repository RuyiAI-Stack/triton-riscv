// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @generic_reduce(%arg0: tensor<4xf32>) -> f32 {
    %0 = "tt.reduce"(%arg0) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      %2 = arith.mulf %1, %rhs : f32
      tt.reduce.return %2 : f32
    }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
    tt.return %0 : f32
  }
}

// CHECK-LABEL: func.func @generic_reduce
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>
// CHECK-NOT: tt.reduce
// CHECK: [[FIRST:%.+]] = tensor.extract [[ARG0]]
// CHECK: [[INIT:%.+]] = tensor.insert [[FIRST]]
// CHECK: [[TAIL:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK: [[REDUCED:%.+]] = linalg.reduce
// CHECK-SAME: ins([[TAIL]] : tensor<3xf32>) outs([[INIT]] : tensor<f32>) dimensions = [0]
// CHECK: ([[CURRENT:%.+]]: f32, [[ACC:%.+]]: f32) {
// CHECK:   [[SUM:%.+]] = arith.addf [[ACC]], [[CURRENT]] : f32
// CHECK:   [[PRODUCT:%.+]] = arith.mulf [[SUM]], [[CURRENT]] : f32
// CHECK:   linalg.yield [[PRODUCT]] : f32
// CHECK: [[RESULT:%.+]] = tensor.extract [[REDUCED]]
// CHECK: return [[RESULT]]
