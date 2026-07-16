// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s

module {
  tt.func public @maxnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.maxnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @maxnumf(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_0]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_1]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xf32>) outs(%[[FILL_1]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) {
// CHECK:               %[[MAXNUMF_0:.*]] = arith.maxnumf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:               linalg.yield %[[MAXNUMF_0]] : f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }


// -----


module {
  tt.func public @minnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.minnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @minnumf(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0x7F800000 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_0]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_1]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xf32>) outs(%[[FILL_1]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) {
// CHECK:               %[[MINNUMF_0:.*]] = arith.minnumf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:               linalg.yield %[[MINNUMF_0]] : f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
