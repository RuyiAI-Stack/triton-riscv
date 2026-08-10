// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s
module {
  tt.func public @minmax_sgt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi sgt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}
// CHECK-LABEL:   func.func @minmax_sgt(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -2147483648 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_1]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xi32>) outs(%[[FILL_1]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:               %[[MAXSI_0:.*]] = arith.maxsi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:               linalg.yield %[[MAXSI_0]] : i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<i32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }


// -----

module {
  tt.func public @minmax_ugt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ugt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}


// CHECK-LABEL:   func.func @minmax_ugt(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_1]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xi32>) outs(%[[FILL_1]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:               %[[MAXUI_0:.*]] = arith.maxui %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:               linalg.yield %[[MAXUI_0]] : i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<i32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }

// -----

module {
  tt.func public @minmax_slt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi slt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL:   func.func @minmax_slt(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2147483647 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_1]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xi32>) outs(%[[FILL_1]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:               %[[MINSI_0:.*]] = arith.minsi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:               linalg.yield %[[MINSI_0]] : i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<i32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }


// -----

module {
  tt.func public @minmax_ult(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ult, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL:   func.func @minmax_ult(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -1 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_1]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4096xi32>) outs(%[[FILL_1]] : tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:               %[[MINUI_0:.*]] = arith.minui %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:               linalg.yield %[[MINUI_0]] : i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<i32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:           affine.store %[[EXTRACT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
