// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @left_projection() -> i32 {
    %input = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %result = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%accumulator: i32, %current: i32):
      tt.reduce.return %accumulator : i32
    }) : (tensor<4xi32>) -> i32
    tt.return %result : i32
  }
}

// CHECK-LABEL: func.func @left_projection
// CHECK:         %[[REDUCED:.*]] = linalg.reduce
// CHECK:           (%[[CURRENT:.*]]: i32, %[[ACCUMULATED:.*]]: i32) {
// CHECK-NEXT:        linalg.yield %[[ACCUMULATED]] : i32
// CHECK:         %[[RESULT:.*]] = tensor.extract %[[REDUCED]][] : tensor<i32>
// CHECK:         return %[[RESULT]] : i32
