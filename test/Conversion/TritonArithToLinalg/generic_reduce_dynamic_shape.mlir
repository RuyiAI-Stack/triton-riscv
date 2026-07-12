// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @dynamic_left_projection(%input: tensor<?x?xi32>) -> tensor<?xi32> {
    %result = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%accumulator: i32, %current: i32):
      tt.reduce.return %accumulator : i32
    }) : (tensor<?x?xi32>) -> tensor<?xi32>
    tt.return %result : tensor<?xi32>
  }
}

// CHECK-LABEL: func.func @dynamic_left_projection
// CHECK: %[[RETAINED_DIM:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xi32>
// CHECK: %[[FIRST:.*]] = tensor.extract_slice %{{.*}}[0, 0] [1, %[[RETAINED_DIM]]] [1, 1] : tensor<?x?xi32> to tensor<?xi32>
// CHECK: %[[REDUCED_DIM:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?xi32>
// CHECK: %[[REST_SIZE:.*]] = arith.subi %[[REDUCED_DIM]], %{{.*}} : index
// CHECK: %[[REST:.*]] = tensor.extract_slice %{{.*}}[1, 0] [%[[REST_SIZE]], %[[RETAINED_DIM]]] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
// CHECK: linalg.reduce ins(%[[REST]] : tensor<?x?xi32>) outs(%[[FIRST]] : tensor<?xi32>) dimensions = [0]
