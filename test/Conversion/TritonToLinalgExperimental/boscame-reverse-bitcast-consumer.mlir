// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: reverse_bitcast_consumer = reject: op was not bufferized
module {
  func.func @reverse_bitcast_consumer(%input: tensor<4xi32>, %upper: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty_i32 = tensor.empty() : tensor<4xi32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 700 : i64}}
        ins(%input : tensor<4xi32>) outs(%empty_i32 : tensor<4xi32>) {
      ^bb0(%value: i32, %unused: i32):
        %next = arith.addi %value, %value : i32
        linalg.yield %next : i32
    } -> tensor<4xi32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xi32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 701 : i64}
          ins(%acc : tensor<4xi32>) outs(%empty_i32 : tensor<4xi32>) {
        ^bb0(%value: i32, %unused: i32):
          %doubled = arith.addi %value, %value : i32
          linalg.yield %doubled : i32
      } -> tensor<4xi32>
      scf.yield %next : tensor<4xi32>
    }
    %bits = tensor.bitcast %loop : tensor<4xi32> to tensor<4xf32>
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 702 : i64}
        ins(%bits : tensor<4xf32>) outs(%empty_f32 : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
