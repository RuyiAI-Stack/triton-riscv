// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: reverse_insert_destination = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: reverse_memref_reshape_consumer = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: reverse_scatter_destination = reject: op was not bufferized
module {
  func.func @reverse_insert_destination(%input: tensor<4xf32>, %upper: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %idx = arith.constant 2 : index
    %value = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 730 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%element: f32, %unused: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xf32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 731 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%element: f32, %unused: f32):
          %doubled = arith.addf %element, %element : f32
          linalg.yield %doubled : f32
      } -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %inserted = tensor.insert %value into %loop[%idx] : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 732 : i64}
        ins(%inserted : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%element: f32, %unused: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @reverse_scatter_destination(%input: tensor<4xf32>, %upper: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 733 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%element: f32, %unused: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed) -> (tensor<4xf32>) {
      %next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 734 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%element: f32, %unused: f32):
          %doubled = arith.addf %element, %element : f32
          linalg.yield %doubled : f32
      } -> tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    %source = tensor.empty() : tensor<1x1xf32>
    %indices = tensor.empty() : tensor<1x1xindex>
    %scattered = tensor.scatter %source into %loop[%indices]
        scatter_dims([0]) unique :
        (tensor<1x1xf32>, tensor<4xf32>, tensor<1x1xindex>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 735 : i64}
        ins(%scattered : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%element: f32, %unused: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @reverse_memref_reshape_consumer(%input: memref<4xf32>,
                                              %output: memref<4xf32>,
                                              %shape: memref<1xi32>,
                                              %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %input) -> (memref<4xf32>) {
      linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 737 : i64}
          ins(%acc : memref<4xf32>) outs(%output : memref<4xf32>) {
        ^bb0(%element: f32, %unused: f32):
          linalg.yield %element : f32
      }
      scf.yield %output : memref<4xf32>
    }
    %reshaped = memref.reshape %loop(%shape)
        : (memref<4xf32>, memref<1xi32>) -> memref<4xf32>
    linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 736 : i64}
        ins(%reshaped : memref<4xf32>) outs(%output : memref<4xf32>) {
      ^bb0(%element: f32, %unused: f32):
        linalg.yield %element : f32
    }
    return
  }
}
