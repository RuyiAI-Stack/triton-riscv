// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: dynamic_seed_sum = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: generic_reduction = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: hidden_read = reject: kernel contract V25: generic requires pure scalar arithmetic operations
// CONTRACT-CASE: hidden_write = reject: kernel contract V25: generic requires pure scalar arithmetic operations
// CONTRACT-CASE: multi_operation_reduction = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: multi_output_extrema = complete
// CONTRACT-CASE: nan_neutral_with_nnan = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: nan_neutral_without_nnan = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: parallel_and_fill = complete
// CONTRACT-CASE: unknown_call = reject: kernel contract V25: generic requires pure scalar arithmetic operations
// CONTRACT-CASE: unmatched_multi_operation_combiner = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: vector_element_maximum = reject: kernel contract V25: pointwise update requires parallel identity indexing
// CONTRACT-CASE: zero_filled_sum = reject: kernel contract V25: pointwise update requires parallel identity indexing
module {
  func.func private @unknown(f32) -> f32

  // A reduction written as generic has the same stage boundary as a named
  // reduction. Its tensor semantics alone do not permit rematerialization.
  func.func @generic_reduction(%input: tensor<4x8xf32>, %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%sum : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @multi_operation_reduction(%input: tensor<4x8xf32>,
                                       %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %square = arith.mulf %value, %value : f32
        %next = arith.addf %square, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @zero_filled_sum(%input: tensor<4x8xf32>) -> tensor<4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @dynamic_seed_sum(%input: tensor<4x8xf32>, %seed: f32)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%seed : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.addf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @multi_output_extrema(%maximum_input: tensor<4x8xf32>,
                                  %minimum_input: tensor<4x8xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative_infinity = arith.constant 0xFF800000 : f32
    %positive_infinity = arith.constant 0x7F800000 : f32
    %empty_max = tensor.empty() : tensor<4xf32>
    %empty_min = tensor.empty() : tensor<4xf32>
    %init_max = linalg.fill ins(%negative_infinity : f32)
        outs(%empty_max : tensor<4xf32>) -> tensor<4xf32>
    %init_min = linalg.fill ins(%positive_infinity : f32)
        outs(%empty_min : tensor<4xf32>) -> tensor<4xf32>
    %maximum, %minimum = linalg.reduce
        ins(%maximum_input, %minimum_input : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init_max, %init_min : tensor<4xf32>, tensor<4xf32>)
        dimensions = [1]
        (%max_value: f32, %min_value: f32,
         %max_accumulator: f32, %min_accumulator: f32) {
      %next_max = arith.maximumf %max_value, %max_accumulator : f32
      %next_min = arith.minimumf %min_value, %min_accumulator : f32
      linalg.yield %next_max, %next_min : f32, f32
    }
    return %maximum, %minimum : tensor<4xf32>, tensor<4xf32>
  }

  func.func @unmatched_multi_operation_combiner(%input: tensor<4x8xf32>,
                                                %init: tensor<4xf32>)
      -> tensor<4xf32> {
    %two = arith.constant 2.0 : f32
    %sum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %sum = arith.addf %value, %accumulator : f32
        %next = arith.mulf %sum, %two : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func @nan_neutral_with_nnan(%input: tensor<4x8xf32>)
      -> tensor<4xf32> {
    %nan = arith.constant 0xFFC00000 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%nan : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.maxnumf %value, %accumulator fastmath<nnan> : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %maximum : tensor<4xf32>
  }

  func.func @nan_neutral_without_nnan(%input: tensor<4x8xf32>)
      -> tensor<4xf32> {
    %nan = arith.constant 0xFFC00000 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%nan : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%value: f32, %accumulator: f32):
        %next = arith.maxnumf %value, %accumulator : f32
        linalg.yield %next : f32
    } -> tensor<4xf32>
    return %maximum : tensor<4xf32>
  }

  func.func @vector_element_maximum(
      %input: tensor<4x8xvector<2xf32>>,
      %init: tensor<4xvector<2xf32>>) -> tensor<4xvector<2xf32>> {
    %maximum = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : tensor<4x8xvector<2xf32>>)
        outs(%init : tensor<4xvector<2xf32>>) {
      ^bb0(%value: vector<2xf32>, %accumulator: vector<2xf32>):
        %next = arith.maximumf %value, %accumulator : vector<2xf32>
        linalg.yield %next : vector<2xf32>
    } -> tensor<4xvector<2xf32>>
    return %maximum : tensor<4xvector<2xf32>>
  }

  // Pure fill and all-parallel payloads retain their existing metadata.
  func.func @parallel_and_fill(%input: tensor<4xf32>, %initial: f32)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %filled = linalg.fill ins(%initial : f32)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%filled : tensor<4xf32>) {
      ^bb0(%value: f32, %initial_value: f32):
        %added = arith.addf %value, %initial_value : f32
        linalg.yield %added : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // A captured write is observable even when the enclosing Linalg operation
  // has pure tensor operands and results.
  func.func @hidden_write(%input: tensor<4xf32>, %side: memref<1xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        memref.store %value, %side[%c0] : memref<1xf32>
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // Captured reads also prevent rematerialization: a later write could change
  // the value seen when replaying the payload.
  func.func @hidden_read(%input: tensor<4xf32>, %side: memref<1xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %captured = memref.load %side[%c0] : memref<1xf32>
        %added = arith.addf %value, %captured : f32
        linalg.yield %added : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @unknown_call(%input: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %called = func.call @unknown(%value) : (f32) -> f32
        linalg.yield %called : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
