// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly or retain an incomplete graph. Complete,
// read-free private overwrites may recover local versions after an opaque
// boundary; they do not make its control-flow summary complete.
// CONTRACT-CASE: while_before_result = incomplete
// CONTRACT-CASE: while_memref_condition = incomplete
// CONTRACT-CASE: while_slot_permutation = incomplete
// CONTRACT-CASE: while_writer_exit = incomplete
module {
  func.func @while_before_result(%input: tensor<4xf32>, %condition: i1,
      %condition_buffer: memref<1xi1>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %loaded_condition = memref.load %condition_buffer[%c0] {
      contract.provenance = {origin_id = 30 : i64}
    } : memref<1xi1>
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 40 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>

    %result = scf.while (%before = %seed)
        : (tensor<4xf32>) -> tensor<4xf32> {
      %before_value = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 410 : i64}
          ins(%before : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.condition(%loaded_condition) %before_value : tensor<4xf32>
    } do {
    ^bb0(%after: tensor<4xf32>):
      %after_value = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 420 : i64}
          ins(%after : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %after_value : tensor<4xf32>
    }
    %result_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 430 : i64}
        ins(%result : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result_value : tensor<4xf32>
  }

  func.func @while_slot_permutation(%lhs: tensor<4xf32>,
      %rhs: tensor<4xf32>, %guard: tensor<4xf32>, %condition: i1)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %empty = tensor.empty() : tensor<4xf32>
    %lhs_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 210 : i64}}
        ins(%lhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %rhs_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 220 : i64}}
        ins(%rhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %guard_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 230 : i64}}
        ins(%guard : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>

    %result:3 = scf.while (%before_lhs = %lhs_values,
        %before_rhs = %rhs_values, %before_guard = %guard_values)
        : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
          -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
      scf.condition(%condition) %before_rhs, %before_lhs, %before_guard :
        tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    } do {
    ^bb0(%after_lhs: tensor<4xf32>, %after_rhs: tensor<4xf32>,
         %after_guard: tensor<4xf32>):
      %next_lhs = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 310 : i64}
          ins(%after_lhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next_rhs = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 320 : i64}
          ins(%after_rhs : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next_guard = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 330 : i64}
          ins(%after_guard : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %next_lhs, %next_rhs, %next_guard :
        tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    }

    %result_lhs = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 401 : i64}
        ins(%result#0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %result_rhs = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 402 : i64}
        ins(%result#1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %result_guard = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 403 : i64}
        ins(%result#2 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result_lhs, %result_rhs, %result_guard :
      tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  func.func @while_writer_exit(%initial: tensor<4xf32>,
      %before_input: tensor<4xf32>, %after_input: tensor<4xf32>,
      %condition: i1) -> (tensor<4xf32>, tensor<4xf32>) {
    %empty = tensor.empty() : tensor<4xf32>
    %initial_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 510 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %before_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 520 : i64}}
        ins(%before_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %after_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 530 : i64}}
        ins(%after_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %initial_buffer = bufferization.to_buffer %initial_values
      : tensor<4xf32> to memref<4xf32>
    %before_buffer = bufferization.to_buffer %before_values
      : tensor<4xf32> to memref<4xf32>
    %after_buffer = bufferization.to_buffer %after_values
      : tensor<4xf32> to memref<4xf32>
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial_buffer, %buffer {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 511 : i64}
    } : memref<4xf32> to memref<4xf32>

    %result = scf.while (%carry = %initial_values)
        : (tensor<4xf32>) -> tensor<4xf32> {
      memref.copy %before_buffer, %buffer {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 521 : i64}
      } : memref<4xf32> to memref<4xf32>
      %snapshot = bufferization.to_tensor %buffer restrict
        : memref<4xf32> to tensor<4xf32>
      scf.condition(%condition) %snapshot : tensor<4xf32>
    } do {
    ^bb0(%after: tensor<4xf32>):
      memref.copy %after_buffer, %buffer {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 531 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.yield %after : tensor<4xf32>
    }
    %result_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 541 : i64}
        ins(%result : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %post_snapshot = bufferization.to_tensor %buffer restrict
      : memref<4xf32> to tensor<4xf32>
    %post_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 542 : i64}
        ins(%post_snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result_value, %post_value : tensor<4xf32>, tensor<4xf32>
  }

  func.func @while_memref_condition(%dst: memref<4xf32>,
      %src: memref<4xf32>, %condition: i1) -> memref<4xf32> {
    %result = scf.while (%before = %dst)
        : (memref<4xf32>) -> memref<4xf32> {
      memref.copy %src, %before {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 610 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.condition(%condition) %before : memref<4xf32>
    } do {
    ^bb0(%after: memref<4xf32>):
      scf.yield %src : memref<4xf32>
    }
    return %result : memref<4xf32>
  }
}
