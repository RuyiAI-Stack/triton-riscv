// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: memref_recurrence = incomplete
// CONTRACT-CASE: tensor_slot_swap = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
module {
  // The read precedes the full-buffer update in the loop body. On a later
  // iteration it can observe the update from the previous iteration, while
  // the post-loop read must also retain the zero-trip initial contents.
  func.func @memref_recurrence(%condition: i1, %upper: index,
      %initial: tensor<4xf32>,
      %replacement: tensor<4xf32>, %observed_dst: memref<4xf32>,
      %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<4xf32>
    %initial_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 110 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %buffer = bufferization.to_buffer %initial_values
      : tensor<4xf32> to memref<4xf32>
    %replacement_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 130 : i64}}
        ins(%replacement : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %replacement_buffer = bufferization.to_buffer %replacement_values
      : tensor<4xf32> to memref<4xf32>

    scf.for %iv = %c0 to %upper step %c1 {
      %snapshot = bufferization.to_tensor %buffer restrict
        : memref<4xf32> to tensor<4xf32>
      %observed = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 120 : i64}
          ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %observed_buffer = bufferization.to_buffer %observed
        : tensor<4xf32> to memref<4xf32>
      memref.copy %observed_buffer, %observed_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 121 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.for %inner = %c0 to %upper step %c1 {
        scf.if %condition {
          memref.copy %replacement_buffer, %buffer {
            contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 131 : i64}
          } : memref<4xf32> to memref<4xf32>
        }
      }
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 140 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  // Slots 0 and 1 exchange values. Slot 2 carries an unrelated source and
  // must remain separate from both exchanged slots.
  func.func @tensor_slot_swap(%upper: index, %lhs: tensor<4xf32>,
      %rhs: tensor<4xf32>, %guard: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
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

    %result:3 = scf.for %iv = %c0 to %upper step %c1
        iter_args(%slot0 = %lhs_values, %slot1 = %rhs_values,
                  %slot2 = %guard_values)
        -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
      %next0 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 310 : i64}
          ins(%slot1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next1 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 320 : i64}
          ins(%slot0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %next2 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 330 : i64}
          ins(%slot2 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %next0, %next1, %next2 :
        tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
    }

      %consume0 = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 401 : i64}
        ins(%result#0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %consume1 = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 402 : i64}
        ins(%result#1 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %consume2 = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 403 : i64}
        ins(%result#2 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %consume0, %consume1, %consume2 :
      tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
