// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly or retain an incomplete graph. Complete,
// read-free private overwrites may recover local versions after an opaque
// boundary; they do not make its control-flow summary complete.
// CONTRACT-CASE: loop_local_alloc_reset = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: loop_overwrite_snapshots = incomplete
module {
  // Tensor snapshots describe the buffer when to_tensor executes. Both are
  // consumed after a later overwrite: the first snapshot retains the loop
  // header sources, and the second retains only the middle full-root copy.
  // A separate partial-copy loop keeps sources for untouched elements.
  func.func @loop_overwrite_snapshots(%upper: index, %initial: tensor<4xf32>,
      %replacement: tensor<4xf32>, %late: tensor<4xf32>,
      %patch: tensor<2xf32>,
      %before_dst: memref<4xf32>, %after_dst: memref<4xf32>,
      %partial_dst: memref<4xf32>) {
    %empty = tensor.empty() : tensor<4xf32>
    %empty_patch = tensor.empty() : tensor<2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %initial_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 10 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %initial_buffer = bufferization.to_buffer %initial_values
      : tensor<4xf32> to memref<4xf32>
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial_buffer, %buffer {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 11 : i64}
    } : memref<4xf32> to memref<4xf32>
    %replacement_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 20 : i64}}
        ins(%replacement : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %replacement_buffer = bufferization.to_buffer %replacement_values
      : tensor<4xf32> to memref<4xf32>
    %late_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 25 : i64}}
        ins(%late : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %late_buffer = bufferization.to_buffer %late_values
      : tensor<4xf32> to memref<4xf32>
    scf.for %iv = %c0 to %upper step %c1 {
      %before = bufferization.to_tensor %buffer restrict
        : memref<4xf32> to tensor<4xf32>
      memref.copy %replacement_buffer, %buffer {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 21 : i64}
      } : memref<4xf32> to memref<4xf32>
      %after = bufferization.to_tensor %buffer restrict
        : memref<4xf32> to tensor<4xf32>
      memref.copy %late_buffer, %buffer {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 26 : i64}
      } : memref<4xf32> to memref<4xf32>

      %before_value = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 30 : i64}
          ins(%before : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %after_value = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 40 : i64}
          ins(%after : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %before_buffer = bufferization.to_buffer %before_value
        : tensor<4xf32> to memref<4xf32>
      memref.copy %before_buffer, %before_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 31 : i64}
      } : memref<4xf32> to memref<4xf32>
      %after_buffer = bufferization.to_buffer %after_value
        : tensor<4xf32> to memref<4xf32>
      memref.copy %after_buffer, %after_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 41 : i64}
      } : memref<4xf32> to memref<4xf32>
    }

    %partial_buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial_buffer, %partial_buffer {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 12 : i64}
    } : memref<4xf32> to memref<4xf32>
    %patch_values = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 50 : i64}}
        ins(%patch : tensor<2xf32>) outs(%empty_patch : tensor<2xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<2xf32>
    %patch_buffer = bufferization.to_buffer %patch_values
      : tensor<2xf32> to memref<2xf32>
    scf.for %iv = %c0 to %upper step %c1 {
      %partial_view = memref.subview %partial_buffer[0] [2] [1]
        : memref<4xf32> to memref<2xf32, strided<[1]>>
      memref.copy %patch_buffer, %partial_view {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 51 : i64}
      } : memref<2xf32> to memref<2xf32, strided<[1]>>
      %partial_snapshot = bufferization.to_tensor %partial_buffer restrict
        : memref<4xf32> to tensor<4xf32>
      %partial_value = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 60 : i64}
          ins(%partial_snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %partial_result_buffer = bufferization.to_buffer %partial_value
        : tensor<4xf32> to memref<4xf32>
      memref.copy %partial_result_buffer, %partial_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 61 : i64}
      } : memref<4xf32> to memref<4xf32>
    }
    return
  }

  // This metadata-only case reads fresh, uninitialized local storage before
  // writing it. A new allocation must have no source from a prior iteration,
  // even though the same alloc/alloca SSA operation is visited on loop replay.
  func.func @loop_local_alloc_reset(%upper: index,
      %initial: memref<4xf32>, %heap_dst: memref<4xf32>,
      %stack_dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %upper step %c1 {
      %heap = memref.alloc() : memref<4xf32>
      memref.copy %heap, %heap_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 70 : i64}
      } : memref<4xf32> to memref<4xf32>
      memref.copy %initial, %heap {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 80 : i64}
      } : memref<4xf32> to memref<4xf32>
      memref.dealloc %heap : memref<4xf32>

      %stack = memref.alloca() : memref<4xf32>
      memref.copy %stack, %stack_dst {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 90 : i64}
      } : memref<4xf32> to memref<4xf32>
      memref.copy %initial, %stack {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 100 : i64}
      } : memref<4xf32> to memref<4xf32>
    }
    return
  }
}
