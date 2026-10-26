// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly or retain an incomplete graph. Complete,
// read-free private overwrites may recover local versions after an opaque
// boundary; they do not make its control-flow summary complete.
// CONTRACT-CASE: for_memref_slot_divergent_roots = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: for_memref_slot_same_root = complete
// CONTRACT-CASE: for_zero_trip = incomplete
// CONTRACT-CASE: if_else_writers = incomplete
// CONTRACT-CASE: if_memref_divergent_roots = reject: kernel contract V23/V24: v1 does not support reading an external region after its terminal write
// CONTRACT-CASE: if_memref_same_root = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: if_without_else = reject: kernel contract V17: v1 if requires explicit then and else scopes
// CONTRACT-CASE: partial_view_writers = incomplete
// CONTRACT-CASE: scalar_partial_writer = reject: kernel contract V22/V25: v1 private scalar stores require rank-zero storage
// CONTRACT-CASE: switch_memref_divergent_roots = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: switch_memref_same_root = incomplete
// CONTRACT-CASE: tensor_before_overwrite = complete
// CONTRACT-CASE: tensor_buffer_full_overwrite = complete
// CONTRACT-CASE: tensor_buffer_partial_overwrite = reject: kernel contract V22/V25: v1 private scalar stores require rank-zero storage
// CONTRACT-CASE: tensor_buffer_roundtrip = complete
// CONTRACT-CASE: while_memref_slot_divergent_roots = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: while_memref_slot_same_root = complete
// Structured side effects have no tensor result through which a later read
// can discover both paths. Reaching writers must merge at the region exit.
module {
  func.func @if_else_writers(%condition: i1, %initial: memref<4xf32>,
      %left: memref<4xf32>, %right: memref<4xf32>, %dst: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 10 : i64}
    } : memref<4xf32> to memref<4xf32>
    scf.if %condition {
      memref.copy %left, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 20 : i64}
      } : memref<4xf32> to memref<4xf32>
    } else {
      memref.copy %right, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 30 : i64}
      } : memref<4xf32> to memref<4xf32>
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @if_without_else(%condition: i1, %initial: memref<4xf32>,
      %updated: memref<4xf32>, %dst: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 50 : i64}
    } : memref<4xf32> to memref<4xf32>
    scf.if %condition {
      memref.copy %updated, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 60 : i64}
      } : memref<4xf32> to memref<4xf32>
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 70 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @for_zero_trip(%upper: index, %initial: memref<4xf32>,
      %updated: memref<4xf32>, %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 80 : i64}
    } : memref<4xf32> to memref<4xf32>
    scf.for %iv = %c0 to %upper step %c1 {
      memref.copy %updated, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 90 : i64}
      } : memref<4xf32> to memref<4xf32>
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 100 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  // A partial write cannot kill the incoming full-buffer definition, even
  // when both branches write the same subview.
  func.func @partial_view_writers(%condition: i1, %initial: memref<4xf32>,
      %left: memref<2xf32>, %right: memref<2xf32>, %dst: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    %prefix = memref.subview %buffer[0] [2] [1]
      : memref<4xf32> to memref<2xf32, strided<[1]>>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 110 : i64}
    } : memref<4xf32> to memref<4xf32>
    scf.if %condition {
      memref.copy %left, %prefix {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 120 : i64}
      } : memref<2xf32> to memref<2xf32, strided<[1]>>
    } else {
      memref.copy %right, %prefix {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 130 : i64}
      } : memref<2xf32> to memref<2xf32, strided<[1]>>
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 140 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  // A scalar store targets the root SSA value but does not overwrite every
  // element. The following whole-buffer read must retain the old source.
  func.func @scalar_partial_writer(%condition: i1, %initial: memref<4xf32>,
      %updated: memref<1xf32>, %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 150 : i64}
    } : memref<4xf32> to memref<4xf32>
    scf.if %condition {
      %value = memref.load %updated[%c0] {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 160 : i64}
      } : memref<1xf32>
      memref.store %value, %buffer[%c0] {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 170 : i64}
      } : memref<4xf32>
    } else {
      %value = memref.load %updated[%c0] {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 180 : i64}
      } : memref<1xf32>
      memref.store %value, %buffer[%c0] {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 190 : i64}
      } : memref<4xf32>
    }
    memref.copy %buffer, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 200 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @tensor_before_overwrite(%initial: memref<4xf32>,
      %updated: memref<4xf32>, %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 210 : i64}
    } : memref<4xf32> to memref<4xf32>
    %snapshot = bufferization.to_tensor %buffer restrict
      : memref<4xf32> to tensor<4xf32>
    memref.copy %updated, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 220 : i64}
    } : memref<4xf32> to memref<4xf32>
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 230 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @tensor_buffer_roundtrip(%initial: tensor<4xf32>,
      %keep: memref<4xf32>, %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %loaded = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 240 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %buffer = bufferization.to_buffer %loaded
      : tensor<4xf32> to memref<4xf32>
    // Keep the bridge pair separated so canonicalization cannot fold it.
    memref.copy %buffer, %keep {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 241 : i64}
    } : memref<4xf32> to memref<4xf32>
    %snapshot = bufferization.to_tensor %buffer restrict
      : memref<4xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 250 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @tensor_buffer_full_overwrite(%initial: tensor<4xf32>,
      %updated: memref<4xf32>, %raw_dst: memref<4xf32>,
      %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %loaded = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 260 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %buffer = bufferization.to_buffer %loaded
      : tensor<4xf32> to memref<4xf32>
    memref.copy %updated, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 270 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %buffer, %raw_dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 271 : i64}
    } : memref<4xf32> to memref<4xf32>
    %snapshot = bufferization.to_tensor %buffer restrict
      : memref<4xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 280 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @tensor_buffer_partial_overwrite(%initial: tensor<4xf32>,
      %updated: memref<1xf32>, %raw_dst: memref<4xf32>,
      %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %loaded = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 290 : i64}}
        ins(%initial : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %buffer = bufferization.to_buffer %loaded
      : tensor<4xf32> to memref<4xf32>
    %replacement = memref.load %updated[%c0] {
      contract.provenance = {origin_id = 300 : i64}
    } : memref<1xf32>
    memref.store %replacement, %buffer[%c0] {
      contract.provenance = {origin_id = 310 : i64}
    } : memref<4xf32>
    memref.copy %buffer, %raw_dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 311 : i64}
    } : memref<4xf32> to memref<4xf32>
    %snapshot = bufferization.to_tensor %buffer restrict
      : memref<4xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 320 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @if_memref_same_root(%condition: i1, %initial: memref<8xf32>,
      %updated: memref<8xf32>, %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %buffer = memref.alloc() : memref<8xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 330 : i64}
    } : memref<8xf32> to memref<8xf32>
    %left_base = memref.subview %buffer[0] [4] [1]
      : memref<8xf32> to memref<4xf32, strided<[1]>>
    %right_base = memref.subview %buffer[4] [4] [1]
      : memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
    %left = memref.cast %left_base
      : memref<4xf32, strided<[1]>>
        to memref<4xf32, strided<[1], offset: ?>>
    %right = memref.cast %right_base
      : memref<4xf32, strided<[1], offset: 4>>
        to memref<4xf32, strided<[1], offset: ?>>
    %alias = scf.if %condition -> memref<4xf32, strided<[1], offset: ?>> {
      memref.copy %initial, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 331 : i64}
      } : memref<8xf32> to memref<8xf32>
      scf.yield %left : memref<4xf32, strided<[1], offset: ?>>
    } else {
      memref.copy %updated, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 332 : i64}
      } : memref<8xf32> to memref<8xf32>
      scf.yield %right : memref<4xf32, strided<[1], offset: ?>>
    }
    %snapshot = bufferization.to_tensor %alias restrict
      : memref<4xf32, strided<[1], offset: ?>> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 340 : i64}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 350 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @if_memref_divergent_roots(%condition: i1,
      %initial: memref<4xf32>, %updated: memref<4xf32>,
      %scratch: memref<4xf32>, %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %left = memref.alloc() : memref<4xf32>
    %right = memref.alloc() : memref<4xf32>
    memref.copy %initial, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 360 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %updated, %right {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 370 : i64}
    } : memref<4xf32> to memref<4xf32>
    %alias = scf.if %condition -> memref<4xf32> {
      memref.copy %left, %scratch {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 371 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.yield %left : memref<4xf32>
    } else {
      memref.copy %right, %scratch {
        contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 372 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.yield %right : memref<4xf32>
    }
    %snapshot = bufferization.to_tensor %alias restrict
      : memref<4xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 380 : i64}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 390 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @switch_memref_same_root(%selector: index,
      %initial: memref<8xf32>, %updated: memref<8xf32>,
      %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %buffer = memref.alloc() : memref<8xf32>
    %left_base = memref.subview %buffer[0] [4] [1]
      : memref<8xf32> to memref<4xf32, strided<[1]>>
    %right_base = memref.subview %buffer[4] [4] [1]
      : memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
    %left = memref.cast %left_base
      : memref<4xf32, strided<[1]>>
        to memref<4xf32, strided<[1], offset: ?>>
    %right = memref.cast %right_base
      : memref<4xf32, strided<[1], offset: 4>>
        to memref<4xf32, strided<[1], offset: ?>>
    %alias = scf.index_switch %selector
        -> memref<4xf32, strided<[1], offset: ?>>
    case 0 {
      memref.copy %initial, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 401 : i64}
      } : memref<8xf32> to memref<8xf32>
      scf.yield %left : memref<4xf32, strided<[1], offset: ?>>
    }
    default {
      memref.copy %updated, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 402 : i64}
      } : memref<8xf32> to memref<8xf32>
      scf.yield %right : memref<4xf32, strided<[1], offset: ?>>
    }
    %snapshot = bufferization.to_tensor %alias restrict
      : memref<4xf32, strided<[1], offset: ?>> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 403 : i64}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 404 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @switch_memref_divergent_roots(%selector: index,
      %initial: memref<4xf32>, %updated: memref<4xf32>,
      %dst: memref<4xf32>) {
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %left = memref.alloc() : memref<4xf32>
    %right = memref.alloc() : memref<4xf32>
    memref.copy %initial, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 411 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %updated, %right {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 412 : i64}
    } : memref<4xf32> to memref<4xf32>
    %alias = scf.index_switch %selector -> memref<4xf32>
    case 0 {
      scf.yield %left : memref<4xf32>
    }
    default {
      scf.yield %right : memref<4xf32>
    }
    %snapshot = bufferization.to_tensor %alias restrict
      : memref<4xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 413 : i64}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    %output = bufferization.to_buffer %result
      : tensor<4xf32> to memref<4xf32>
    memref.copy %output, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 414 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @for_memref_slot_same_root(%upper: index,
      %initial: memref<4xf32>, %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 501 : i64}
    } : memref<4xf32> to memref<4xf32>
    %result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%slot = %buffer) -> memref<4xf32> {
      scf.yield %slot : memref<4xf32>
    }
    memref.copy %result, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 502 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @for_memref_slot_divergent_roots(%upper: index,
      %initial: memref<4xf32>, %updated: memref<4xf32>,
      %dst: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %left = memref.alloc() : memref<4xf32>
    %right = memref.alloc() : memref<4xf32>
    memref.copy %initial, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 511 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %updated, %right {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 512 : i64}
    } : memref<4xf32> to memref<4xf32>
    %result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%slot = %left) -> memref<4xf32> {
      scf.yield %right : memref<4xf32>
    }
    memref.copy %result, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 513 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @while_memref_slot_same_root(%condition: i1,
      %initial: memref<4xf32>, %dst: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %initial, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 521 : i64}
    } : memref<4xf32> to memref<4xf32>
    %result = scf.while (%before = %buffer)
        : (memref<4xf32>) -> memref<4xf32> {
      scf.condition(%condition) %before : memref<4xf32>
    } do {
    ^bb0(%after: memref<4xf32>):
      scf.yield %after : memref<4xf32>
    }
    memref.copy %result, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 522 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @while_memref_slot_divergent_roots(%condition: i1,
      %initial: memref<4xf32>, %updated: memref<4xf32>,
      %dst: memref<4xf32>) {
    %left = memref.alloc() : memref<4xf32>
    %right = memref.alloc() : memref<4xf32>
    memref.copy %initial, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 531 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %updated, %right {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 532 : i64}
    } : memref<4xf32> to memref<4xf32>
    %result = scf.while (%before = %left)
        : (memref<4xf32>) -> memref<4xf32> {
      scf.condition(%condition) %before : memref<4xf32>
    } do {
    ^bb0(%after: memref<4xf32>):
      scf.yield %right : memref<4xf32>
    }
    memref.copy %result, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 533 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }
}
