// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly or retain an incomplete graph. Complete,
// read-free private overwrites may recover local versions after an opaque
// boundary; they do not make its control-flow summary complete.
// CONTRACT-CASE: execute_region_loop_consumer = incomplete
// CONTRACT-CASE: execute_region_memref = incomplete
// CONTRACT-CASE: execute_region_program_axis = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: execute_region_tensor = incomplete
module {
  func.func @execute_region_tensor(%input: tensor<4xf32>) -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %forwarded = scf.execute_region -> tensor<4xf32> no_inline {
      %loaded = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"],
          contract.provenance = {origin_id = 10 : i64}}
          ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %loaded : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 20 : i64}
        ins(%forwarded : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @execute_region_memref(%source: memref<4xf32>,
      %destination: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    %forwarded = scf.execute_region -> memref<4xf32> no_inline {
      memref.copy %source, %buffer {
        contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 30 : i64}
      } : memref<4xf32> to memref<4xf32>
      scf.yield %buffer : memref<4xf32>
    }
    memref.copy %forwarded, %destination {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @execute_region_loop_consumer(%input: tensor<4xf32>, %upper: index)
      -> tensor<4xf32> {
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 50 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop_result = scf.for %iv = %c0 to %upper step %c1
        iter_args(%iter = %seed) -> (tensor<4xf32>) {
      %forwarded = scf.execute_region -> tensor<4xf32> no_inline {
        %inside = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"], contract.origin_id = 55 : i64}
            ins(%iter : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %negated = arith.negf %value : f32
            linalg.yield %negated : f32
        } -> tensor<4xf32>
        scf.yield %inside : tensor<4xf32>
      }
      scf.yield %forwarded : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 60 : i64}
        ins(%loop_result : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @execute_region_program_axis(
      %program: index {contract.program = {role = "program_id", axis = 2 : i32}}) {
    %one = arith.constant 1 : index
    %size = scf.execute_region -> index no_inline {
      %extent = arith.addi %program, %one : index
      scf.yield %extent : index
    }
    %source = memref.alloc(%size) : memref<?xf32>
    %destination = memref.alloc(%size) : memref<?xf32>
    memref.copy %source, %destination {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 70 : i64}
    } : memref<?xf32> to memref<?xf32>
    return
  }
}
