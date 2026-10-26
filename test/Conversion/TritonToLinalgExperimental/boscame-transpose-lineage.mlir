// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: transpose_exit_consumer = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
// CONTRACT-CASE: transpose_memref_writer = reject: kernel contract V23/V24: v1 does not support reading an external region after its terminal write
module {
  func.func @transpose_memref_writer(
      %source: memref<4x4xf32>, %buffer: memref<4x4xf32>,
      %destination: memref<4x4xf32>) {
    %transposed = memref.transpose %buffer (i, j) -> (j, i)
      : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %source, %transposed {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 101 : i64}
    } : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %buffer, %destination {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 102 : i64}
    } : memref<4x4xf32> to memref<4x4xf32>
    return
  }

  func.func @transpose_exit_consumer(%input: memref<4x4xf32>,
      %other: memref<4x4xf32>, %destination: memref<4x4xf32>,
      %condition: i1, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %iv = %c0 to %upper step %c1
        iter_args(%iter = %input) -> (memref<4x4xf32>) {
      %selected = scf.if %condition -> (memref<4x4xf32>) {
        scf.yield %iter : memref<4x4xf32>
      } else {
        scf.yield %other : memref<4x4xf32>
      }
      scf.yield %selected : memref<4x4xf32>
    }
    %transposed = memref.transpose %loop (i, j) -> (j, i)
      : memref<4x4xf32> to
        memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
    memref.copy %transposed, %destination {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 103 : i64}
    } : memref<4x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 4)>>
        to memref<4x4xf32>
    return
  }
}
