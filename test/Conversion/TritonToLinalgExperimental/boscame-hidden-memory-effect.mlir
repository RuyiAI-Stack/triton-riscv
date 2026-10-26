// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: hidden_memory_effect = reject: kernel contract V25: generic requires pure scalar arithmetic operations
module {
  memref.global "private" @side_effect : memref<16xf32> = uninitialized

  func.func @hidden_memory_effect(%src: memref<16xf32>,
                                 %dst: memref<16xf32>) {
    %zero = arith.constant 0 : index
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      %global = memref.get_global @side_effect : memref<16xf32>
      memref.store %value, %global[%zero] : memref<16xf32>
      linalg.yield %value : f32
    }
    return
  }
}
