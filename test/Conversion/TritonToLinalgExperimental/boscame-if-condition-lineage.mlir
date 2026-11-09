//
// FileCheck lit test input 'boscame-if-condition-lineage.mlir' (boscame if
// condition lineage lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock boscame if
// condition lineage lowering and contract facts.
//

module {
  func.func @if_condition_is_a_data_dependency(
      %condition_buffer: memref<1xi1>, %side_effect: memref<1xi1>,
      %then_input: tensor<4xf32>, %else_input: tensor<4xf32>)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<4xf32>
    %condition = memref.load %condition_buffer[%c0] {
      contract.provenance = {origin_id = 80 : i64}
    } : memref<1xi1>
    %then_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 81 : i64}}
        ins(%then_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %else_value = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 82 : i64}}
        ins(%else_input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    %selected = scf.if %condition -> (tensor<4xf32>) {
      memref.store %condition, %side_effect[%c0] : memref<1xi1>
      scf.yield %then_value : tensor<4xf32>
    } else {
      scf.yield %else_value : tensor<4xf32>
    }
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 83 : i64}
        ins(%selected : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
