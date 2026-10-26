// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: reverse_if_consumer = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
module {
  func.func @reverse_if_consumer(
      %input: tensor<4xf32>,
      %mask_input: tensor<4xi1> {
        contract.program = {role = "program_id", axis = 0 : i32}},
      %condition: i1, %upper: index, %offset: index)
      -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %empty = tensor.empty() : tensor<4xf32>
    %seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"],
        contract.provenance = {origin_id = 700 : i64}}
        ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<4xf32>
    %mask_empty = tensor.empty() : tensor<4xi1>
    %mask_seed = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%mask_input : tensor<4xi1>) outs(%mask_empty : tensor<4xi1>) {
      ^bb0(%value: i1, %unused: i1):
        %inverted = arith.xori %value, %true : i1
        linalg.yield %inverted : i1
    } -> tensor<4xi1>

    %loop:2 = scf.for %iv = %c0 to %upper step %c1
        iter_args(%acc = %seed, %mask_acc = %mask_seed)
        -> (tensor<4xf32>, tensor<4xi1>) {
      %body = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 701 : i64}
          ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      %mask_next = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%mask_acc : tensor<4xi1>) outs(%mask_empty : tensor<4xi1>) {
        ^bb0(%value: i1, %unused: i1):
          %inverted = arith.xori %value, %true : i1
          linalg.yield %inverted : i1
      } -> tensor<4xi1>
      // The mask backedge exercises the cycle guard. Its initial input
      // supplies the program axis through a compute-node operand.
      scf.yield %body, %mask_next : tensor<4xf32>, tensor<4xi1>
    }
    %slice = tensor.extract_slice %loop#0[%offset] [4] [1]
        : tensor<4xf32> to tensor<4xf32>

    %selected = scf.if %condition -> (tensor<4xf32>) {
      scf.yield %slice : tensor<4xf32>
    } else {
      %branch = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"], contract.origin_id = 702 : i64}
          ins(%slice : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
        ^bb0(%value: f32, %unused: f32):
          %negated = arith.negf %value : f32
          linalg.yield %negated : f32
      } -> tensor<4xf32>
      scf.yield %branch : tensor<4xf32>
    }

    %masked = arith.select %loop#1, %selected, %empty
        : tensor<4xi1>, tensor<4xf32>

    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"], contract.origin_id = 703 : i64}
        ins(%masked : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
