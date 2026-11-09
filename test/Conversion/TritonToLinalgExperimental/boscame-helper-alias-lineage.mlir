//
// FileCheck lit test input 'boscame-helper-alias-lineage.mlir' (boscame helper
// alias lineage lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock boscame helper
// alias lineage lowering and contract facts.
//

module {
  func.func private @private_helper(%src: memref<4xf32>,
      %left: memref<4xf32>, %right: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 10 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 20 : i64}
    } : memref<4xf32> to memref<4xf32>
    // This read has a definite same-root writer regardless of aliases.
    memref.copy %left, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 30 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @public_helper(%src: memref<4xf32>,
      %left: memref<4xf32>, %right: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %left {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 40 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 50 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @unranked_helper(%src: memref<4xf32>,
      %left: memref<*xf32>, %right: memref<*xf32>, %dst: memref<4xf32>) {
    %left_view = memref.cast %left : memref<*xf32> to memref<4xf32>
    %right_view = memref.cast %right : memref<*xf32> to memref<4xf32>
    memref.copy %src, %left_view {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 60 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %right_view, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 70 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func @caller(%src: memref<4xf32>, %shared: memref<4xf32>,
      %dst: memref<4xf32>) {
    func.call @private_helper(%src, %shared, %shared, %dst)
      : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    func.call @public_helper(%src, %shared, %shared, %dst)
      : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %unranked = memref.cast %shared : memref<4xf32> to memref<*xf32>
    func.call @unranked_helper(%src, %unranked, %unranked, %dst)
      : (memref<4xf32>, memref<*xf32>, memref<*xf32>, memref<4xf32>) -> ()
    return
  }

  // No module-internal uses: the explicit entry ABI proves these parameters
  // disjoint, so existing complete entry graphs remain complete.
  func.func @entry(%src: memref<4xf32>, %dst: memref<4xf32>) {
    memref.copy %src, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 80 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @fresh_local_helper(%src: memref<4xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    memref.copy %src, %buffer {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 90 : i64}
    } : memref<4xf32> to memref<4xf32>
    memref.copy %buffer, %src {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 100 : i64}
    } : memref<4xf32> to memref<4xf32>
    return
  }

  func.func private @tensor_helper(%src: tensor<4xf32>,
      %dst: tensor<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%src : tensor<4xf32>) outs(%dst : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // A tensor read through to_tensor still reads its memref formal root.
  func.func private @tensor_bridge_helper(%src: memref<4xf32>,
      %dst: memref<4xf32>) -> tensor<4xf32> {
    %one = arith.constant 1.0 : f32
    linalg.fill ins(%one : f32) outs(%dst : memref<4xf32>)
    %snapshot = bufferization.to_tensor %src restrict
      : memref<4xf32> to tensor<4xf32>
    %empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
        iterator_types = ["parallel"]}
        ins(%snapshot : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
      } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  // Two potential aliases only matter to this gate when an access writes.
  func.func private @readonly_helper(%left: memref<4xf32>,
      %right: memref<4xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %lhs = memref.load %left[%c0] {
      contract.provenance = {origin_id = 110 : i64}
    } : memref<4xf32>
    %rhs = memref.load %right[%c0] {
      contract.provenance = {origin_id = 120 : i64}
    } : memref<4xf32>
    %sum = arith.addf %lhs, %rhs : f32
    return %sum : f32
  }

  // The entry ABI only provides a nonalias promise in the default space.
  func.func @custom_space_entry(%src: memref<4xf32, 3>,
      %dst: memref<4xf32, 3>) {
    memref.copy %src, %dst {
      contract.access = {kind = "store", boundary = "external_ddr", mask = {kind = "none"}, origin_id = 130 : i64}
    } : memref<4xf32, 3> to memref<4xf32, 3>
    return
  }
}
