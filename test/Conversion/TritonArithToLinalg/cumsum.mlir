// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

// @triton.jit
// def test_cumsum_op(
//     input_ptr, output_ptr, n_columns
// ):
//     row = tl.program_id(axis=0)
//     row_start = row * n_columns
//     columns = tl.arange(0, 4096)
//     offsets = row_start + columns
//     data = tl.load(input_ptr + offsets)
//     result = tl.cumsum(data, axis=0)
//     tl.store(output_ptr + offsets, result)
//
// ret = triton.compiler.compile(
//     test_cumsum_op,
//     signature=" *fp32,*i32,i32",
//     print_triton_ir_only=True,
// )
// print(ret.asm["ttir"])

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @test_cumsum_op_012(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4096 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG6]], %[[ARG2]] : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_1]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4096xi32>, tensor<4096xi32>) outs(%[[FILL_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[ARG0]] : !tt.ptr<f32>) outs(%[[EMPTY_2]] : tensor<4096x!tt.ptr<f32>>) -> tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_1]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs(%[[FILL_1]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_2]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[LOAD_0]]{{\[}}%[[CONSTANT_2]]] : tensor<4096xf32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[EXTRACT_0]] into %[[EMPTY_3]]{{\[}}%[[CONSTANT_2]]] : tensor<4096xf32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_8:.*]] = %[[EXTRACT_0]], %[[VAL_9:.*]] = %[[INSERT_0]]) -> (f32, tensor<4096xf32>) {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[LOAD_0]]{{\[}}%[[VAL_7]]] : tensor<4096xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_8]], %[[EXTRACT_1]] : f32
// CHECK:             %[[INSERT_1:.*]] = tensor.insert %[[ADDF_0]] into %[[VAL_9]]{{\[}}%[[VAL_7]]] : tensor<4096xf32>
// CHECK:             scf.yield %[[ADDF_0]], %[[INSERT_1]] : f32, tensor<4096xf32>
// CHECK:           }
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<4096x!tt.ptr<i32>>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[ARG1]] : !tt.ptr<i32>) outs(%[[EMPTY_4]] : tensor<4096x!tt.ptr<i32>>) -> tensor<4096x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_2]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>) outs(%[[FILL_2]] : tensor<4096x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: !tt.ptr<i32>, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_10]], %[[VAL_11]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<4096x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_13:.*]]#1 : tensor<4096xf32>) outs(%[[EMPTY_5]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: i32):
// CHECK:             %[[FPTOSI_0:.*]] = arith.fptosi %[[VAL_14]] : f32 to i32
// CHECK:             linalg.yield %[[FPTOSI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           tt.store %[[GENERIC_3]], %[[GENERIC_4]] : tensor<4096x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }
  tt.func public @test_cumsum_op_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8 = "tt.scan"(%7) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %12 = arith.addf %arg3, %arg4 : f32
      tt.scan.return %12 : f32
    }) : (tensor<4096xf32>) -> tensor<4096xf32>
    %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>>
    %10 = tt.addptr %9, %4 : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>
    %11 = arith.fptosi %8 : tensor<4096xf32> to tensor<4096xi32>
    tt.store %10, %11 : tensor<4096x!tt.ptr<i32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @scan_forward_rank1(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[CONSTANT_2]]] : tensor<4xf32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[EXTRACT_0]] into %[[EMPTY_0]]{{\[}}%[[CONSTANT_2]]] : tensor<4xf32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[EXTRACT_0]], %[[VAL_2:.*]] = %[[INSERT_0]]) -> (f32, tensor<4xf32>) {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]]] : tensor<4xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[EXTRACT_1]] : f32
// CHECK:             %[[INSERT_1:.*]] = tensor.insert %[[ADDF_0]] into %[[VAL_2]]{{\[}}%[[VAL_0]]] : tensor<4xf32>
// CHECK:             scf.yield %[[ADDF_0]], %[[INSERT_1]] : f32, tensor<4xf32>
// CHECK:           }
// CHECK:           return %[[VAL_3:.*]]#1 : tensor<4xf32>
// CHECK:         }
  tt.func @scan_forward_rank1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %1 : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @scan_reverse_rank1(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[CONSTANT_0]]] : tensor<4xf32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[EXTRACT_0]] into %[[EMPTY_0]]{{\[}}%[[CONSTANT_0]]] : tensor<4xf32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[EXTRACT_0]], %[[VAL_2:.*]] = %[[INSERT_0]]) -> (f32, tensor<4xf32>) {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[VAL_0]] : index
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[SUBI_1]]] : tensor<4xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[EXTRACT_1]] : f32
// CHECK:             %[[INSERT_1:.*]] = tensor.insert %[[ADDF_0]] into %[[VAL_2]]{{\[}}%[[SUBI_1]]] : tensor<4xf32>
// CHECK:             scf.yield %[[ADDF_0]], %[[INSERT_1]] : f32, tensor<4xf32>
// CHECK:           }
// CHECK:           return %[[VAL_3:.*]]#1 : tensor<4xf32>
// CHECK:         }
  tt.func @scan_reverse_rank1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %1 : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @scan_forward_rank2(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<2x4xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<2x4xf32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]], %[[CONSTANT_3]]] : tensor<2x4xf32>
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[EXTRACT_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]], %[[CONSTANT_3]]] : tensor<2x4xf32>
// CHECK:             %[[FOR_1:.*]]:2 = scf.for %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] iter_args(%[[VAL_3:.*]] = %[[EXTRACT_0]], %[[VAL_4:.*]] = %[[INSERT_0]]) -> (f32, tensor<2x4xf32>) {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]], %[[VAL_2]]] : tensor<2x4xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[EXTRACT_1]] : f32
// CHECK:               %[[INSERT_1:.*]] = tensor.insert %[[ADDF_0]] into %[[VAL_4]]{{\[}}%[[VAL_0]], %[[VAL_2]]] : tensor<2x4xf32>
// CHECK:               scf.yield %[[ADDF_0]], %[[INSERT_1]] : f32, tensor<2x4xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_5:.*]]#1 : tensor<2x4xf32>
// CHECK:           }
// CHECK:           return %[[FOR_0]] : tensor<2x4xf32>
// CHECK:         }
  tt.func @scan_forward_rank2(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "tt.scan"(%arg0) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %1 : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %0 : tensor<2x4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @scan_reverse_rank2_multi_result(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]], %[[VAL_2:.*]] = %[[EMPTY_1]]) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] : tensor<2x4xf32>
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] : tensor<2x4xf32>
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[EXTRACT_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] : tensor<2x4xf32>
// CHECK:             %[[INSERT_1:.*]] = tensor.insert %[[EXTRACT_1]] into %[[VAL_2]]{{\[}}%[[VAL_0]], %[[CONSTANT_0]]] : tensor<2x4xf32>
// CHECK:             %[[FOR_1:.*]]:4 = scf.for %[[VAL_3:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] iter_args(%[[VAL_4:.*]] = %[[EXTRACT_0]], %[[VAL_5:.*]] = %[[EXTRACT_1]], %[[VAL_6:.*]] = %[[INSERT_0]], %[[VAL_7:.*]] = %[[INSERT_1]]) -> (f32, f32, tensor<2x4xf32>, tensor<2x4xf32>) {
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[VAL_3]] : index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[CONSTANT_3]] : index
// CHECK:               %[[EXTRACT_2:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]], %[[SUBI_1]]] : tensor<2x4xf32>
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[VAL_0]], %[[SUBI_1]]] : tensor<2x4xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[EXTRACT_2]] : f32
// CHECK:               %[[MAXIMUMF_0:.*]] = arith.maximumf %[[VAL_5]], %[[EXTRACT_3]] : f32
// CHECK:               %[[INSERT_2:.*]] = tensor.insert %[[ADDF_0]] into %[[VAL_6]]{{\[}}%[[VAL_0]], %[[SUBI_1]]] : tensor<2x4xf32>
// CHECK:               %[[INSERT_3:.*]] = tensor.insert %[[MAXIMUMF_0]] into %[[VAL_7]]{{\[}}%[[VAL_0]], %[[SUBI_1]]] : tensor<2x4xf32>
// CHECK:               scf.yield %[[ADDF_0]], %[[MAXIMUMF_0]], %[[INSERT_2]], %[[INSERT_3]] : f32, f32, tensor<2x4xf32>, tensor<2x4xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_8:.*]]#2, %[[VAL_8]]#3 : tensor<2x4xf32>, tensor<2x4xf32>
// CHECK:           }
// CHECK:           return %[[VAL_9:.*]]#0, %[[VAL_9]]#1 : tensor<2x4xf32>, tensor<2x4xf32>
// CHECK:         }
  tt.func @scan_reverse_rank2_multi_result(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %0:2 = "tt.scan"(%arg0, %arg1) <{axis = 1 : i32, reverse = true}> ({
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
      %1 = arith.addf %arg2, %arg4 : f32
      %2 = arith.maximumf %arg3, %arg5 : f32
      tt.scan.return %1, %2 : f32, f32
    }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tt.return %0#0, %0#1 : tensor<2x4xf32>, tensor<2x4xf32>
  }
}
