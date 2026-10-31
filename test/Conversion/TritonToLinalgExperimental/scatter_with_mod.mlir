// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure scatter created when index has mod.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @scatter_with_mod(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xbf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xbf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi8> {maia.rank = 2 : i32, tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG19:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG20:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG21:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG22:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG23:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG24:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG25:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG26:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG27:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG28:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG29:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG30:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG31:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG32:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[TYPE_OFFSET_0:.*]] = tptr.type_offset i8  : i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 512 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 32 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 20 : i32
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG6]] : memref<*xi8> to memref<1xi8>
// CHECK:           %[[FROM_MEMREF_0:.*]] = tptr.from_memref %[[CAST_0]] : memref<1xi8> to <#tptr.default_memory_space>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<512x128xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_3]] : f32) outs(%[[EMPTY_0]] : tensor<512x128xf32>) -> tensor<512x128xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<512x1xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_4]] : i32) outs(%[[EMPTY_1]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG31]], %[[CONSTANT_5]] : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[CONSTANT_6]] : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[TYPE_OFFSET_0]] : i32
// CHECK:           %[[PTRADD_0:.*]] = tptr.ptradd %[[FROM_MEMREF_0]] %[[MULI_1]] : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
// CHECK:           %[[TO_MEMREF_0:.*]] = tptr.to_memref %[[PTRADD_0]] : <#tptr.default_memory_space> to memref<1xi32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[TO_MEMREF_0]] to offset: [0], sizes: [1], strides: [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<512xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [512, 1] : tensor<512xi32> into tensor<512x1xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_1]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[REMSI_0:.*]] = arith.remsi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[REMSI_0]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_1]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:             linalg.yield %[[DIVSI_0]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[LOAD_0]] : i32) outs(%[[EMPTY_1]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]], %[[FILL_2]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[GENERIC_2]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[ARG30]], %[[ARG19]] : i32
// CHECK:           %[[MULI_3:.*]] = arith.muli %[[ARG32]], %[[ARG21]] : i32
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[MULI_2]], %[[MULI_3]] : i32
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[ARG18]] : i32) outs(%[[EMPTY_1]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_3]], %[[FILL_3]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[GENERIC_3]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:             %[[MULI_4:.*]] = arith.muli %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             linalg.yield %[[MULI_4]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[EXTSI_0:.*]] = arith.extsi %[[ADDI_2]] : i32 to i64
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<512x1xi64>
// CHECK:           %[[FILL_4:.*]] = linalg.fill ins(%[[EXTSI_0]] : i64) outs(%[[EMPTY_3]] : tensor<512x1xi64>) -> tensor<512x1xi64>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_4]] : tensor<512x1xi32>) outs(%[[EMPTY_3]] : tensor<512x1xi64>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i64):
// CHECK:             %[[EXTSI_1:.*]] = arith.extsi %[[VAL_13]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_1]] : i64
// CHECK:           } -> tensor<512x1xi64>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[FILL_4]], %[[GENERIC_5]] : tensor<512x1xi64>, tensor<512x1xi64>) outs(%[[FILL_4]] : tensor<512x1xi64>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: i64, %[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: i64):
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:             linalg.yield %[[ADDI_3]] : i64
// CHECK:           } -> tensor<512x1xi64>
// CHECK:           %[[FILL_5:.*]] = linalg.fill ins(%[[ARG20]] : i32) outs(%[[EMPTY_1]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]], %[[FILL_5]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[GENERIC_1]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32):
// CHECK:             %[[MULI_5:.*]] = arith.muli %[[VAL_18]], %[[VAL_19]] : i32
// CHECK:             linalg.yield %[[MULI_5]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_7]] : tensor<512x1xi32>) outs(%[[EMPTY_3]] : tensor<512x1xi64>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i64):
// CHECK:             %[[EXTSI_2:.*]] = arith.extsi %[[VAL_21]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_2]] : i64
// CHECK:           } -> tensor<512x1xi64>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_6]], %[[GENERIC_8]] : tensor<512x1xi64>, tensor<512x1xi64>) outs(%[[GENERIC_6]] : tensor<512x1xi64>) {
// CHECK:           ^bb0(%[[VAL_23:.*]]: i64, %[[VAL_24:.*]]: i64, %[[VAL_25:.*]]: i64):
// CHECK:             %[[ADDI_4:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:             linalg.yield %[[ADDI_4]] : i64
// CHECK:           } -> tensor<512x1xi64>
// CHECK:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[GENERIC_9]] {{\[\[}}0, 1]] : tensor<512x1xi64> into tensor<512xi64>
// CHECK:           scf.for %[[VAL_26:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[COLLAPSE_SHAPE_0]]{{\[}}%[[VAL_26]]] : tensor<512xi64>
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_0]] : i64 to index
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[FILL_0]]{{\[}}%[[VAL_26]], 0] [1, 128] [1, 1] : tensor<512x128xf32> to tensor<1x128xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG5]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [1, 128], strides: [1, 1] : memref<*xf32> to memref<1x128xf32, strided<[1, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1x128xf32>, memref<1x128xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }



module attributes {maia.triton_kernel} {
  tt.func public @scatter_with_mod(%arg0: !tt.ptr<bf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %arg2: f32, %arg3: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<512x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c20_i32 = arith.constant 20 : i32
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<512x128xf32>

    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.addptr %arg6, %2 : !tt.ptr<i8>, i32
    %14 = tt.addptr %3, %c20_i32 : !tt.ptr<i8>, i32
    %15 = tt.bitcast %14 : !tt.ptr<i8> -> !tt.ptr<i32>
    %16 = tt.load %15 : !tt.ptr<i32>
    %94 = tt.get_program_id z : i32

    %97 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %99 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>


    %101 = tt.expand_dims %97 {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %102 = arith.remsi %101, %cst : tensor<512x1xi32>
    %103 = arith.divsi %101, %cst : tensor<512x1xi32>
    %104 = tt.splat %16 : i32 -> tensor<512x1xi32>
    %105 = arith.addi %103, %104 : tensor<512x1xi32>

    %115 = tt.expand_dims %99 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %117 = tt.broadcast %115 : tensor<1x128xi32> -> tensor<512x128xi32>

    %131 = arith.muli %0, %arg19 : i32
    %132 = arith.muli %94, %arg21 : i32
    %133 = arith.addi %131, %132 : i32
    %134 = tt.addptr %arg5, %133 : !tt.ptr<f32>, i32
    %135 = tt.splat %arg18 : i32 -> tensor<512x1xi32>
    %136 = arith.muli %105, %135 : tensor<512x1xi32>
    %137 = tt.splat %134 : !tt.ptr<f32> -> tensor<512x1x!tt.ptr<f32>>
    %138 = tt.addptr %137, %136 : tensor<512x1x!tt.ptr<f32>>, tensor<512x1xi32>
    %139 = tt.splat %arg20 : i32 -> tensor<512x1xi32>
    %140 = arith.muli %102, %139 : tensor<512x1xi32>
    %141 = tt.addptr %138, %140 : tensor<512x1x!tt.ptr<f32>>, tensor<512x1xi32>
    %142 = tt.broadcast %141 : tensor<512x1x!tt.ptr<f32>> -> tensor<512x128x!tt.ptr<f32>>
    %143 = tt.addptr %142, %117 : tensor<512x128x!tt.ptr<f32>>, tensor<512x128xi32>
    tt.store %143, %cst_6 : tensor<512x128x!tt.ptr<f32>>
    tt.return
  }
}
