// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure scatter created when index has mod.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @scatter_with_mod(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<*xbf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: memref<*xbf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %[[VAL_4:.*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %[[VAL_5:.*]]: memref<*xf32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %[[VAL_6:.*]]: memref<*xi8> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %[[VAL_7:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_8:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_9:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_10:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_11:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_12:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_13:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_14:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_15:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_16:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_17:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_18:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_19:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_20:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_21:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_22:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_23:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_24:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_25:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_26:.*]]: i32 {tt.divisibility = 16 : i32}, %[[VAL_27:.*]]: i32, %[[VAL_28:.*]]: i32, %[[VAL_29:.*]]: i32, %[[VAL_30:.*]]: i32, %[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: i32) {
// CHECK:           %[[BYTE_SCALE:.*]] = tptr.type_offset i8 : i32
// CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_34:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_35:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_36:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_37:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_38:.*]] = arith.constant 32 : i32
// CHECK:           %[[VAL_39:.*]] = arith.constant 20 : i32
// CHECK:           %[[BYTE_MEMREF:.*]] = memref.cast %[[VAL_6]] : memref<*xi8> to memref<1xi8>
// CHECK:           %[[BYTE_PTR:.*]] = tptr.from_memref %[[BYTE_MEMREF]] : memref<1xi8> to <#tptr.default_memory_space>
// CHECK:           %[[VAL_40:.*]] = tensor.empty() : tensor<512x128xf32>
// CHECK:           %[[VAL_41:.*]] = linalg.fill ins(%[[VAL_36]] : f32) outs(%[[VAL_40]] : tensor<512x128xf32>) -> tensor<512x128xf32>
// CHECK:           %[[VAL_42:.*]] = tensor.empty() : tensor<512x1xi32>
// CHECK:           %[[VAL_43:.*]] = linalg.fill ins(%[[VAL_37]] : i32) outs(%[[VAL_42]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[VAL_44:.*]] = arith.muli %[[VAL_31]], %[[VAL_38]] : i32
// CHECK:           %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_39]] : i32
// CHECK:           %[[BYTE_OFFSET:.*]] = arith.muli %[[VAL_45]], %[[BYTE_SCALE]] : i32
// CHECK:           %[[SHIFTED_PTR:.*]] = tptr.ptradd %[[BYTE_PTR]] %[[BYTE_OFFSET]] : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
// CHECK:           %[[I32_MEMREF:.*]] = tptr.to_memref %[[SHIFTED_PTR]] : <#tptr.default_memory_space> to memref<1xi32>
// CHECK:           %[[I32_VIEW:.*]] = memref.reinterpret_cast %[[I32_MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
// CHECK:           %[[VAL_49:.*]] = affine.load %[[I32_VIEW]][0] : memref<1xi32, strided<[1]>>
// CHECK:           %[[VAL_50:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:           %[[VAL_51:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_50]] : tensor<512xi32>) {
// CHECK:           ^bb0(%[[VAL_52:.*]]: i32):
// CHECK:             %[[VAL_53:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_54:.*]] = arith.index_cast %[[VAL_53]] : index to i32
// CHECK:             linalg.yield %[[VAL_54]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           %[[VAL_55:.*]] = tensor.expand_shape %[[VAL_51]] {{\[\[}}0, 1]] output_shape [512, 1] : tensor<512xi32> into tensor<512x1xi32>
// CHECK:           %[[VAL_56:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_55]], %[[VAL_43]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_55]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_57:.*]]: i32, %[[VAL_58:.*]]: i32, %[[VAL_59:.*]]: i32):
// CHECK:             %[[VAL_60:.*]] = arith.remsi %[[VAL_57]], %[[VAL_58]] : i32
// CHECK:             linalg.yield %[[VAL_60]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_61:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_55]], %[[VAL_43]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_55]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_62:.*]]: i32, %[[VAL_63:.*]]: i32, %[[VAL_64:.*]]: i32):
// CHECK:             %[[VAL_65:.*]] = arith.divsi %[[VAL_62]], %[[VAL_63]] : i32
// CHECK:             linalg.yield %[[VAL_65]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_66:.*]] = arith.muli %[[VAL_30]], %[[VAL_19]] : i32
// CHECK:           %[[VAL_67:.*]] = arith.muli %[[VAL_32]], %[[VAL_21]] : i32
// CHECK:           %[[VAL_68:.*]] = arith.addi %[[VAL_66]], %[[VAL_67]] : i32
// CHECK:           %[[VAL_69:.*]] = linalg.fill ins(%[[VAL_49]] : i32) outs(%[[VAL_42]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[VAL_70:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_61]], %[[VAL_69]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_61]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_71:.*]]: i32, %[[VAL_72:.*]]: i32, %[[VAL_73:.*]]: i32):
// CHECK:             %[[VAL_74:.*]] = arith.addi %[[VAL_71]], %[[VAL_72]] : i32
// CHECK:             linalg.yield %[[VAL_74]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_75:.*]] = linalg.fill ins(%[[VAL_18]] : i32) outs(%[[VAL_42]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[VAL_76:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_70]], %[[VAL_75]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_70]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_77:.*]]: i32, %[[VAL_78:.*]]: i32, %[[VAL_79:.*]]: i32):
// CHECK:             %[[VAL_80:.*]] = arith.muli %[[VAL_77]], %[[VAL_78]] : i32
// CHECK:             linalg.yield %[[VAL_80]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_81:.*]] = linalg.fill ins(%[[VAL_68]] : i32) outs(%[[VAL_42]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[VAL_82:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_81]], %[[VAL_76]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_81]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_83:.*]]: i32, %[[VAL_84:.*]]: i32, %[[VAL_85:.*]]: i32):
// CHECK:             %[[VAL_86:.*]] = arith.addi %[[VAL_83]], %[[VAL_84]] : i32
// CHECK:             linalg.yield %[[VAL_86]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_87:.*]] = linalg.fill ins(%[[VAL_20]] : i32) outs(%[[VAL_42]] : tensor<512x1xi32>) -> tensor<512x1xi32>
// CHECK:           %[[VAL_88:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_56]], %[[VAL_87]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_56]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_89:.*]]: i32, %[[VAL_90:.*]]: i32, %[[VAL_91:.*]]: i32):
// CHECK:             %[[VAL_92:.*]] = arith.muli %[[VAL_89]], %[[VAL_90]] : i32
// CHECK:             linalg.yield %[[VAL_92]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_93:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_82]], %[[VAL_88]] : tensor<512x1xi32>, tensor<512x1xi32>) outs(%[[VAL_82]] : tensor<512x1xi32>) {
// CHECK:           ^bb0(%[[VAL_94:.*]]: i32, %[[VAL_95:.*]]: i32, %[[VAL_96:.*]]: i32):
// CHECK:             %[[VAL_97:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : i32
// CHECK:             linalg.yield %[[VAL_97]] : i32
// CHECK:           } -> tensor<512x1xi32>
// CHECK:           %[[VAL_98:.*]] = tensor.collapse_shape %[[VAL_93]] {{\[\[}}0, 1]] : tensor<512x1xi32> into tensor<512xi32>
// CHECK:           scf.for %[[VAL_99:.*]] = %[[VAL_35]] to %[[VAL_34]] step %[[VAL_33]] {
// CHECK:             %[[VAL_100:.*]] = tensor.extract %[[VAL_98]]{{\[}}%[[VAL_99]]] : tensor<512xi32>
// CHECK:             %[[VAL_101:.*]] = arith.index_cast %[[VAL_100]] : i32 to index
// CHECK:             %[[VAL_102:.*]] = tensor.extract_slice %[[VAL_41]]{{\[}}%[[VAL_99]], 0] [1, 128] [1, 1] : tensor<512x128xf32> to tensor<1x128xf32>
// CHECK:             %[[VAL_103:.*]] = memref.reinterpret_cast %[[VAL_5]] to offset: {{\[}}%[[VAL_101]]], sizes: [1, 128], strides: [1, 1] : memref<*xf32> to memref<1x128xf32, strided<[1, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_102]] in writable %[[VAL_103]] : (tensor<1x128xf32>, memref<1x128xf32, strided<[1, 1], offset: ?>>) -> ()
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
