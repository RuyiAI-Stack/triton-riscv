// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @gather_simple_mask_with_other(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<-1.000000e+00> : tensor<64xf32>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<64> : tensor<64xi32>
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant dense<4> : tensor<64xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c8_i32, %arg4 = %0, %arg5 = %0) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg4, %cst_1 : tensor<64xi32>
      %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
      %7 = tt.addptr %1, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %8 = tt.load %7, %6, %cst : tensor<64x!tt.ptr<f32>>
      %9 = tt.addptr %2, %arg5 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %9, %8 : tensor<64x!tt.ptr<f32>>
      %10 = arith.addi %arg3, %c16_i32 : i32
      %11 = arith.addi %arg4, %cst_0 : tensor<64xi32>
      %12 = arith.addi %arg5, %cst_0 : tensor<64xi32>
      scf.yield %10, %11, %12 : i32, tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @gather_simple_mask_with_other(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<4> : tensor<64xi32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<64> : tensor<64xi32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_12:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:           %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[VAL_14:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[VAL_15:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:           %[[VAL_16:.*]]:3 = scf.for %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_9]] step %[[VAL_10]] iter_args(%[[VAL_18:.*]] = %[[VAL_5]], %[[VAL_19:.*]] = %[[VAL_15]], %[[VAL_20:.*]] = %[[VAL_15]]) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:             %[[VAL_21:.*]] = arith.divsi %[[VAL_19]], %[[VAL_6]] : tensor<64xi32>
// CHECK:             %[[VAL_22:.*]] = tt.splat %[[VAL_18]] : i32 -> tensor<64xi32>
// CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_22]] : tensor<64xi32>
// CHECK:             %[[VAL_24:.*]] = memref.cast %[[VAL_14]] : memref<*xf32> to memref<?xf32>
// CHECK:             %[[VAL_25:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:             %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]] iter_args(%[[VAL_28:.*]] = %[[VAL_25]]) -> (tensor<64xf32>) {
// CHECK:               %[[VAL_29:.*]] = tensor.extract %[[VAL_21]]{{\[}}%[[VAL_27]]] : tensor<64xi32>
// CHECK:               %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : i32 to index
// CHECK:               %[[VAL_31:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[VAL_27]]] : tensor<64xi1>
// CHECK:               %[[VAL_32:.*]] = scf.if %[[VAL_31]] -> (f32) {
// CHECK:                 %[[VAL_33:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_30]]] : memref<?xf32>
// CHECK:                 scf.yield %[[VAL_33]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_12]] : f32
// CHECK:               }
// CHECK:               %[[VAL_34:.*]] = tensor.insert %[[VAL_32]] into %[[VAL_28]]{{\[}}%[[VAL_27]]] : tensor<64xf32>
// CHECK:               scf.yield %[[VAL_34]] : tensor<64xf32>
// CHECK:             }
// CHECK:             %[[VAL_35:.*]] = memref.cast %[[VAL_13]] : memref<*xf32> to memref<?xf32>
// CHECK:             scf.for %[[VAL_36:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]] {
// CHECK:               %[[VAL_37:.*]] = tensor.extract %[[VAL_20]]{{\[}}%[[VAL_36]]] : tensor<64xi32>
// CHECK:               %[[VAL_38:.*]] = arith.index_cast %[[VAL_37]] : i32 to index
// CHECK:               %[[VAL_39:.*]] = tensor.extract %[[VAL_26]]{{\[}}%[[VAL_36]]] : tensor<64xf32>
// CHECK:               memref.store %[[VAL_39]], %[[VAL_35]]{{\[}}%[[VAL_38]]] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_18]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_19]], %[[VAL_8]] : tensor<64xi32>
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_20]], %[[VAL_8]] : tensor<64xi32>
// CHECK:             scf.yield %[[VAL_40]], %[[VAL_41]], %[[VAL_42]] : i32, tensor<64xi32>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
