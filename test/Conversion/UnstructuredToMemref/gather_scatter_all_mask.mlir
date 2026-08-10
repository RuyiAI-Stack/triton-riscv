// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<9.900000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %cst_1 = arith.constant dense<64> : tensor<4xi32>
    %cst_2 = arith.constant dense<3> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_2 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
      %8 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %9 = tt.load %8, %7, %cst : tensor<4x!tt.ptr<f32>>
      %10 = tt.addptr %2, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %10, %9, %7 : tensor<4x!tt.ptr<f32>>
      %11 = arith.addi %6, %cst_0 : tensor<4xi32>
      %12 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      scf.yield %11, %12 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @masked_gather_scatter([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<3> : tensor<4xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<64> : tensor<4xi32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<4> : tensor<4xi32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_9_dot_900000_:%.+]] = arith.constant 9.900000e+01 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_2_]], [[VAR_arg4_:%.+]] = [[VAR_2_]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.divsi [[VAR_arg3_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_5_:%.+]] = tt.splat [[VAR_arg2_]] : i32 -> tensor<4xi32>
// CHECK:             [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[VAR_5_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[VAR_cst_0_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_cast_:%.+]] = memref.cast [[VAR_1_]] : memref<*xf32> to memref<?xf32>
// CHECK:             scf.for
// CHECK:               tensor.extract
// CHECK:               scf.if
// CHECK:                 memref.load
// CHECK:               tensor.insert
// CHECK:             [[VAR_cast_3_:%.+]] = memref.cast [[VAR_0_]] : memref<*xf32> to memref<?xf32>
// CHECK:             scf.for [[VAR_ARG5:%.+]] =
// CHECK:               [[VAR_OFFSET:%.+]] = tensor.extract [[VAR_6_]]{{.}}[[VAR_ARG5]]{{.}} : tensor<4xi32>
// CHECK:               [[VAR_INDEX:%.+]] = arith.index_cast [[VAR_OFFSET]] : i32 to index
// CHECK:               [[VAR_VALUE:%.+]] = tensor.extract {{.*}}{{.}}[[VAR_ARG5]]{{.}} : tensor<4xf32>
// CHECK:               [[VAR_MASK:%.+]] = tensor.extract [[VAR_7_]]{{.}}[[VAR_ARG5]]{{.}} : tensor<4xi1>
// CHECK:               scf.if [[VAR_MASK]] {
// CHECK:                 memref.store [[VAR_VALUE]], [[VAR_cast_3_]]{{.}}[[VAR_INDEX]]{{.}} : memref<?xf32>
// CHECK:               }
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_6_]], [[VAR_cst_1_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_arg4_]], [[VAR_cst_1_]] : tensor<4xi32>
// CHECK:             scf.yield [[VAR_11_]], [[VAR_12_]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
