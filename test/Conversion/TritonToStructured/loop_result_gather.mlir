// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// Regression: tensor offsets produced by a loop and used after the loop for tt.addptr
// must stay as gather/scatter pointer state instead of collapsing to a contiguous
// tts.make_tptr with neutral offset/stride placeholders.

module {
  tt.func public @loop_result_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>, %arg3: !tt.ptr<i64>, %arg4: !tt.ptr<i64>) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<4xi64>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.extsi %0 : tensor<4xi32> to tensor<4xi64>
    %2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<4xi64>) : i32 {
      %3 = tt.addptr %arg2, %arg5 : !tt.ptr<i64>, i32
      %4 = tt.load %3 : !tt.ptr<i64>
      %5 = tt.addptr %arg3, %arg5 : !tt.ptr<i64>, i32
      %6 = tt.load %5 : !tt.ptr<i64>
      %7 = tt.addptr %arg4, %arg5 : !tt.ptr<i64>, i32
      %8 = tt.load %7 : !tt.ptr<i64>
      %9 = tt.splat %6 : i64 -> tensor<4xi64>
      %10 = arith.divsi %1, %9 : tensor<4xi64>
      %11 = tt.splat %4 : i64 -> tensor<4xi64>
      %12 = arith.remsi %10, %11 : tensor<4xi64>
      %13 = tt.splat %8 : i64 -> tensor<4xi64>
      %14 = arith.muli %12, %13 : tensor<4xi64>
      %15 = arith.addi %arg6, %14 : tensor<4xi64>
      scf.yield %15 : tensor<4xi64>
    }
    %16 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %17 = tt.addptr %16, %2 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %18 = tt.load %17 : tensor<4x!tt.ptr<f32>>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %20 = arith.extsi %0 : tensor<4xi32> to tensor<4xi64>
    %21 = tt.addptr %19, %20 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    tt.store %21, %18 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @loop_result_gather(
// CHECK: [[ARG0:%.+]]: !tt.ptr<f32>, [[ARG1:%.+]]: !tt.ptr<f32>
// CHECK: [[LOOP:%.+]] = scf.for
// CHECK: [[SRC:%.+]] = tts.make_gather_scatter_tptr [[ARG0]] to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: [[LOOP]]
// CHECK-NOT: tts.make_tptr [[ARG0]]
// CHECK: [[VAL:%.+]] = "tts.load"([[SRC]])
// CHECK: "tts.store"
