// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s
//
// Coverage for the TTIR dialect surface that the migration has to keep intact:
// tt.func with argument attributes, !tt.ptr address spaces, the load/store
// variants, elementwise shape ops, tt.dot with an input precision, the program
// id ops and call/return.  Everything is printed and re-parsed by the second
// triton-shared-opt in the RUN line.

module {
  // CHECK-LABEL: tt.func public @kernel
  // CHECK-SAME:  (%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f16, "constant">, %arg2: i32 {tt.divisibility = 16 : i32})
  // CHECK-SAME:  -> tensor<16x16xf32>
  tt.func public @kernel(%a: !tt.ptr<f32>, %b: !tt.ptr<f16, "constant">,
                         %n: i32 {tt.divisibility = 16 : i32}) -> tensor<16x16xf32> {
    // CHECK: %[[PID_X:.*]] = tt.get_program_id x : i32
    %pid = tt.get_program_id x : i32
    // CHECK: tt.get_program_id y : i32
    %pid_y = tt.get_program_id y : i32
    // CHECK: tt.get_num_programs z : i32
    %nprog = tt.get_num_programs z : i32
    // CHECK: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    // CHECK: tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %1 = tt.splat %a : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    // CHECK: %[[PTR:.*]] = tt.addptr {{.*}} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>

    // The historical spelling used by Triton-generated TTIR: the cache and
    // eviction enumerations arrive as plain i32 attributes.
    // CHECK: %[[LD0:.*]] = tt.load %[[PTR]] : tensor<16x!tt.ptr<f32>>
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x!tt.ptr<f32>>
    // The current Triton spelling prints the enumerations as keywords, and the
    // non-default attribute is printed in the attribute dictionary.
    // CHECK: tt.load %[[PTR]] cacheModifier = ca evictionPolicy = evict_first {isVolatile = true} : tensor<16x!tt.ptr<f32>>
    %4 = tt.load %2 {cache = 2 : i32, evict = 2 : i32, isVolatile = true} : tensor<16x!tt.ptr<f32>>
    // A scalar load.
    // CHECK: tt.load %arg0 : !tt.ptr<f32>
    %5 = tt.load %a : !tt.ptr<f32>

    // CHECK: %[[TRUE:.*]] = arith.constant true
    %true = arith.constant true
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    %cst = arith.constant 0.0 : f32
    // CHECK: tt.splat %[[TRUE]] : i1 -> tensor<16xi1>
    %mask = tt.splat %true : i1 -> tensor<16xi1>
    // CHECK: tt.splat %[[CST]] : f32 -> tensor<16xf32>
    %other = tt.splat %cst : f32 -> tensor<16xf32>
    // CHECK: %[[LD2:.*]] = tt.load %[[PTR]], {{.*}}, {{.*}} : tensor<16x!tt.ptr<f32>>
    %6 = tt.load %2, %mask, %other : tensor<16x!tt.ptr<f32>>

    // CHECK: tt.expand_dims %[[LD0]] {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %7 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    // CHECK: %[[BCAST:.*]] = tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x8xf32>
    %8 = tt.broadcast %7 : tensor<16x1xf32> -> tensor<16x8xf32>
    // CHECK: %[[TRANS:.*]] = tt.trans %[[BCAST]] {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
    %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<16x8xf32> -> tensor<8x16xf32>
    // CHECK: tt.reshape %[[BCAST]] : tensor<16x8xf32> -> tensor<128xf32>
    %10 = tt.reshape %8 : tensor<16x8xf32> -> tensor<128xf32>
    // CHECK: tt.bitcast %[[BCAST]] : tensor<16x8xf32> -> tensor<16x8xi32>
    %11 = tt.bitcast %8 : tensor<16x8xf32> -> tensor<16x8xi32>

    // CHECK: arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32>
    // CHECK: tt.dot %[[BCAST]], %[[TRANS]], {{.*}}, inputPrecision = tf32 : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    %12 = tt.dot %8, %9, %acc {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>

    // CHECK: tt.store %[[PTR]], %[[LD0]] : tensor<16x!tt.ptr<f32>>
    tt.store %2, %3 : tensor<16x!tt.ptr<f32>>
    // CHECK: tt.store %[[PTR]], %[[LD2]], {{.*}} : tensor<16x!tt.ptr<f32>>
    tt.store %2, %6, %mask : tensor<16x!tt.ptr<f32>>
    // CHECK: tt.return
    tt.return %12 : tensor<16x16xf32>
  }
}
