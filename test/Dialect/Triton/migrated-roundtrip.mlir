// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s
//
// TTIR round trip through triton-shared-opt: the first invocation parses and verifies
// the input, prints it again; the second invocation re-parses the printed form.
// Any assembly format that cannot be read back therefore fails this test.

module {
  // CHECK-LABEL: tt.func public @kernel
  // CHECK-SAME:  (%arg0: !tt.ptr<f32>, %arg1: tensor<64x!tt.ptr<f32>>, %arg2: i32)
  tt.func public @kernel(%a: !tt.ptr<f32>, %b: tensor<64x!tt.ptr<f32>>, %n: i32) {
    // CHECK: tt.get_program_id x : i32
    %pid = tt.get_program_id x : i32
    // CHECK: tt.get_num_programs y : i32
    %nprog = tt.get_num_programs y : i32
    // CHECK: tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // CHECK: tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %1 = tt.splat %a : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    // CHECK: tt.addptr {{.*}} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>>
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x!tt.ptr<f32>>
    // CHECK: tt.load %arg1 : tensor<64x!tt.ptr<f32>>
    %4 = tt.load %b : tensor<64x!tt.ptr<f32>>
    // CHECK: arith.addf {{.*}} : tensor<64xf32>
    %5 = arith.addf %3, %4 : tensor<64xf32>
    // CHECK: tt.store {{.*}} : tensor<64x!tt.ptr<f32>>
    tt.store %2, %5 : tensor<64x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func private @callee
  tt.func private @callee(%x: f32) -> f32 {
    // CHECK: tt.return %arg0 : f32
    tt.return %x : f32
  }

  // CHECK-LABEL: tt.func @caller
  tt.func @caller(%x: f32) -> f32 {
    // CHECK: tt.call @callee({{.*}}) : (f32) -> f32
    %0 = tt.call @callee(%x) : (f32) -> f32
    tt.return %0 : f32
  }
}
