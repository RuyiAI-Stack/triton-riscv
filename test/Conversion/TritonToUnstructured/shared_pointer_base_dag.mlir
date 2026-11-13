// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s
// Each choice shares both prior choices. Walking paths instead of memoizing
// values makes this small, 40-node DAG take exponentially many visits.

module {
  tt.func public @shared_pointer_base_dag(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %c0: i1, %c1: i1, %c2: i1, %c3: i1, %c4: i1,
      %c5: i1, %c6: i1, %c7: i1, %c8: i1, %c9: i1,
      %c10: i1, %c11: i1, %c12: i1, %c13: i1, %c14: i1,
      %c15: i1, %c16: i1, %c17: i1, %c18: i1, %c19: i1,
      %c20: i1, %c21: i1, %c22: i1, %c23: i1, %c24: i1,
      %c25: i1, %c26: i1, %c27: i1, %c28: i1, %c29: i1,
      %c30: i1, %c31: i1, %c32: i1, %c33: i1, %c34: i1,
      %c35: i1, %c36: i1, %c37: i1, %c38: i1, %c39: i1) {
    %s0 = arith.select %c0, %a, %b : !tt.ptr<f32>
    %s1 = arith.select %c1, %s0, %b : !tt.ptr<f32>
    %s2 = arith.select %c2, %s1, %s0 : !tt.ptr<f32>
    %s3 = arith.select %c3, %s2, %s1 : !tt.ptr<f32>
    %s4 = arith.select %c4, %s3, %s2 : !tt.ptr<f32>
    %s5 = arith.select %c5, %s4, %s3 : !tt.ptr<f32>
    %s6 = arith.select %c6, %s5, %s4 : !tt.ptr<f32>
    %s7 = arith.select %c7, %s6, %s5 : !tt.ptr<f32>
    %s8 = arith.select %c8, %s7, %s6 : !tt.ptr<f32>
    %s9 = arith.select %c9, %s8, %s7 : !tt.ptr<f32>
    %s10 = arith.select %c10, %s9, %s8 : !tt.ptr<f32>
    %s11 = arith.select %c11, %s10, %s9 : !tt.ptr<f32>
    %s12 = arith.select %c12, %s11, %s10 : !tt.ptr<f32>
    %s13 = arith.select %c13, %s12, %s11 : !tt.ptr<f32>
    %s14 = arith.select %c14, %s13, %s12 : !tt.ptr<f32>
    %s15 = arith.select %c15, %s14, %s13 : !tt.ptr<f32>
    %s16 = arith.select %c16, %s15, %s14 : !tt.ptr<f32>
    %s17 = arith.select %c17, %s16, %s15 : !tt.ptr<f32>
    %s18 = arith.select %c18, %s17, %s16 : !tt.ptr<f32>
    %s19 = arith.select %c19, %s18, %s17 : !tt.ptr<f32>
    %s20 = arith.select %c20, %s19, %s18 : !tt.ptr<f32>
    %s21 = arith.select %c21, %s20, %s19 : !tt.ptr<f32>
    %s22 = arith.select %c22, %s21, %s20 : !tt.ptr<f32>
    %s23 = arith.select %c23, %s22, %s21 : !tt.ptr<f32>
    %s24 = arith.select %c24, %s23, %s22 : !tt.ptr<f32>
    %s25 = arith.select %c25, %s24, %s23 : !tt.ptr<f32>
    %s26 = arith.select %c26, %s25, %s24 : !tt.ptr<f32>
    %s27 = arith.select %c27, %s26, %s25 : !tt.ptr<f32>
    %s28 = arith.select %c28, %s27, %s26 : !tt.ptr<f32>
    %s29 = arith.select %c29, %s28, %s27 : !tt.ptr<f32>
    %s30 = arith.select %c30, %s29, %s28 : !tt.ptr<f32>
    %s31 = arith.select %c31, %s30, %s29 : !tt.ptr<f32>
    %s32 = arith.select %c32, %s31, %s30 : !tt.ptr<f32>
    %s33 = arith.select %c33, %s32, %s31 : !tt.ptr<f32>
    %s34 = arith.select %c34, %s33, %s32 : !tt.ptr<f32>
    %s35 = arith.select %c35, %s34, %s33 : !tt.ptr<f32>
    %s36 = arith.select %c36, %s35, %s34 : !tt.ptr<f32>
    %s37 = arith.select %c37, %s36, %s35 : !tt.ptr<f32>
    %s38 = arith.select %c38, %s37, %s36 : !tt.ptr<f32>
    %s39 = arith.select %c39, %s38, %s37 : !tt.ptr<f32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %s39 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %values = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    %output = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %outputs = tt.addptr %output, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %outputs, %values : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @shared_pointer_base_dag(
// CHECK-COUNT-39: arith.select
// CHECK: %[[BASE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<f32>
// CHECK: tts.gather %[[BASE]]
// CHECK: tts.scatter
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
