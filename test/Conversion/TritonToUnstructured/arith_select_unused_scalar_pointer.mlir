// RUN: triton-shared-opt --triton-to-unstructured --verify-each %s 2>&1 | FileCheck %s

module {
  tt.func public @dead_scalar_pointer_select_does_not_block_lowering(
      %base: !tt.ptr<f32>, %address: i64, %cond: i1) -> f32 {
    %from_int = tt.int_to_ptr %address : i64 -> !tt.ptr<f32>
    %selected = arith.select %cond, %from_int, %base : !tt.ptr<f32>

    %c1_i32 = arith.constant 1 : i32
    %shifted = tt.addptr %base, %c1_i32 : !tt.ptr<f32>, i32
    %value = tt.load %shifted : !tt.ptr<f32>
    tt.return %value : f32
  }
}

// CHECK-NOT:   warning:
// CHECK-LABEL: tt.func public @dead_scalar_pointer_select_does_not_block_lowering(
// CHECK-NOT:     arith.select
// CHECK-NOT:     tt.load
// CHECK:         %[[VALUE:.*]] = tts.gather %{{.*}}[%{{.*}}] : (<f32>, i32) -> f32
// CHECK:         tt.return %[[VALUE]] : f32
