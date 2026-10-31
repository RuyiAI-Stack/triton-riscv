// RUN: triton-shared-opt --split-input-file --triton-to-unstructured --verify-each %s 2>&1 | FileCheck %s

module {
  tt.func public @dead_scalar_pointer_if_does_not_block_lowering(
      %base: !tt.ptr<f32>, %address: i64, %cond: i1) -> f32 {
    %from_int = tt.int_to_ptr %address : i64 -> !tt.ptr<f32>
    %selected = scf.if %cond -> (!tt.ptr<f32>) {
      scf.yield %from_int : !tt.ptr<f32>
    } else {
      scf.yield %base : !tt.ptr<f32>
    }

    %c1_i32 = arith.constant 1 : i32
    %shifted = tt.addptr %base, %c1_i32 : !tt.ptr<f32>, i32
    %value = tt.load %shifted : !tt.ptr<f32>
    tt.return %value : f32
  }
}

// CHECK-NOT:   warning:
// CHECK-LABEL: tt.func public @dead_scalar_pointer_if_does_not_block_lowering(
// CHECK-SAME:    %[[BASE:.*]]: !tt.ptr<f32>, %{{.*}}: i64, %{{.*}}: i1) -> f32 {
// CHECK-NOT:     tt.load
// CHECK:         %[[VALUE:.*]] = tts.gather %[[BASE]][%{{.*}}] : (<f32>, i32) -> f32
// CHECK:         tt.return %[[VALUE]] : f32

// -----

module {
  tt.func public @dead_derived_scalar_pointer_if_preserves_side_effects(
      %base: !tt.ptr<f32>, %then_out: !tt.ptr<f32>,
      %else_out: !tt.ptr<f32>, %offset: i32, %value: f32,
      %cond: i1) -> f32 {
    %selected = scf.if %cond -> (!tt.ptr<f32>) {
      %derived = tt.addptr %base, %offset : !tt.ptr<f32>, i32
      tt.store %then_out, %value : !tt.ptr<f32>
      scf.yield %derived : !tt.ptr<f32>
    } else {
      %derived = tt.addptr %base, %offset : !tt.ptr<f32>, i32
      tt.store %else_out, %value : !tt.ptr<f32>
      scf.yield %derived : !tt.ptr<f32>
    }

    %shifted = tt.addptr %base, %offset : !tt.ptr<f32>, i32
    %loaded = tt.load %shifted : !tt.ptr<f32>
    tt.return %loaded : f32
  }
}

// CHECK-NOT:   warning:
// CHECK-LABEL: tt.func public @dead_derived_scalar_pointer_if_preserves_side_effects(
// CHECK:         scf.if %{{.*}} {
// CHECK:           tts.scatter %{{.*}} into %{{.*}}[%{{.*}}] : f32 into (<f32>, i32)
// CHECK:         } else {
// CHECK:           tts.scatter %{{.*}} into %{{.*}}[%{{.*}}] : f32 into (<f32>, i32)
// CHECK:         }
// CHECK-NOT:     tt.load
// CHECK:         %[[LOADED:.*]] = tts.gather %{{.*}}[%{{.*}}] : (<f32>, i32) -> f32
// CHECK:         tt.return %[[LOADED]] : f32
