// Regression test for the LLVM strict-properties compatibility patch
// (0005-llvm-strict-properties-compat): inherent attributes such as the
// argument attributes of tt.call must stay in the printed attr-dict so the
// textual IR round-trips identically. The equivalent test in Triton's own
// test/Triton/ops.mlir only runs in GPU builds (triton's test/ directory is
// excluded from CPU-only builds), so this copy guards the behavior in the
// triton-shared lit suite.

// RUN: triton-shared-opt %s | FileCheck %s
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: @call_attribute_roundtrip
tt.func @call_attribute_roundtrip(%arg0: f32) -> f32 {
  // CHECK: tt.call @attribute_callee(%arg0) {arg_attrs = [{tt.divisibility = 16 : i32}], res_attrs = [{}]} : (f32) -> f32
  %0 = tt.call @attribute_callee(%arg0) {arg_attrs = [{tt.divisibility = 16 : i32}], res_attrs = [{}]} : (f32) -> f32
  tt.return %0 : f32
}

tt.func private @attribute_callee(f32) -> f32
