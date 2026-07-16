// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s --check-prefix=FULL

module {
  tt.func public @dynamic_pointer_base_selection(
      %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>,
      %out: !tt.ptr<f32>, %cond0: i1, %cond1: i1) {
    %0 = scf.if %cond0 -> (!tt.ptr<f32>) {
      scf.yield %arg0 : !tt.ptr<f32>
    } else {
      %1 = arith.select %cond1, %arg1, %arg2 : !tt.ptr<f32>
      scf.yield %1 : !tt.ptr<f32>
    }
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %5 = tt.load %4 : tensor<16x!tt.ptr<f32>>
    %6 = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %7 = tt.addptr %6, %2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %7, %5 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: warning: Cannot transform tensor of pointers
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK: %[[BASE:.*]] = scf.if %{{.*}} -> (!tt.ptr<f32>) {
// CHECK: arith.select %{{.*}}, %arg1, %arg2 : !tt.ptr<f32>
// CHECK: tts.gather %[[BASE]]
// CHECK: tts.scatter {{.*}} into %{{.*}}

// FULL-LABEL: func.func @dynamic_pointer_base_selection(
// FULL-NOT: !ptr.ptr
// FULL-NOT: tptr.
// FULL: %[[BASE:.*]] = scf.if %{{.*}} -> (memref<*xf32>) {
// FULL: scf.yield %arg0 : memref<*xf32>
// FULL: } else {
// FULL: arith.select %{{.*}}, %arg1, %arg2 : memref<*xf32>
// FULL: scf.yield %{{.*}} : memref<*xf32>
// FULL: memref.reinterpret_cast %[[BASE]]
// FULL-NOT: !ptr.ptr
// FULL-NOT: tptr.
