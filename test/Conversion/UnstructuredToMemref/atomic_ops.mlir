// RUN: triton-shared-opt --unstructured-to-memref %s | FileCheck %s
// RUN: triton-shared-opt --unstructured-to-memref %s | triton-shared-opt --lower-atomic-cas-to-llvm | FileCheck %s --check-prefix=LLVM

module {
  tt.func public @atomic_add_scalar_masked(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i1) -> i32 {
    %0 = tts.atomic_rmw add, relaxed, gpu, %arg2 into %arg0[%arg1] mask = %arg3 : i32 into (!tt.ptr<i32>, i32) -> i32
    tt.return %0 : i32
  }

  tt.func public @atomic_cas_tensor(%arg0: !tt.ptr<i32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>) -> tensor<2xi32> {
    %0 = tts.atomic_cas acq_rel, gpu, %arg2, %arg3 into %arg0[%arg1] : tensor<2xi32>, tensor<2xi32> into (!tt.ptr<i32>, tensor<2xi32>) -> tensor<2xi32>
    tt.return %0 : tensor<2xi32>
  }

  tt.func public @atomic_cas_float(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: f32, %arg3: f32) -> f32 {
    %0 = tts.atomic_cas acq_rel, gpu, %arg2, %arg3 into %arg0[%arg1] : f32, f32 into (!tt.ptr<f32>, i32) -> f32
    tt.return %0 : f32
  }

  tt.func public @atomic_add_tensor_dynamic(%arg0: !tt.ptr<i32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tts.atomic_rmw add, relaxed, gpu, %arg2 into %arg0[%arg1] : tensor<?xi32> into (!tt.ptr<i32>, tensor<?xi32>) -> tensor<?xi32>
    tt.return %0 : tensor<?xi32>
  }

  tt.func public @atomic_cas_tensor_dynamic(%arg0: !tt.ptr<i32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tts.atomic_cas acq_rel, gpu, %arg2, %arg3 into %arg0[%arg1] : tensor<?xi32>, tensor<?xi32> into (!tt.ptr<i32>, tensor<?xi32>) -> tensor<?xi32>
    tt.return %0 : tensor<?xi32>
  }
}

// CHECK-LABEL:   tt.func public @atomic_add_scalar_masked(
// CHECK:           %cast = memref.cast %0 : memref<*xi32> to memref<?xi32>
// CHECK:           %1 = arith.index_cast %arg1 : i32 to index
// CHECK:           %c0_i32 = arith.constant 0 : i32
// CHECK:           %2 = scf.if %arg3 -> (i32) {
// CHECK:             %3 = memref.atomic_rmw addi %arg2, %cast[%1] : (i32, memref<?xi32>) -> i32
// CHECK:             scf.yield %3 : i32
// CHECK:           } else {
// CHECK:             scf.yield %c0_i32 : i32
// CHECK:           }
// CHECK:           tt.return %2 : i32

// CHECK-LABEL:   tt.func public @atomic_cas_tensor(
// CHECK:           %cast = memref.cast %0 : memref<*xi32> to memref<?xi32>
// CHECK:           %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %1) -> (tensor<2xi32>) {
// CHECK:           %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %cast : memref<?xi32> -> index
// CHECK:           %[[OLD:.*]] = func.call @__triton_shared_atomic_cas_acq_rel(%{{.*}}, %{{.*}}, %{{.*}}) : (index, i32, i32) -> i32
// CHECK:           tt.return

// CHECK-LABEL:   tt.func public @atomic_cas_float(
// CHECK:           %[[CAST:.*]] = memref.cast {{.*}} : memref<*xf32> to memref<?xf32>
// CHECK:           %[[INDEX:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK:           %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %[[CAST]] : memref<?xf32> -> index
// CHECK:           %[[OLD:.*]] = func.call @__triton_shared_atomic_cas_acq_rel_1(%{{.*}}, %arg2, %arg3) : (index, f32, f32) -> f32
// CHECK:           tt.return %[[OLD]] : f32

// LLVM-LABEL:   tt.func public @atomic_cas_tensor(
// LLVM:           %[[PTR:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
// LLVM:           %[[CAS:.*]] = llvm.cmpxchg %[[PTR]], {{.*}}, {{.*}} acq_rel acquire : !llvm.ptr, i32
// LLVM:           %[[OLD:.*]] = llvm.extractvalue %[[CAS]][0] : !llvm.struct<(i32, i1)>
// LLVM:           tt.return

// LLVM-LABEL:   tt.func public @atomic_cas_float(
// LLVM:           %[[PTR:.*]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
// LLVM:           %[[CMP_BITS:.*]] = llvm.bitcast %arg2 : f32 to i32
// LLVM:           %[[NEW_BITS:.*]] = llvm.bitcast %arg3 : f32 to i32
// LLVM:           %[[CAS:.*]] = llvm.cmpxchg %[[PTR]], %[[CMP_BITS]], %[[NEW_BITS]] acq_rel acquire : !llvm.ptr, i32
// LLVM:           %[[OLD_BITS:.*]] = llvm.extractvalue %[[CAS]][0] : !llvm.struct<(i32, i1)>
// LLVM:           %[[OLD:.*]] = llvm.bitcast %[[OLD_BITS]] : i32 to f32
// LLVM:           tt.return %[[OLD]] : f32

// CHECK-LABEL:   tt.func public @atomic_add_tensor_dynamic(
// CHECK:           %[[ADD_DIM:.*]] = tensor.dim %arg2, %{{.*}} : tensor<?xi32>
// CHECK:           tensor.empty(%[[ADD_DIM]]) : tensor<?xi32>

// CHECK-LABEL:   tt.func public @atomic_cas_tensor_dynamic(
// CHECK:           %[[CAS_DIM:.*]] = tensor.dim %arg3, %{{.*}} : tensor<?xi32>
// CHECK:           tensor.empty(%[[CAS_DIM]]) : tensor<?xi32>
