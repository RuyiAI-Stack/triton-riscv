// RUN: triton-shared-opt --expand-float8-conversions %s | FileCheck %s

module {
  func.func @vector_fp8(%arg0: vector<4xf32>, %arg1: vector<4xf8E4M3FN>) -> (vector<4xf8E4M3FN>, vector<4xf32>) {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E4M3FN>
    %1 = arith.extf %arg1 : vector<4xf8E4M3FN> to vector<4xf32>
    return %0, %1 : vector<4xf8E4M3FN>, vector<4xf32>
  }

  func.func @vector_fp8_f16(%arg0: vector<4xf16>, %arg1: vector<4xf8E4M3FN>) -> (vector<4xf8E4M3FN>, vector<4xf16>) {
    %0 = arith.truncf %arg0 : vector<4xf16> to vector<4xf8E4M3FN>
    %1 = arith.extf %arg1 : vector<4xf8E4M3FN> to vector<4xf16>
    return %0, %1 : vector<4xf8E4M3FN>, vector<4xf16>
  }

  func.func @keeps_loaded_llvm_dialect(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>) -> vector<4xi32> {
    %0 = llvm.select %arg0, %arg1, %arg2 : vector<4xi1>, vector<4xi32>
    return %0 : vector<4xi32>
  }
}

// CHECK-DAG: func.func private @__triton_shared_f32_to_fp8e4nv(f32) -> i8
// CHECK-DAG: func.func private @__triton_shared_fp8e4nv_to_f32(i8) -> f32
// CHECK:     func.func @vector_fp8
// CHECK-NOT: arith.truncf {{.*}} to vector<4xf8E4M3FN>
// CHECK-NOT: arith.extf {{.*}} : vector<4xf8E4M3FN>
// CHECK:       scf.for
// CHECK:         vector.extractelement
// CHECK:         call @__triton_shared_f32_to_fp8e4nv
// CHECK:         vector.insertelement
// CHECK:       scf.for
// CHECK:         vector.extractelement
// CHECK:         call @__triton_shared_fp8e4nv_to_f32
// CHECK:         vector.insertelement
// CHECK:       return
// CHECK:     func.func @vector_fp8_f16
// CHECK:       scf.for
// CHECK:         vector.extractelement
// CHECK:         arith.extf {{.*}} : f16 to f32
// CHECK:         call @__triton_shared_f32_to_fp8e4nv
// CHECK:         vector.insertelement
// CHECK:       scf.for
// CHECK:         vector.extractelement
// CHECK:         call @__triton_shared_fp8e4nv_to_f32
// CHECK:         arith.truncf {{.*}} : f32 to f16
// CHECK:         vector.insertelement
// CHECK:       return
// CHECK:     func.func @keeps_loaded_llvm_dialect
// CHECK:       llvm.select
