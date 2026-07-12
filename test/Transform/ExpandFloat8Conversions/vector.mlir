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

  func.func @fp8_to_f64(%arg0: f8E4M3FN) -> f64 {
    %0 = arith.extf %arg0 : f8E4M3FN to f64
    return %0 : f64
  }

  func.func @keeps_loaded_llvm_dialect(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>) -> vector<4xi32> {
    %0 = llvm.select %arg0, %arg1, %arg2 : vector<4xi1>, vector<4xi32>
    return %0 : vector<4xi32>
  }

  func.func @no_fp8_passthrough(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
    %0 = arith.addf %arg0, %arg1 : vector<4xf32>
    return %0 : vector<4xf32>
  }
}

// CHECK:     func.func @vector_fp8
// CHECK-NOT: arith.truncf {{.*}} to vector<4xf8E4M3FN>
// CHECK-NOT: arith.extf {{.*}} : vector<4xf8E4M3FN>
// CHECK-NOT:   scf.for
// CHECK-NOT:   func.call
// CHECK:       arith.bitcast {{.*}} : vector<4xf32> to vector<4xi32>
// CHECK:       arith.shrui
// CHECK:       arith.select
// CHECK:       arith.bitcast {{.*}} : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:       arith.bitcast {{.*}} : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK:       arith.shrui
// CHECK:       arith.select
// CHECK:       arith.bitcast {{.*}} : vector<4xi32> to vector<4xf32>
// CHECK:       return
// CHECK:     func.func @vector_fp8_f16
// CHECK-NOT:   scf.for
// CHECK-NOT:   func.call
// CHECK:       arith.extf {{.*}} : vector<4xf16> to vector<4xf32>
// CHECK:       arith.bitcast {{.*}} : vector<4xf32> to vector<4xi32>
// CHECK:       arith.bitcast {{.*}} : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK:       arith.bitcast {{.*}} : vector<4xi32> to vector<4xf32>
// CHECK:       arith.truncf {{.*}} : vector<4xf32> to vector<4xf16>
// CHECK:       return
// CHECK:     func.func @fp8_to_f64
// CHECK-NOT:   func.call
// CHECK:       arith.bitcast {{.*}} : f8E4M3FN to i8
// CHECK:       arith.bitcast {{.*}} : i32 to f32
// CHECK:       arith.extf {{.*}} : f32 to f64
// CHECK:       return
// CHECK:     func.func @keeps_loaded_llvm_dialect
// CHECK:       llvm.select
// CHECK:     func.func @no_fp8_passthrough
// CHECK:       arith.addf
// CHECK-NOT:   func.call
