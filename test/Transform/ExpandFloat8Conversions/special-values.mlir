// RUN: triton-shared-opt --expand-float8-conversions --canonicalize %s | FileCheck %s

module {
  func.func @scalar_fp8_specials() -> (f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN) {
    %overflow_pos = arith.constant 1.0e+9 : f32
    %overflow_neg = arith.constant -1.0e+9 : f32
    %inf_pos = arith.constant 0x7F800000 : f32
    %inf_neg = arith.constant 0xFF800000 : f32
    %nan_pos = arith.constant 0x7FC00000 : f32
    %nan_neg = arith.constant 0xFFC00000 : f32
    %0 = arith.truncf %overflow_pos : f32 to f8E4M3FN
    %1 = arith.truncf %overflow_neg : f32 to f8E4M3FN
    %2 = arith.truncf %inf_pos : f32 to f8E4M3FN
    %3 = arith.truncf %inf_neg : f32 to f8E4M3FN
    %4 = arith.truncf %nan_pos : f32 to f8E4M3FN
    %5 = arith.truncf %nan_neg : f32 to f8E4M3FN
    return %0, %1, %2, %3, %4, %5 : f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN, f8E4M3FN
  }
}

// CHECK: func.func @scalar_fp8_specials
// CHECK:   %[[POS_MAX:.+]] = arith.constant 4.480000e+02 : f8E4M3FN
// CHECK:   %[[NEG_MAX:.+]] = arith.constant -4.480000e+02 : f8E4M3FN
// CHECK:   %[[POS_NAN:.+]] = arith.constant 0x7F : f8E4M3FN
// CHECK:   %[[NEG_NAN:.+]] = arith.constant 0xFF : f8E4M3FN
// CHECK:   return %[[POS_MAX]], %[[NEG_MAX]], %[[POS_MAX]], %[[NEG_MAX]], %[[POS_NAN]], %[[NEG_NAN]]
