// math.erf survives convert-math-to-llvm unless convert-math-to-libm follows.
// FlagGems gelu (approximate=none) and erf kernels fail mlir-translate without
// libm lowering (compiler.py standard/IME llvm_lowering_passes).

// RUN: buddy-opt %s --convert-math-to-llvm --convert-math-to-libm | FileCheck %s

// CHECK-LABEL: func.func @erf_tile
// CHECK: call @erff
// CHECK-NOT: math.erf

func.func @erf_tile(%arg0: f32) -> f32 {
  %0 = math.erf %arg0 : f32
  return %0 : f32
}
