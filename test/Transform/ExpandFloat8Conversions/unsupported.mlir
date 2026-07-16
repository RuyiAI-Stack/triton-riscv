// RUN: not triton-shared-opt --expand-float8-conversions %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @f64_to_fp8(%arg0: f64) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : f64 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

// CHECK: error: conversion from a float wider than f32 to f8E4M3FN is unsupported
