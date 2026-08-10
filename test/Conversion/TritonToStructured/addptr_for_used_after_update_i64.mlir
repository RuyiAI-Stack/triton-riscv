// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(%arg0 : !tt.ptr<bf16>) {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %c3_i64 = arith.constant 3 : i64
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32} : tensor<256xi32>
    %2 = arith.extsi %1 : tensor<256xi32> to tensor<256xi64>
    %3 = tt.addptr %0, %2 : tensor<256x!tt.ptr<bf16>>, tensor<256xi64>
    %4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %3) -> (tensor<256x!tt.ptr<bf16>>) {
      %5 = tt.splat %c3_i64 : i64 -> tensor<256xi64>
      %6 = tt.addptr %ptr, %5 : tensor<256x!tt.ptr<bf16>>, tensor<256xi64>
      %7 = tt.load %6 : tensor<256x!tt.ptr<bf16>>
      tt.store %6, %7 : tensor<256x!tt.ptr<bf16>>
      scf.yield %6 : tensor<256x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// CHECK-LABEL:       tt.func @kernel(%arg0: !tt.ptr<bf16>) {
// CHECK:             %c0 = arith.constant 0 : index
// CHECK:             %c12 = arith.constant 12 : index
// CHECK:             %c3 = arith.constant 3 : index
// CHECK:             %c1024 = arith.constant 1024 : index
// CHECK:             %c1 = arith.constant 1 : index
// CHECK:             %0 = scf.for %arg1 = %c0 to %c12 step %c3 iter_args(%arg2 = %c1024) -> (index) {
// CHECK:               %1 = arith.addi %arg2, %c3 : index
// CHECK:               %2 = tts.make_tptr %arg0 to sizes: [256], strides: [%c1], offsets: [%1], shape: [0], order: [] : <bf16> to tensor<256x!tt.ptr<bf16>>
// CHECK:               %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16>>) -> tensor<256xbf16>
// CHECK:               "tts.store"(%2, %3) <{static_mask_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16>>, tensor<256xbf16>) -> ()
// CHECK:               scf.yield %1 : index
// CHECK:             }
// CHECK:             tt.return
