// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:  %[[VAL_0:.*]]: memref<*xbf16>, %[[VAL_1:.*]]: memref<*xbf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32) {
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128], strides: [1] : memref<*xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           memref.copy %[[VAL_9]], %[[VAL_10]] : memref<128xbf16, strided<[1]>> to memref<128xbf16>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<128xbf16> to tensor<128xbf16>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_8]] : f32) outs(%[[VAL_12]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_14:.*]] = linalg.reduce ins(%[[VAL_11]] : tensor<128xbf16>) outs(%[[VAL_13]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_15:.*]]: bf16, %[[VAL_16:.*]]: f32) {
// CHECK:               %[[VAL_17:.*]] = arith.extf %[[VAL_15]] : bf16 to f32
// CHECK:               %[[VAL_18:.*]] = arith.addf %[[VAL_17]], %[[VAL_16]] : f32
// CHECK:               linalg.yield %[[VAL_18]] : f32
// CHECK:             }
// CHECK:           %[[VAL_19:.*]] = tensor.extract %[[VAL_14]][] : tensor<f32>
// CHECK:           %[[VAL_20:.*]] = arith.truncf %[[VAL_19]] : f32 to bf16
// CHECK:           %[[VAL_21:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1]>>
// CHECK:           affine.store %[[VAL_20]], %[[VAL_21]][0] : memref<1xbf16, strided<[1]>>
// CHECK:           return
// CHECK:         }
