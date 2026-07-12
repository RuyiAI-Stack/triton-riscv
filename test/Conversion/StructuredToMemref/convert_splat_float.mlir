// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save_ptr0 : !tt.ptr<f32>,
                    %save_ptr1 : !tt.ptr<bf16>) -> () {
        %0 = tt.splat %fin : f32 -> tensor<1024xf32>
        %1 = tt.splat %bin : bf16 -> tensor<128x256xbf16>
        // save pointers, intentionally splat the base pointer for brevity
        %save0 = tt.splat %save_ptr0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %save1 = tt.splat %save_ptr1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
        tt.store %save0, %0 : tensor<1024x!tt.ptr<f32>>
        tt.store %save1, %1 : tensor<128x256x!tt.ptr<bf16>>
        tt.return
    }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:  %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: bf16, %[[VAL_2:.*]]: memref<*xf32>, %[[VAL_3:.*]]: memref<*xbf16>, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32) {
// CHECK:           %[[VAL_10:.*]] = arith.constant 256 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_2]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] {
// CHECK:             memref.store %[[VAL_0]], %[[VAL_15]]{{\[}}%[[VAL_13]]] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_3]] : memref<*xbf16> to memref<?xbf16>
// CHECK:           scf.for %[[VAL_18:.*]] = %[[VAL_13]] to %[[VAL_11]] step %[[VAL_12]] {
// CHECK:             scf.for %[[VAL_19:.*]] = %[[VAL_13]] to %[[VAL_10]] step %[[VAL_12]] {
// CHECK:               memref.store %[[VAL_1]], %[[VAL_17]]{{\[}}%[[VAL_13]]] : memref<?xbf16>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
