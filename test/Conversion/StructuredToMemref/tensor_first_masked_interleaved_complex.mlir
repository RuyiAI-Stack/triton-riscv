// RUN: triton-shared-opt --structured-to-memref=enable-tensor-first-vector-cpu=true %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

module {
  func.func @masked_interleaved_complex(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: index, %arg3: index) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg2, %c2 : index
    %1 = tts.make_tptr %arg0 to sizes: [1024], strides: [%c2], offsets: [%0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %2 = "tts.load"(%1, %arg3) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %3 = arith.addi %0, %c1 : index
    %4 = tts.make_tptr %arg0 to sizes: [1024], strides: [%c2], offsets: [%3], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %5 = "tts.load"(%4, %arg3) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%2, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<1024xf32>
    %7 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%arg2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%7, %6, %arg3) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    return
  }
}

// CHECK-LABEL: func.func @masked_interleaved_complex
// CHECK-NOT: tts.make_tptr
// CHECK-NOT: tts.load
// CHECK-NOT: tts.store
// CHECK: memref.reinterpret_cast
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: linalg.generic
// CHECK: vector.transfer_read
// CHECK: vector.store
// CHECK: tensor.extract
// CHECK: memref.store
