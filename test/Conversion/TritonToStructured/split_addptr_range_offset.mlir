// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s

module {
// Preserve a nonzero range start and an i64 stride above 2^31.
// CHECK-LABEL: tt.func @split_addptr_wide_stride
// CHECK-DAG: arith.constant dense<2147483665> : tensor<8xi64>
// CHECK: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// CHECK: arith.muli {{.*}} : tensor<8xi64>
// CHECK: arith.addi {{.*}} : tensor<8xi64>
// CHECK: tts.make_gather_scatter_tptr {{.*}} strides: [1], offsets: [0]
  tt.func @split_addptr_wide_stride(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %stride = arith.constant dense<2147483665> : tensor<8xi64>
    %range = tt.make_range {start = 1 : i32, end = 9 : i32} : tensor<8xi32>
    %wide = arith.extsi %range : tensor<8xi32> to tensor<8xi64>
    %scaled = arith.muli %wide, %stride : tensor<8xi64>
    %idx_range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %idx_range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %structured = tt.addptr %base, %scaled : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %ptr = tt.addptr %structured, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = arith.constant dense<1.0> : tensor<8xf32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// Separate pointer additions sign-extend each operand before summing.
// CHECK-LABEL: tt.func @split_addptr_i32_overflow
// CHECK-DAG: arith.constant dense<2147483647> : tensor<8xi64>
// CHECK: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// CHECK: arith.addi {{.*}} : tensor<8xi64>
// CHECK: tts.make_gather_scatter_tptr {{.*}} strides: [1], offsets: [0]
  tt.func @split_addptr_i32_overflow(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %offset = arith.constant dense<2147483647> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %structured = tt.addptr %base, %offset : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr = tt.addptr %structured, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = arith.constant dense<1.0> : tensor<8xf32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// A grouped integer addition must retain its original i32 overflow.
// CHECK-LABEL: tt.func @grouped_i32_overflow
// CHECK: arith.addi {{.*}} : tensor<8xi32>
// CHECK-NOT: arith.addi {{.*}} : tensor<8xi64>
// CHECK: tts.make_gather_scatter_tptr
  tt.func @grouped_i32_overflow(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %offset = arith.constant dense<2147483647> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %mixed = arith.addi %offset, %idx : tensor<8xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %mixed : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = arith.constant dense<1.0> : tensor<8xf32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// Compute lane * stride in i32 before extending the pointer offset.
// For lane 2, i32(2 * 2147483647) is -2, not 4294967294.
// CHECK-LABEL: tt.func @structured_first_i32_overflow
// CHECK: %[[PRODUCT:.*]] = arith.muli {{.*}} : tensor<8xi32>
// CHECK: %[[WIDE_PRODUCT:.*]] = arith.extsi %[[PRODUCT]] : tensor<8xi32> to tensor<8xi64>
// CHECK: %[[MIXED_PRODUCT:.*]] = arith.addi %[[WIDE_PRODUCT]], {{.*}} : tensor<8xi64>
// CHECK: tts.make_gather_scatter_tptr {{.*}} gather_scatter_offset: %[[MIXED_PRODUCT]], strides: [1], offsets: [0]
  tt.func @structured_first_i32_overflow(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %stride = arith.constant dense<2147483647> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %scaled = arith.muli %range, %stride : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %structured = tt.addptr %base, %scaled : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr = tt.addptr %structured, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = arith.constant dense<1.0> : tensor<8xf32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// Preserve the same i32 arithmetic when the loaded pointer comes first.
// CHECK-LABEL: tt.func @structured_second_i32_overflow
// CHECK: %[[PRODUCT_R:.*]] = arith.muli {{.*}} : tensor<8xi32>
// CHECK: %[[WIDE_PRODUCT_R:.*]] = arith.extsi %[[PRODUCT_R]] : tensor<8xi32> to tensor<8xi64>
// CHECK: %[[MIXED_PRODUCT_R:.*]] = arith.addi {{.*}}, %[[WIDE_PRODUCT_R]] : tensor<8xi64>
// CHECK: tts.make_gather_scatter_tptr {{.*}} gather_scatter_offset: %[[MIXED_PRODUCT_R]], strides: [1], offsets: [0]
  tt.func @structured_second_i32_overflow(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %stride = arith.constant dense<2147483647> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %scaled = arith.muli %range, %stride : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %loaded = tt.addptr %base, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr = tt.addptr %loaded, %scaled : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %value = arith.constant dense<1.0> : tensor<8xf32>
    tt.store %ptr, %value : tensor<8x!tt.ptr<f32>>
    tt.return
  }

// Independent structured rows must keep their stride alongside gathered columns.
// CHECK-LABEL: tt.func @split_addptr_disjoint_dimensions
// CHECK: %[[ROW_STRIDE:.*]] = arith.constant 16 : index
// CHECK: %[[COLUMNS:.*]] = "tts.load"
// CHECK: %[[ROW_GATHER:.*]] = tts.make_gather_scatter_tptr {{.*}} to sizes: [4, 8] gather_scatter_dim: 1 gather_scatter_offset: %[[COLUMNS]], strides: [%[[ROW_STRIDE]], 1], offsets: [0, 0]
// CHECK: "tts.store"(%[[ROW_GATHER]],
  tt.func @split_addptr_disjoint_dimensions(%input: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {
    %row_stride = arith.constant dense<16> : tensor<4xi32>
    %rows = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %scaled = arith.muli %rows, %row_stride : tensor<4xi32>
    %row_offsets = tt.expand_dims %scaled {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %base, %row_offsets : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %row_matrix = tt.broadcast %row_ptrs : tensor<4x1x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
    %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %idx_base = tt.splat %indices : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr = tt.addptr %idx_base, %cols : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx = tt.load %idx_ptr : tensor<8x!tt.ptr<i32>>
    %col_offsets = tt.expand_dims %idx {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %col_matrix = tt.broadcast %col_offsets : tensor<1x8xi32> -> tensor<4x8xi32>
    %ptr = tt.addptr %row_matrix, %col_matrix : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %value = arith.constant dense<1.0> : tensor<4x8xf32>
    tt.store %ptr, %value : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }

// Normalize tensor<8xi32> to tensor<8x1xi32> before sign-extending it
// to combine with the first pointer offset, which is already tensor<8x1xi64>.
// CHECK-LABEL: tt.func @chained_expanded_gather_offsets
// CHECK-DAG: %[[BASE:.*]] = arith.constant dense<13> : tensor<8x1xi64>
// CHECK-DAG: %[[INDEX1:.*]] = "tts.load"
// CHECK-DAG: %[[INDEX2:.*]] = "tts.load"
// CHECK: %[[EXPANDED1:.*]] = tt.expand_dims %[[INDEX1]] {{.*}} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK: %[[WIDE1:.*]] = arith.extsi %[[EXPANDED1]] : tensor<8x1xi32> to tensor<8x1xi64>
// CHECK: %[[FIRST:.*]] = arith.addi %[[WIDE1]], %[[BASE]] : tensor<8x1xi64>
// CHECK: %[[RESHAPED2:.*]] = tt.reshape %[[INDEX2]] {{.*}} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK: %[[WIDE2:.*]] = arith.extsi %[[RESHAPED2]] : tensor<8x1xi32> to tensor<8x1xi64>
// CHECK: %[[COMBINED:.*]] = arith.addi %[[FIRST]], %[[WIDE2]] : tensor<8x1xi64>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[COMBINED]] {{.*}} : tensor<8x1xi64> into tensor<8xi64>
// CHECK: %[[PTR:.*]] = tts.make_gather_scatter_tptr {{.*}} gather_scatter_dim: 0 gather_scatter_offset: %[[FLAT]], strides: [1, 0], offsets: [0, 0]
// CHECK: "tts.store"(%[[PTR]],
  tt.func @chained_expanded_gather_offsets(%input: !tt.ptr<f32>, %indices1: !tt.ptr<i32>, %indices2: !tt.ptr<i32>) {
    %thirteen = arith.constant 13 : i32
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %idx_base1 = tt.splat %indices1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr1 = tt.addptr %idx_base1, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx1 = tt.load %idx_ptr1 : tensor<8x!tt.ptr<i32>>
    %idx_base2 = tt.splat %indices2 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_ptr2 = tt.addptr %idx_base2, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %idx2 = tt.load %idx_ptr2 : tensor<8x!tt.ptr<i32>>
    %expanded1 = tt.expand_dims %idx1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %expanded2 = tt.expand_dims %idx2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %shifted = tt.addptr %input, %thirteen : !tt.ptr<f32>, i32
    %base = tt.splat %shifted : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
    %first = tt.addptr %base, %expanded1 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %ptr = tt.addptr %first, %expanded2 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
    %value = arith.constant dense<1.0> : tensor<8x1xf32>
    tt.store %ptr, %value : tensor<8x1x!tt.ptr<f32>>
    tt.return
  }
}
