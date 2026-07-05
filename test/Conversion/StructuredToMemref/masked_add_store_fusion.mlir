// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func @masked_add(%lhs : !tt.ptr<f32>, %rhs : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<16xi32>
    %lhs_vals = tt.load %lhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %rhs_vals = tt.load %rhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %lhs_vals, %rhs_vals : tensor<16xf32>
    tt.store %out_ptrs, %sum, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_add
// CHECK:      vector.load
// CHECK:      arith.addf {{.*}} : vector<16xf32>
// CHECK:      vector.store
// CHECK-NOT:  memref.alloc
// CHECK-NOT:  memref.dealloc

// -----

module {
  tt.func @masked_float_chain(%lhs : !tt.ptr<f32>, %rhs : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<16xi32>
    %a = tt.load %lhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %b = tt.load %rhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<16xf32>
    %product = arith.mulf %sum, %b : tensor<16xf32>
    %result = arith.maximumf %product, %a : tensor<16xf32>
    tt.store %out_ptrs, %result, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_float_chain
// CHECK:      arith.addf {{.*}} : vector<16xf32>
// CHECK:      arith.mulf {{.*}} : vector<16xf32>
// CHECK:      arith.maximumf {{.*}} : vector<16xf32>
// CHECK-NOT:  memref.alloc

// -----

module {
  tt.func @masked_compare_select(%lhs : !tt.ptr<f32>, %rhs : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<16xi32>
    %a = tt.load %lhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %b = tt.load %rhs_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %condition = arith.cmpf ogt, %a, %b : tensor<16xf32>
    %result = arith.select %condition, %a, %b : tensor<16xi1>, tensor<16xf32>
    tt.store %out_ptrs, %result, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_compare_select
// The canonicalizer recognizes cmp+select as maximumf before fusion.
// CHECK:      arith.maximumf {{.*}} : vector<16xf32>
// CHECK-NOT:  memref.alloc

// -----

module {
  tt.func @masked_exp(%in : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %in_splat = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<16xi32>
    %value = tt.load %in_ptrs, %mask : tensor<16x!tt.ptr<f32>>
    %result = math.exp %value : tensor<16xf32>
    tt.store %out_ptrs, %result, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_exp
// CHECK:      math.exp {{.*}} : vector<16xf32>
// CHECK-NOT:  memref.alloc

// -----

module {
  tt.func @masked_integer_chain(%lhs : !tt.ptr<i32>, %rhs : !tt.ptr<i32>, %out : !tt.ptr<i32>, %n : i32) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %rhs_splat = tt.splat %rhs : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %out_splat = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %bound = tt.splat %n : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<16xi32>
    %a = tt.load %lhs_ptrs, %mask : tensor<16x!tt.ptr<i32>>
    %b = tt.load %rhs_ptrs, %mask : tensor<16x!tt.ptr<i32>>
    %sum = arith.addi %a, %b : tensor<16xi32>
    %result = arith.muli %sum, %b : tensor<16xi32>
    tt.store %out_ptrs, %result, %mask : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_integer_chain
// CHECK:      arith.addi {{.*}} : vector<16xi32>
// CHECK:      arith.muli {{.*}} : vector<16xi32>
// CHECK-NOT:  memref.alloc
