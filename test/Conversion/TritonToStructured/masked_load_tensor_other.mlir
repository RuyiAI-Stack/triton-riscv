// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s --check-prefix=STRUCTURED
// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s --check-prefix=UNSTRUCTURED
// RUN: triton-shared-opt --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu %s | FileCheck %s --check-prefix=LOWERED --implicit-check-not=tt.load --implicit-check-not=tptr.

// A nonuniform constant cannot be replaced by one scalar fill value.
// STRUCTURED-LABEL: @constant_other
// STRUCTURED: [[FILL:%.*]] = arith.constant dense<[1.000000e+00, -0.000000e+00, 3.000000e+00, 4.000000e+00]>
// STRUCTURED: [[MASK:%.*]] = arith.cmpi slt
// STRUCTURED: [[LOAD:%.*]] = "tts.load"
// STRUCTURED: arith.select [[MASK]], [[LOAD]], [[FILL]]
// UNSTRUCTURED-LABEL: @constant_other
// UNSTRUCTURED: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
// UNSTRUCTURED: [[MASK:%.*]] = arith.cmpi slt
// UNSTRUCTURED: [[LOAD:%.*]] = tts.gather {{.*}} mask = [[MASK]] default = [[ZERO]]
// UNSTRUCTURED: arith.select [[MASK]], [[LOAD]],
// LOWERED-LABEL: @constant_other
// LOWERED: arith.select
tt.func @constant_other(%src: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %limit = tt.splat %n : i32 -> tensor<4xi32>
  %mask = arith.cmpi slt, %idx, %limit : tensor<4xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %other = arith.constant dense<[1.0, -0.0, 3.0, 4.0]> : tensor<4xf32>
  %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %out_ptr = tt.addptr %out_base, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  tt.store %out_ptr, %value : tensor<4x!tt.ptr<f32>>
  tt.return
}

// This is the unique_dim pattern: a tensor load supplies inactive lane values
// for a second load with a non-prefix mask and irregular pointer offsets.
// STRUCTURED-LABEL: @runtime_other
// STRUCTURED: arith.select
// UNSTRUCTURED-LABEL: @runtime_other
// UNSTRUCTURED: [[ZERO:%.*]] = arith.constant 0 : i64
// UNSTRUCTURED: [[OTHER:%.*]] = tts.gather
// UNSTRUCTURED: [[MASK:%.*]] = arith.cmpi sgt
// UNSTRUCTURED: [[LOAD:%.*]] = tts.gather {{.*}} mask = [[MASK]] default = [[ZERO]]
// UNSTRUCTURED: arith.select [[MASK]], [[LOAD]], [[OTHER]]
// LOWERED-LABEL: @runtime_other
// LOWERED: arith.select
tt.func @runtime_other(%src: !tt.ptr<i64>, %out: !tt.ptr<i64>) {
  %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %zero = arith.constant dense<0> : tensor<4xi32>
  %one = arith.constant dense<1> : tensor<4xi32>
  %base = tt.splat %src : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %cur_ptr = tt.addptr %base, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  %cur = tt.load %cur_ptr : tensor<4x!tt.ptr<i64>>
  %mask = arith.cmpi sgt, %idx, %zero : tensor<4xi32>
  %prev_idx = arith.subi %idx, %one : tensor<4xi32>
  %safe_idx = arith.select %mask, %prev_idx, %zero : tensor<4xi1>, tensor<4xi32>
  %prev_ptr = tt.addptr %base, %safe_idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  %prev = tt.load %prev_ptr, %mask, %cur : tensor<4x!tt.ptr<i64>>
  %out_base = tt.splat %out : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %out_ptr = tt.addptr %out_base, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  tt.store %out_ptr, %prev : tensor<4x!tt.ptr<i64>>
  tt.return
}

// Runtime fills also work on the rectangular structured path.
// STRUCTURED-LABEL: @structured_runtime_other
// STRUCTURED: [[OTHER:%.*]] = "tts.load"
// STRUCTURED: [[LOAD:%.*]] = "tts.load"
// STRUCTURED: arith.select {{%.*}}, [[LOAD]], [[OTHER]]
// UNSTRUCTURED-LABEL: @structured_runtime_other
// UNSTRUCTURED: [[OTHER:%.*]] = tts.gather
// UNSTRUCTURED: [[LOAD:%.*]] = tts.gather
// UNSTRUCTURED: arith.select {{%.*}}, [[LOAD]], [[OTHER]]
// LOWERED-LABEL: @structured_runtime_other
// LOWERED: arith.select
tt.func @structured_runtime_other(%src: !tt.ptr<i64>, %fill: !tt.ptr<i64>, %out: !tt.ptr<i64>, %n: i32) {
  %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %limit = tt.splat %n : i32 -> tensor<4xi32>
  %mask = arith.cmpi slt, %idx, %limit : tensor<4xi32>
  %base = tt.splat %src : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %ptr = tt.addptr %base, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  %fill_base = tt.splat %fill : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %fill_ptr = tt.addptr %fill_base, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  %other = tt.load %fill_ptr : tensor<4x!tt.ptr<i64>>
  %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<i64>>
  %out_base = tt.splat %out : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %out_ptr = tt.addptr %out_base, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  tt.store %out_ptr, %value : tensor<4x!tt.ptr<i64>>
  tt.return
}

// A scalar splat retains the existing fast path without a tensor select.
// STRUCTURED-LABEL: @splat_other
// STRUCTURED: tts.load
// STRUCTURED-NOT: arith.select
// STRUCTURED: tt.return
// UNSTRUCTURED-LABEL: @splat_other
// UNSTRUCTURED: tts.gather
// UNSTRUCTURED-NOT: arith.select
// UNSTRUCTURED: tt.return
// LOWERED-LABEL: @splat_other
// LOWERED-NOT: arith.select
// LOWERED: return
tt.func @splat_other(%src: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %limit = tt.splat %n : i32 -> tensor<4xi32>
  %mask = arith.cmpi slt, %idx, %limit : tensor<4xi32>
  %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %ptr = tt.addptr %base, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %other = arith.constant dense<7.0> : tensor<4xf32>
  %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
  %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %out_ptr = tt.addptr %out_base, %idx : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  tt.store %out_ptr, %value : tensor<4x!tt.ptr<f32>>
  tt.return
}
