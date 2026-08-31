// Regression for riscv64: StructuredToMemref must not hardcode AVX-512
// vector<16xT> when cpu-vector-width=4. FlagGems fill_neg_inf (fp8_paged_mqa_logits)
// lowers through this path; W=16 miscompiles on riscv64 llc, W=4 matches VLEN=128.

// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu cpu-vector-width=4" %s | FileCheck %s

// CHECK-LABEL: func.func @fill_neg_inf
// CHECK: arith.constant 4 : index
// CHECK-NOT: arith.constant 16 : index
// CHECK-NOT: vector<16x
// CHECK: vector<4xf32>
// CHECK: vector.store

module {
  tt.func @fill_neg_inf(%out : !tt.ptr<f32>, %n : i32) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %neg_inf_f32 = arith.constant 0xFF800000 : f32
    %vals = tt.splat %neg_inf_f32 : f32 -> tensor<128xf32>
    %bound = tt.splat %n : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<128xi32>
    tt.store %out_ptrs, %vals, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
