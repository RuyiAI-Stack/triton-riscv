//
// FileCheck lit test input 'boscame-frontend-metadata.mlir' (boscame frontend
// metadata lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock boscame
// frontend metadata lowering and contract facts.
//

module {
  tt.func public @masked_copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %limit = arith.constant dense<16> : tensor<32xi32>
    %mask = arith.cmpi slt, %r, %limit : tensor<32xi32>
    %other = arith.constant dense<2.0> : tensor<32xf32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %dp = tt.addptr %d, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %v = tt.load %sp, %mask, %other : tensor<32x!tt.ptr<f32>>
    tt.store %dp, %v, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
// CHECK: contract.producer = "triton-riscv"
// CHECK-LABEL: func.func @masked_copy
// CHECK-SAME: contract.program = {axis = 0 : i32, role = "num_programs"}
// CHECK-SAME: contract.compute_graph = {complete = true
// CHECK: memref.alloc() alignment = 64
// CHECK: linalg.fill {{.*}}contract.fill = {padding = true}
// CHECK: memref.copy {{.*}}contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 16>}, origin_id = 0 : i64, other = 2.000000e+00 : f32}
// CHECK: memref.copy {{.*}}contract.access = {boundary = "external_ddr", kind = "store", mask = {kind = "rectangular", valid_shape = array<i64: 16>}, origin_id = 1 : i64}
// BAD-FACTS: dump-facts require contract=true
