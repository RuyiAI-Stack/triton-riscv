// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: for_lineage = complete
module {
  tt.func public @for_lineage(
      %initial_ptr: !tt.ptr<f32>, %body_ptr: !tt.ptr<f32>,
      %dst_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %initial_splat = tt.splat %initial_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %body_splat = tt.splat %body_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %initial_ptrs = tt.addptr %initial_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %body_ptrs = tt.addptr %body_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %initial = tt.load %initial_ptrs : tensor<4x!tt.ptr<f32>>
    %result = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%acc = %initial) -> (tensor<4xf32>) {
      %body = tt.load %body_ptrs : tensor<4x!tt.ptr<f32>>
      %next = arith.addf %acc, %body : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    tt.store %dst_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
