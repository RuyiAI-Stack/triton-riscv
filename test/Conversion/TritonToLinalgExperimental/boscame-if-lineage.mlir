// RUN: %PYTHON %S/Inputs/contract_frontend_cases.py %s %t triton-shared-opt
// Each function independently checks the v1 boundary; unsupported content or
// control flow must fail explicitly instead of claiming a complete graph.
// CONTRACT-CASE: if_lineage = reject: kernel contract V22/V23: read or partial inheritance has no initialized content version
module {
  tt.func public @if_lineage(
      %condition: i1, %lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>,
      %dst: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %selected = scf.if %condition -> (tensor<4xf32>) {
      %lhs_values = tt.load %lhs_ptrs : tensor<4x!tt.ptr<f32>>
      scf.yield %lhs_values : tensor<4xf32>
    } else {
      %rhs_values = tt.load %rhs_ptrs : tensor<4x!tt.ptr<f32>>
      scf.yield %rhs_values : tensor<4xf32>
    }
    %result = arith.addf %selected, %selected : tensor<4xf32>
    tt.store %dst_ptrs, %result : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
