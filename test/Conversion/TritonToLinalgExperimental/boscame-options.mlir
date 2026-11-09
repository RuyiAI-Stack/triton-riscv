//
// FileCheck lit test input 'boscame-options.mlir' (boscame options lowering and
// contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock boscame
// options lowering and contract facts.
//

module {
  tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %dp = tt.addptr %d, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %v = tt.load %sp : tensor<16x!tt.ptr<f32>>
    tt.store %dp, %v : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
