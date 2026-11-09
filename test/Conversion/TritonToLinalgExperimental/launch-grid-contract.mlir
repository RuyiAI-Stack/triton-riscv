//
// FileCheck lit test input 'launch-grid-contract.mlir' (launch grid contract
// lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock launch grid
// contract lowering and contract facts.
//

module {
  tt.func public @kernel() attributes {test.custom = "retained"} {
    tt.return
  }
  tt.func public @preserved() attributes {gpu.known_grid_size = array<i32: 8, 9, 10>, test.custom = "also_retained"} {
    tt.return
  }
  func.func private @not_func() {
    return
  }
}
