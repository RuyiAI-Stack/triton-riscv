//
// FileCheck lit test input 'boscame-load-store.mlir' (boscame load store
// lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: BOSCAME FPGA integration suite to lock boscame load store lowering
// and contract facts.
//

module {
  tt.func public @masked(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) attributes {test.custom = "preserved"} {
    %c768 = arith.constant 768 : i32
    %ones = arith.constant dense<1.0> : tensor<1024xf32>
    %limit = arith.constant dense<768> : tensor<1024xi32>
    %row = tt.get_program_id x : i32
    %offset = arith.muli %row, %c768 : i32
    %srow = tt.addptr %src, %offset : !tt.ptr<f32>, i32
    %drow = tt.addptr %dst, %offset : !tt.ptr<f32>, i32
    %r = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %mask = arith.cmpi slt, %r, %limit : tensor<1024xi32>
    %s = tt.splat %srow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %d = tt.splat %drow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %sp = tt.addptr %s, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %dp = tt.addptr %d, %r : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %v = tt.load %sp, %mask, %ones : tensor<1024x!tt.ptr<f32>>
    tt.store %dp, %v, %mask : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
