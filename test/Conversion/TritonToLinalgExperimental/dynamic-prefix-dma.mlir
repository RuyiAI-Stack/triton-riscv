//
// FileCheck lit test input 'dynamic-prefix-dma.mlir' (dynamic prefix dma
// lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock dynamic prefix
// dma lowering and contract facts.
//

module {
  tt.func public @dynamic_prefix_aligned(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %limit: i32) {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %packet_mask = arith.constant -16 : i32
    %aligned_bound = arith.andi %limit, %packet_mask : i32
    %bound = tt.splat %aligned_bound : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<32xi32>
    %other = arith.constant dense<0.0> : tensor<32xf32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %src_ptr = tt.addptr %srcs, %range : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %dst_ptr = tt.addptr %dsts, %range : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %value = tt.load %src_ptr, %mask, %other : tensor<32x!tt.ptr<f32>>
    tt.store %dst_ptr, %value, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
