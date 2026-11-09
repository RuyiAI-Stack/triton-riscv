//
// FileCheck lit test input 'dynamic-mask-lineage.mlir' (dynamic mask lineage
// lowering and contract facts).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: TritonToLinalgExperimental conversion suite to lock dynamic mask
// lineage lowering and contract facts.
//

module {
  tt.func public @dynamic_prefix(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %limit: i32) {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %bound = tt.splat %limit : i32 -> tensor<32xi32>
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


// IR: %[[LIMIT:.*]] = arith.index_cast %arg2 : i32 to index
// IR: %[[CLAMPED:.*]] = arith.minsi %[[LIMIT]], %{{.*}} : index
// IR: %[[SIZE:.*]] = arith.maxsi %[[CLAMPED]], %{{.*}} : index
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.copy {{.*}}bosc_ame.memory = {{.*}}kind = "load"{{.*}}valid_shape = array<i64: -9223372036854775808>
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.copy {{.*}}bosc_ame.memory = {{.*}}kind = "store"{{.*}}valid_shape = array<i64: -9223372036854775808>

// NO-DYNAMIC-DMA: dynamic BOSCAME prefix extent must be clamped to the static maximum and provably divisible by a DMA packet
