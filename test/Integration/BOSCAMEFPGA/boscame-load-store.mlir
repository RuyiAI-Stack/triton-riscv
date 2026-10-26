// RUN: contract-pipeline-opt %s --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias' --bosc-ame-plan --bosc-ame-plan-verify --bosc-ame-emit --promote-buffers-to-stack | FileCheck %s
// RUN: contract-pipeline-opt %s --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias structured-ldst-mode=tensor-first-vector-cpu' --bosc-ame-plan --bosc-ame-plan-verify --bosc-ame-emit --promote-buffers-to-stack | FileCheck %s
// RUN: contract-pipeline-opt %s --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias' --bosc-ame-plan --bosc-ame-plan-verify --bosc-ame-emit --promote-buffers-to-stack --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// CHECK: contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
// CHECK-LABEL: func.func @masked
// CHECK: bosc_ame.spm.view {{.*}} : memref<1024xf32, #bosc_ame.spm<{{[01]}}>>
// CHECK: vector.store {{.*}} : memref<1024xf32, #bosc_ame.spm<{{[01]}}>>, vector<16xf32>
// CHECK: arith.constant 3072 : index
// CHECK: bosc_ame.dma.start {{.*}} : (memref<768xf32, strided<[1], offset: ?>>,
// CHECK: bosc_ame.dma.wait
// CHECK: bosc_ame.dma.start {{.*}} : (memref<768xf32, strided<[1], offset: ?>, #bosc_ame.spm<{{[01]}}>>, memref<768xf32, strided<[1], offset: ?>>
// CHECK: bosc_ame.dma.wait
// CHECK-NOT: memref.alloc(
// CHECK-NOT: bufferization.
// LLVM: store <16 x float>
// LLVM: call i64 @bosc_ame_dma_start
// LLVM: call void @bosc_ame_dma_wait
// LLVM: call i64 @bosc_ame_dma_start
// LLVM: call void @bosc_ame_dma_wait

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
