// RUN: triton-shared-opt %s --triton-to-linalg-experimental -o %t.default
// RUN: triton-shared-opt %s --triton-to-linalg-experimental='contract=false' -o %t.false
// RUN: diff %t.default %t.false
// RUN: triton-shared-opt %s --triton-to-linalg-experimental='contract=true' | FileCheck %s --check-prefix=CONTRACT --implicit-check-not=bosc_ame. --implicit-check-not=plan. --implicit-check-not='tensor<' --implicit-check-not=bufferization.
// RUN: triton-shared-opt %s --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias' | FileCheck %s --check-prefix=ABI
// RUN: not triton-shared-opt %s --triton-to-linalg-experimental='contract=true caller-contract=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not triton-shared-opt %s --triton-to-linalg-experimental='boscame=true' 2>&1 | FileCheck %s --check-prefix=OLD-MODE
// RUN: not triton-shared-opt %s --triton-to-linalg-experimental='contract=true dma-abi=args_aligned64_nonalias' 2>&1 | FileCheck %s --check-prefix=OLD-ABI
// CONTRACT: contract.producer = "triton-riscv"
// CONTRACT-SAME: contract.version = 1 : i32
// CONTRACT-LABEL: func.func @copy
// CONTRACT: contract.compute_graph
// CONTRACT: memref.alloc
// CONTRACT: memref.copy
// CONTRACT-SAME: contract.access = {boundary = "external_ddr", kind = "load"
// CONTRACT-SAME: contract.task
// CONTRACT: memref.copy
// CONTRACT-SAME: contract.access = {boundary = "external_ddr", kind = "store"
// ABI: contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
// INVALID: invalid caller-contract: unknown
// OLD-MODE: no such option boscame
// OLD-ABI: no such option dma-abi

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
