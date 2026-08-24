// CAS-style kernels (scatter_reduce prod/amax) lower to scf.while bodies that
// allocate tile buffers each iteration. Hoist allocs with buffer-loop-hoisting
// before buffer-deallocation so deallocs are inserted once at function exit
// (compiler.py pipeline order). Per-iteration heap alloc/free corrupts the
// heap on riscv64 (free(): invalid next size / SIGABRT).

// RUN: buddy-opt --buffer-loop-hoisting %s | FileCheck %s

// CHECK-LABEL: func.func @while_tile_alloc
// CHECK: memref.alloc() : memref<4xf32>
// CHECK: scf.while
// CHECK: memref.store
// CHECK: return
// CHECK-NOT: memref.alloc

func.func @while_tile_alloc() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1
  scf.while (%arg = %c0_i1) : (i1) -> i1 {
    %done = arith.cmpi eq, %arg, %c1_i1 : i1
    scf.condition(%done) %arg : i1
  } do {
  ^bb0(%arg: i1):
    %tile = memref.alloc() : memref<4xf32>
    memref.store %cst, %tile[%c0] : memref<4xf32>
    scf.yield %c1_i1 : i1
  }
  return
}
