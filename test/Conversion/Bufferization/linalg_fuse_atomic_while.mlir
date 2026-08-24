// linalg-fuse-elementwise-ops makes atomic-CAS scf.while loops carry memref
// state with per-iteration alloc/copy/dealloc (scatter_reduce prod/amax/amin).
// convert-scf-to-cf then emits imbalanced heap free calls. compiler.py skips
// linalg-fuse when ttshared IR contains __triton_shared_atomic_cas_ helpers.

// RUN: buddy-opt %S/Inputs/scatter_prod_while.ttshared.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize=allow-return-allocs-from-loops=true --buffer-loop-hoisting --buffer-deallocation-pipeline | FileCheck %s --check-prefix=GOOD
// RUN: buddy-opt %S/Inputs/scatter_prod_while.ttshared.mlir --linalg-fuse-elementwise-ops --empty-tensor-to-alloc-tensor --one-shot-bufferize=allow-return-allocs-from-loops=true --buffer-loop-hoisting --buffer-deallocation-pipeline | FileCheck %s --check-prefix=BAD

// GOOD: scf.while {{.*}} : (i1) -> ()
// BAD: scf.while {{.*}} : (memref<128xi1>, i1, i1) -> memref<128xi1>
