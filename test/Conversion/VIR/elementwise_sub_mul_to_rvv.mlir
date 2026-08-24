// Elementwise linalg (euclidean distance tile) must lower through VIR to
// explicit vector ops before convert-linalg-to-loops scalarizes them.
// compiler.py adds -lower-linalg-to-vir -lower-vir-to-vector=vector-width=4
// on RISC-V targets so llc emits vfsub.vv / vfmul.vv.

// RUN: buddy-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize=allow-return-allocs-from-loops=true --lower-linalg-to-vir --lower-vir-to-vector=vector-width=4 | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @tile_sub_mul
// CHECK: arith.subf {{.*}} : vector<4xf32>
// CHECK: arith.mulf {{.*}} : vector<4xf32>
// CHECK-NOT: linalg.generic

func.func @tile_sub_mul(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) {
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b : memref<4xf32>, memref<4xf32>) outs(%out : memref<4xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %d = arith.subf %x, %y : f32
      %s = arith.mulf %d, %d : f32
      linalg.yield %s : f32
  }
  return
}
