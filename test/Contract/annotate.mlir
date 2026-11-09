//
// FileCheck lit test input 'annotate.mlir' (kernel-contract-annotate attribute
// placement).
//
// Inputs: textual IR plus RUN: triton-shared-opt (or buddy-opt) and CHECK
// lines.
//
// Outputs: pass/fail in lit; IR dumps only on failure.
//
// When used: Contract suite to lock kernel-contract-annotate attribute
// placement.
//

module {
  func.func @kill_boundaries() {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %ub step %step {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- views.mlir
module {
  func.func @disjoint_byte_views() {
    %zero = arith.constant 0 : i8
    %b0 = arith.constant 0 : index
    %b32 = arith.constant 32 : index
    %bytes = memref.alloc() : memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%bytes : memref<64xi8>)
    %a = memref.view %bytes[%b0][] : memref<64xi8> to memref<4xf32>
    %b = memref.view %bytes[%b32][] : memref<64xi8> to memref<4xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : memref<4xf32>, memref<4xf32>) outs(%a : memref<4xf32>) {
    ^bb0(%x: f32, %y: f32, %old: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}

//--- staging.mlir
module {
  func.func @generated_access(%source: memref<4xi32>) {
    %snapshot = memref.alloc() : memref<4xi32>
    memref.copy %source, %snapshot {
      contract.access = {kind = "load", boundary = "external_ddr", mask = {kind = "none"}},
      contract.provenance = {generated_by = "planner-staging"}}
      : memref<4xi32> to memref<4xi32>
    return
  }
}

//--- zero-trip.mlir
module {
  func.func @unmodeled_zero_trip() {
    %zero = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    scf.for %i = %lb to %lb step %step {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    return
  }
}

//--- private-merge.mlir
module {
  func.func @unmodeled_private_merge(
      %pid: i32 {contract.program = {role = "program_id", axis = 0 : i32}}) {
    %zero = arith.constant 0 : i32
    %c = memref.alloc() : memref<4xi32>
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    %condition = arith.cmpi eq, %pid, %zero : i32
    scf.if %condition {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    } else {
      linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    }
    linalg.fill ins(%zero : i32) outs(%c : memref<4xi32>)
    return
  }
}

//--- overlap.mlir
module {
  func.func @overlapping_byte_views() {
    %zero = arith.constant 0 : i8
    %b0 = arith.constant 0 : index
    %b4 = arith.constant 4 : index
    %bytes = memref.alloc() : memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%bytes : memref<64xi8>)
    %a = memref.view %bytes[%b0][] : memref<64xi8> to memref<4xf32>
    %b = memref.view %bytes[%b4][] : memref<64xi8> to memref<4xf32>
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : memref<4xf32>, memref<4xf32>) outs(%a : memref<4xf32>) {
    ^bb0(%x: f32, %y: f32, %old: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}

//--- shifted-init.mlir
module {
  func.func @shift_does_not_initialize_root() {
    %zero = arith.constant 0 : i8
    %shift = arith.constant 1 : index
    %bytes = memref.alloc() : memref<64xi8>
    %shifted = memref.view %bytes[%shift][] : memref<64xi8> to memref<64xi8>
    linalg.fill ins(%zero : i8) outs(%shifted : memref<64xi8>)
    return
  }
}

//--- hidden-target.mlir
module attributes {test.hidden = {nested = [{bosc_ame.store_policy = "direct"}]}} {
  func.func @hidden(%x: memref<4xi32>) { return }
}

//--- hidden-tensor.mlir
module {
  func.func @hidden(%x: memref<4xi32> {test.hidden = {payload = tensor<4xf32>}}) {
    return
  }
}
