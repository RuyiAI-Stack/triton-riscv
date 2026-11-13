// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @gather_rows(%x: tensor<4x8xf32>, %indices: tensor<2x8xi32>) -> tensor<2x8xf32> {
    %y = tt.gather %x[%indices] {axis = 0 : i32} : (tensor<4x8xf32>, tensor<2x8xi32>) -> tensor<2x8xf32>
    tt.return %y : tensor<2x8xf32>
  }
}
// CHECK-LABEL: func.func @gather_rows
// CHECK: linalg.generic
// CHECK: %[[ROW:.*]] = arith.index_castui {{.*}} : i32 to index
// CHECK: %[[COL:.*]] = linalg.index 1
// CHECK: tensor.extract {{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NOT: tt.gather

// -----

module {
  tt.func public @gather_columns(%x: tensor<4x8xf64>, %indices: tensor<4x4xi64>) -> tensor<4x4xf64> {
    %y = tt.gather %x[%indices] {axis = 1 : i32} : (tensor<4x8xf64>, tensor<4x4xi64>) -> tensor<4x4xf64>
    tt.return %y : tensor<4x4xf64>
  }
}
// CHECK-LABEL: func.func @gather_columns
// CHECK: linalg.generic
// CHECK: %[[ROW:.*]] = linalg.index 0
// CHECK: %[[COL:.*]] = arith.index_castui {{.*}} : i64 to index
// CHECK: tensor.extract {{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NOT: tt.gather

// -----

module {
  tt.func public @gather_bool_indices(%x: tensor<2xf32>, %indices: tensor<4xi1>) -> tensor<4xf32> {
    %y = tt.gather %x[%indices] {axis = 0 : i32} : (tensor<2xf32>, tensor<4xi1>) -> tensor<4xf32>
    tt.return %y : tensor<4xf32>
  }
}
// CHECK-LABEL: func.func @gather_bool_indices
// CHECK: linalg.generic
// CHECK: %[[INDEX:.*]] = arith.index_castui {{.*}} : i1 to index
// CHECK: tensor.extract {{.*}}[%[[INDEX]]]
// CHECK-NOT: tt.gather

// -----

module {
  tt.func public @gather_byte_indices(%x: tensor<256xf32>, %indices: tensor<4xi8>) -> tensor<4xf32> {
    %y = tt.gather %x[%indices] {axis = 0 : i32} : (tensor<256xf32>, tensor<4xi8>) -> tensor<4xf32>
    tt.return %y : tensor<4xf32>
  }
}
// CHECK-LABEL: func.func @gather_byte_indices
// CHECK: linalg.generic
// CHECK: %[[INDEX:.*]] = arith.index_castui {{.*}} : i8 to index
// CHECK: tensor.extract {{.*}}[%[[INDEX]]]
// CHECK-NOT: tt.gather
