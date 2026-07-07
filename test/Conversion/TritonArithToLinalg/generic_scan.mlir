// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @generic_scan(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%prev: f32, %cur: f32):
      %1 = arith.maximumf %prev, %cur : f32
      tt.scan.return %1 : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @generic_scan
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xf32>
// CHECK-NOT: tt.scan
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<4xf32>
// CHECK: [[IF:%.+]] = scf.if {{%.+}} -> (tensor<4xf32>) {
// CHECK:   [[FIRST:%.+]] = tensor.extract [[ARG0]]
// CHECK:   [[INIT:%.+]] = tensor.insert [[FIRST]] into [[EMPTY]]
// CHECK:   [[LOOP:%.+]]:2 = scf.for
// CHECK:     [[CUR:%.+]] = tensor.extract [[ARG0]]
// CHECK:     [[NEXT:%.+]] = arith.maximumf
// CHECK:     [[UPDATED:%.+]] = tensor.insert [[NEXT]] into
// CHECK:     scf.yield [[UPDATED]], [[NEXT]]
// CHECK:   scf.yield [[LOOP]]#0
// CHECK: } else {
// CHECK:   scf.yield [[EMPTY]]
// CHECK: }
// CHECK: return [[IF]]

// -----

// Use the associative, non-commutative left-projection combiner so that scan
// direction is semantically observable: a forward scan propagates the first
// element, while a reverse scan propagates the last element.
module {
  tt.func public @generic_reverse_scan(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%prev: i32, %cur: i32):
      tt.scan.return %prev : i32
    }) : (tensor<4xi32>) -> tensor<4xi32>
    tt.return %0 : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @generic_reverse_scan
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xi32>
// CHECK-NOT: tt.scan
// CHECK: [[LAST_INDEX:%.+]] = arith.constant 3 : index
// CHECK: [[C1:%.+]] = arith.constant 1 : index
// CHECK: [[UPPER:%.+]] = arith.constant 4 : index
// CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<4xi32>
// CHECK: [[IF:%.+]] = scf.if {{%.+}} -> (tensor<4xi32>) {
// CHECK:   [[LAST:%.+]] = tensor.extract [[ARG0]]{{\[}}[[LAST_INDEX]]]
// CHECK:   [[INIT:%.+]] = tensor.insert [[LAST]] into [[EMPTY]]{{\[}}[[LAST_INDEX]]]
// CHECK:   [[LOOP:%.+]]:2 = scf.for [[IV:%.+]] = [[C1]] to [[UPPER]] step [[C1]]
// CHECK:     [[FROM_END:%.+]] = arith.subi [[UPPER]], [[IV]] : index
// CHECK:     [[INDEX:%.+]] = arith.subi [[FROM_END]], [[C1]] : index
// CHECK:     [[UPDATED:%.+]] = tensor.insert {{%.+}} into {{%.+}}{{\[}}[[INDEX]]]
// CHECK:     scf.yield [[UPDATED]], {{%.+}}
// CHECK:   scf.yield [[LOOP]]#0
// CHECK: } else {
// CHECK:   scf.yield [[EMPTY]]
// CHECK: }
// CHECK: return [[IF]]
