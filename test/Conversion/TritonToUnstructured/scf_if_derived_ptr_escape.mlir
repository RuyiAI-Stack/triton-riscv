// RUN: triton-shared-opt --triton-to-unstructured %s 2>&1 | FileCheck %s

module {
  tt.func public @if_yields_bitcast_addptr(%arg0: !tt.ptr<i32>, %cond: i1) -> !tt.ptr<i64> {
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.if %cond -> (!tt.ptr<i64>) {
      %1 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i32>, i32
      %2 = tt.bitcast %1 : !tt.ptr<i32> -> !tt.ptr<i64>
      scf.yield %2 : !tt.ptr<i64>
    } else {
      %1 = tt.bitcast %arg0 : !tt.ptr<i32> -> !tt.ptr<i64>
      scf.yield %1 : !tt.ptr<i64>
    }
    tt.return %0 : !tt.ptr<i64>
  }
}

// CHECK: warning: Cannot transform tensor of pointers into a single base pointer with tensor of offsets
// CHECK-LABEL: tt.func public @if_yields_bitcast_addptr
// CHECK:         %[[IF:.*]] = scf.if
// CHECK:           %[[ADDPTR:.*]] = tt.addptr
// CHECK:           %[[BITCAST:.*]] = tt.bitcast %[[ADDPTR]]
// CHECK:           scf.yield %[[BITCAST]] : !tt.ptr<i64>
// CHECK:         } else {
// CHECK:           %[[ELSE:.*]] = tt.bitcast
// CHECK:           scf.yield %[[ELSE]] : !tt.ptr<i64>
// CHECK:         }
// CHECK:         tt.return %[[IF]] : !tt.ptr<i64>
