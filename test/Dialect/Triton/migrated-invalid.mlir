// RUN: triton-shared-opt -verify-diagnostics -split-input-file %s

// The migrated TTIR verifiers must still reject malformed input.  These cases
// exercise op verifiers, type/interface constraints (TT_PtrLike / TT_Type on
// tt.load and tt.addptr) and the "no unknown operations" dialect contract.

// -----

module {
  tt.func @dot_inner_dim(%a: tensor<4x8xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) {
    // expected-error @+1 {{expected the last dimension of the first operand to be equal to the second-to-last dimension of the second operand}}
    %0 = tt.dot %a, %b, %c : tensor<4x8xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @return_arity() -> (i32, i32) {
    %0 = arith.constant 0 : i32
    // expected-error @+1 {{has 1 operands, but enclosing function (@return_arity) returns 2}}
    tt.return %0 : i32
  }
}

// -----

module {
  tt.func @broadcast_shrink(%a: tensor<4x8xf32>) {
    // expected-error @+1 {{Broadcast requires the source dimension to be 1}}
    %0 = tt.broadcast %a : tensor<4x8xf32> -> tensor<4x4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @load_not_a_pointer(%a: tensor<4xf32>) {
    // expected-error @+1 {{'ptr' must be ptr or ranked tensor of ptr values, but got 'tensor<4xf32>'}}
    %0 = tt.load %a : tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @addptr_not_a_pointer(%a: tensor<4xi32>, %b: tensor<4xi32>) {
    // expected-error @+1 {{'result' must be ptr or ranked tensor of ptr values, but got 'tensor<4xi32>'}}
    %0 = tt.addptr %a, %b : tensor<4xi32>, tensor<4xi32>
    tt.return
  }
}

// -----

module {
  tt.func @unknown_op() {
    // expected-error @+1 {{unregistered operation 'tt.not_an_op' found in dialect ('tt') that does not allow unknown operations}}
    "tt.not_an_op"() : () -> ()
    tt.return
  }
}
