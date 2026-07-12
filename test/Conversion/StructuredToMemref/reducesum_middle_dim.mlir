// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>,
        %out_base: !tt.ptr<bf16>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : tensor<32x1xi32> -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : tensor<1x256xi32> -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : tensor<1x16xi32> -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : tensor<256x16xi32> -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : tensor<1x256x16xi32> -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 : tensor<32x256x16x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 1 : i32} : (tensor<32x256x16xbf16>) -> tensor<32x16xbf16>
    // out pointer, intentionally splat the base pointer for brevity
    %out = tt.splat %out_base : !tt.ptr<bf16> -> tensor<32x16x!tt.ptr<bf16>>
    tt.store %out, %5 : tensor<32x16x!tt.ptr<bf16>>
    tt.return
    }
}

// CHECK-LABEL: func.func @kernel
// CHECK: linalg.reduce
// CHECK: memref.store
// CHECK: return
