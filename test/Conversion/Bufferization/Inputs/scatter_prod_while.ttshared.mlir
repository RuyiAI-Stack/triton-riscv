#loc = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1)
#loc1 = loc(unknown)
#loc2 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":117:20)
#loc4 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":117:41)
#loc6 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":121:16)
#loc7 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":123:15)
#loc8 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":133:27)
#loc11 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":120:20)
#loc12 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":120:19)
#loc13 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":124:15)
#loc14 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":131:27)
#loc15 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":132:19)
#loc16 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":135:19)
#loc17 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":138:16)
#loc20 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":141:23)
#loc21 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":142:47)
#loc23 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":143:23)
#loc24 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":146:21)
#loc25 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":146:59)
#loc26 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":146:13)
#loc27 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":149:33)
#loc28 = loc("/home/caihanyi/work/triton-riscv/.venv/lib/python3.14/site-packages/triton/language/standard.py":293:12)
#loc29 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":149:26)
#map = affine_map<(d0) -> (d0)>
#loc30 = loc("index_ptr"(#loc))
#loc31 = loc("src_ptr"(#loc))
#loc32 = loc("out_ptr"(#loc))
#loc33 = loc("mask_ptr"(#loc))
#loc34 = loc("N"(#loc))
#loc35 = loc("src_ncols"(#loc))
#loc36 = loc("out_ncols"(#loc))
#loc37 = loc("base_offsets"(#loc2))
#loc38 = loc("base_offsets"(#loc4))
#loc40 = loc("mask"(#loc6))
#loc41 = loc("row"(#loc7))
#loc42 = loc("out_offsets"(#loc8))
#loc45 = loc("offsets"(#loc11))
#loc46 = loc("offsets"(#loc12))
#loc47 = loc("col"(#loc13))
#loc48 = loc("src_offsets"(#loc14))
#loc49 = loc("idx"(#loc15))
#loc50 = loc("src_val"(#loc16))
#loc51 = loc("stop"(#loc17))
#loc53 = loc("cur_val"(#loc20))
#loc54 = loc("new_val"(#loc21))
#loc56 = loc("cas_res"(#loc23))
#loc57 = loc("stop"(#loc24))
#loc58 = loc("stop"(#loc25))
#loc59 = loc("stop"(#loc26))
#loc60 = loc("block_stop"(#loc27))
#loc61 = loc("block_stop"(#loc29))
#loc64 = loc("stop"(#loc51))
#loc66 = loc(callsite(#loc28 at #loc61))
module {
  func.func private @__triton_shared_atomic_cas_relaxed(index, f32, f32) -> f32 loc(#loc)
  func.func @scatter_reduce_prod_2d_kernel(%arg0: memref<*xi64> {tt.divisibility = 16 : i32} loc("index_ptr"(#loc)), %arg1: memref<*xf32> {tt.divisibility = 16 : i32} loc("src_ptr"(#loc)), %arg2: memref<*xf32> {tt.divisibility = 16 : i32} loc("out_ptr"(#loc)), %arg3: memref<*xi32> {tt.divisibility = 16 : i32} loc("mask_ptr"(#loc)), %arg4: i32 loc("N"(#loc)), %arg5: i32 loc("src_ncols"(#loc)), %arg6: i32 loc("out_ncols"(#loc)), %arg7: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1), %arg8: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1), %arg9: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1), %arg10: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1), %arg11: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1), %arg12: i32 loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":103:1)) {
    %c4 = arith.constant 4 : index loc(#loc1)
    %c1 = arith.constant 1 : index loc(#loc1)
    %c0 = arith.constant 0 : index loc(#loc1)
    %false = arith.constant false loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %c512_i32 = arith.constant 512 : i32 loc(#loc37)
    %c4_i32 = arith.constant 4 : i32 loc(#loc3)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c128 = arith.constant 128 : index loc(#loc1)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %0 = tensor.empty() : tensor<128xi32> loc(#loc38)
    %1 = linalg.fill ins(%c1_i32 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32> loc(#loc1)
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32> loc(#loc1)
    %3 = arith.muli %arg10, %c512_i32 : i32 loc(#loc37)
    %4 = arith.index_cast %3 : i32 to index loc(#loc39)
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<128xi32>) {
    ^bb0(%out: i32 loc("base_offsets"(#loc4))):
      %16 = linalg.index 0 : index loc(#loc38)
      %17 = arith.index_cast %16 : index to i32 loc(#loc38)
      linalg.yield %17 : i32 loc(#loc38)
    } -> tensor<128xi32> loc(#loc38)
    %6 = linalg.fill ins(%3 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32> loc(#loc37)
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %5 : tensor<128xi32>, tensor<128xi32>) outs(%6 : tensor<128xi32>) {
    ^bb0(%in: i32 loc("base_offsets"(#loc2)), %in_0: i32 loc("base_offsets"(#loc4)), %out: i32 loc("base_offsets"(#loc2))):
      %16 = arith.addi %in, %in_0 : i32 loc(#loc37)
      linalg.yield %16 : i32 loc(#loc37)
    } -> tensor<128xi32> loc(#loc37)
    %8 = arith.extsi %arg4 : i32 to i64 loc(#loc40)
    %9 = tensor.empty() : tensor<128xi64> loc(#loc40)
    %10 = linalg.fill ins(%8 : i64) outs(%9 : tensor<128xi64>) -> tensor<128xi64> loc(#loc40)
    %11 = arith.extsi %arg5 : i32 to i64 loc(#loc41)
    %12 = linalg.fill ins(%11 : i64) outs(%9 : tensor<128xi64>) -> tensor<128xi64> loc(#loc41)
    %13 = arith.extsi %arg6 : i32 to i64 loc(#loc42)
    %14 = arith.index_cast %arg6 : i32 to index loc(#loc62)
    %15 = linalg.fill ins(%13 : i64) outs(%9 : tensor<128xi64>) -> tensor<128xi64> loc(#loc42)
    scf.for %arg13 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %16 = arith.muli %arg13, %c128_i32 : i32 loc(#loc44)
      %17 = arith.index_cast %16 : i32 to index loc(#loc39)
      %18 = linalg.fill ins(%16 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32> loc(#loc45)
      %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %18 : tensor<128xi32>, tensor<128xi32>) outs(%7 : tensor<128xi32>) {
      ^bb0(%in: i32 loc("base_offsets"(#loc2)), %in_1: i32 loc("offsets"(#loc11)), %out: i32 loc("base_offsets"(#loc2))):
        %39 = arith.addi %in, %in_1 : i32 loc(#loc45)
        linalg.yield %39 : i32 loc(#loc45)
      } -> tensor<128xi32> loc(#loc45)
      %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%19 : tensor<128xi32>) outs(%9 : tensor<128xi64>) {
      ^bb0(%in: i32 loc("offsets"(#loc11)), %out: i64 loc("offsets"(#loc12))):
        %39 = arith.extsi %in : i32 to i64 loc(#loc46)
        linalg.yield %39 : i64 loc(#loc46)
      } -> tensor<128xi64> loc(#loc46)
      %21 = tensor.empty() : tensor<128xi1> loc(#loc40)
      %22 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%20, %10 : tensor<128xi64>, tensor<128xi64>) outs(%21 : tensor<128xi1>) {
      ^bb0(%in: i64 loc("offsets"(#loc12)), %in_1: i64 loc("mask"(#loc6)), %out: i1 loc("mask"(#loc6))):
        %39 = arith.cmpi slt, %in, %in_1 : i64 loc(#loc40)
        linalg.yield %39 : i1 loc(#loc40)
      } -> tensor<128xi1> loc(#loc40)
      %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%20, %12 : tensor<128xi64>, tensor<128xi64>) outs(%20 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("offsets"(#loc12)), %in_1: i64 loc("row"(#loc7)), %out: i64 loc("offsets"(#loc12))):
        %39 = arith.divsi %in, %in_1 : i64 loc(#loc41)
        linalg.yield %39 : i64 loc(#loc41)
      } -> tensor<128xi64> loc(#loc41)
      %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%20, %12 : tensor<128xi64>, tensor<128xi64>) outs(%20 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("offsets"(#loc12)), %in_1: i64 loc("row"(#loc7)), %out: i64 loc("offsets"(#loc12))):
        %39 = arith.remsi %in, %in_1 : i64 loc(#loc47)
        linalg.yield %39 : i64 loc(#loc47)
      } -> tensor<128xi64> loc(#loc47)
      %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%23, %12 : tensor<128xi64>, tensor<128xi64>) outs(%23 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("row"(#loc7)), %in_1: i64 loc("row"(#loc7)), %out: i64 loc("row"(#loc7))):
        %39 = arith.muli %in, %in_1 : i64 loc(#loc48)
        linalg.yield %39 : i64 loc(#loc48)
      } -> tensor<128xi64> loc(#loc48)
      %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%25, %24 : tensor<128xi64>, tensor<128xi64>) outs(%25 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("src_offsets"(#loc14)), %in_1: i64 loc("col"(#loc13)), %out: i64 loc("src_offsets"(#loc14))):
        %39 = arith.addi %in, %in_1 : i64 loc(#loc48)
        linalg.yield %39 : i64 loc(#loc48)
      } -> tensor<128xi64> loc(#loc48)
      %cast = memref.cast %arg0 : memref<*xi64> to memref<?xi64> loc(#loc49)
      %27 = scf.for %arg14 = %c0 to %c128 step %c1 iter_args(%arg15 = %9) -> (tensor<128xi64>) {
        %extracted = tensor.extract %26[%arg14] : tensor<128xi64> loc(#loc49)
        %39 = arith.index_cast %extracted : i64 to index loc(#loc49)
        %extracted_1 = tensor.extract %22[%arg14] : tensor<128xi1> loc(#loc49)
        %40 = scf.if %extracted_1 -> (i64) {
          %41 = memref.load %cast[%39] : memref<?xi64> loc(#loc49)
          scf.yield %41 : i64 loc(#loc49)
        } else {
          scf.yield %c0_i64 : i64 loc(#loc49)
        } loc(#loc49)
        %inserted = tensor.insert %40 into %arg15[%arg14] : tensor<128xi64> loc(#loc49)
        scf.yield %inserted : tensor<128xi64> loc(#loc49)
      } loc(#loc49)
      %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%23, %15 : tensor<128xi64>, tensor<128xi64>) outs(%23 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("row"(#loc7)), %in_1: i64 loc("out_offsets"(#loc8)), %out: i64 loc("row"(#loc7))):
        %39 = arith.muli %in, %in_1 : i64 loc(#loc42)
        linalg.yield %39 : i64 loc(#loc42)
      } -> tensor<128xi64> loc(#loc42)
      %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%28, %27 : tensor<128xi64>, tensor<128xi64>) outs(%28 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("out_offsets"(#loc8)), %in_1: i64 loc("idx"(#loc15)), %out: i64 loc("out_offsets"(#loc8))):
        %39 = arith.addi %in, %in_1 : i64 loc(#loc42)
        linalg.yield %39 : i64 loc(#loc42)
      } -> tensor<128xi64> loc(#loc42)
      %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32> loc(#loc50)
      %30 = tensor.empty() : tensor<128xf32> loc(#loc50)
      %31 = scf.for %arg14 = %c0 to %c128 step %c1 iter_args(%arg15 = %30) -> (tensor<128xf32>) {
        %extracted = tensor.extract %26[%arg14] : tensor<128xi64> loc(#loc50)
        %39 = arith.index_cast %extracted : i64 to index loc(#loc50)
        %extracted_1 = tensor.extract %22[%arg14] : tensor<128xi1> loc(#loc50)
        %40 = scf.if %extracted_1 -> (f32) {
          %41 = memref.load %cast_0[%39] : memref<?xf32> loc(#loc50)
          scf.yield %41 : f32 loc(#loc50)
        } else {
          scf.yield %cst : f32 loc(#loc50)
        } loc(#loc50)
        %inserted = tensor.insert %40 into %arg15[%arg14] : tensor<128xf32> loc(#loc50)
        scf.yield %inserted : tensor<128xf32> loc(#loc50)
      } loc(#loc50)
      %32 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%22, %2, %1 : tensor<128xi1>, tensor<128xi32>, tensor<128xi32>) outs(%2 : tensor<128xi32>) {
      ^bb0(%in: i1 loc("mask"(#loc6)), %in_1: i32 loc(unknown), %in_2: i32 loc(unknown), %out: i32 loc(unknown)):
        %39 = arith.select %in, %in_1, %in_2 : i32 loc(#loc51)
        linalg.yield %39 : i32 loc(#loc51)
      } -> tensor<128xi32> loc(#loc51)
      %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%32, %2 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi1>) {
      ^bb0(%in: i32 loc("stop"(#loc17)), %in_1: i32 loc(unknown), %out: i1 loc("stop"(#loc17))):
        %39 = arith.cmpi ne, %in, %in_1 : i32 loc(#loc51)
        linalg.yield %39 : i1 loc(#loc51)
      } -> tensor<128xi1> loc(#loc51)
      %34 = arith.index_cast %14 : index to i64 loc(#loc42)
      %35 = linalg.fill ins(%34 : i64) outs(%9 : tensor<128xi64>) -> tensor<128xi64> loc(#loc42)
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%23, %35 : tensor<128xi64>, tensor<128xi64>) outs(%23 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("row"(#loc7)), %in_1: i64 loc("out_offsets"(#loc8)), %out: i64 loc("row"(#loc7))):
        %39 = arith.muli %in, %in_1 : i64 loc(#loc42)
        linalg.yield %39 : i64 loc(#loc42)
      } -> tensor<128xi64> loc(#loc42)
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%36, %27 : tensor<128xi64>, tensor<128xi64>) outs(%36 : tensor<128xi64>) {
      ^bb0(%in: i64 loc("out_offsets"(#loc8)), %in_1: i64 loc("idx"(#loc15)), %out: i64 loc("out_offsets"(#loc8))):
        %39 = arith.addi %in, %in_1 : i64 loc(#loc42)
        linalg.yield %39 : i64 loc(#loc42)
      } -> tensor<128xi64> loc(#loc42)
      %38 = scf.while (%arg14 = %33, %arg15 = %false) : (tensor<128xi1>, i1) -> tensor<128xi1> {
        %39 = arith.xori %arg15, %true : i1 loc(#loc19)
        scf.condition(%39) %arg14 : tensor<128xi1> loc(#loc19)
      } do {
      ^bb0(%arg14: tensor<128xi1> loc("stop"(#loc51))):
        %39 = arith.addi %4, %c128 : index loc(#loc53)
        %40 = arith.addi %4, %17 : index loc(#loc53)
        %41 = arith.addi %39, %17 : index loc(#loc53)
        %42 = arith.index_cast %arg4 : i32 to index loc(#loc65)
        %43 = arith.minsi %41, %42 : index loc(#loc53)
        %44 = arith.maxsi %43, %40 : index loc(#loc53)
        %45 = arith.subi %44, %40 : index loc(#loc53)
        %46 = arith.minsi %45, %c128 : index loc(#loc53)
        %47 = linalg.fill ins(%cst : f32) outs(%30 : tensor<128xf32>) -> tensor<128xf32> loc(#loc53)
        %48 = scf.for %arg15 = %c0 to %46 step %c1 iter_args(%arg16 = %47) -> (tensor<128xf32>) {
          %extracted_2 = tensor.extract %37[%arg15] : tensor<128xi64> loc(#loc53)
          %60 = arith.index_cast %extracted_2 : i64 to index loc(#loc53)
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%60], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>> loc(#loc43)
          %61 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>> loc(#loc53)
          %inserted = tensor.insert %61 into %arg16[%arg15] : tensor<128xf32> loc(#loc53)
          scf.yield %inserted : tensor<128xf32> loc(#loc53)
        } loc(#loc53)
        %49 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%48, %31 : tensor<128xf32>, tensor<128xf32>) outs(%48 : tensor<128xf32>) {
        ^bb0(%in: f32 loc("cur_val"(#loc20)), %in_2: f32 loc("src_val"(#loc16)), %out: f32 loc("cur_val"(#loc20))):
          %60 = arith.mulf %in, %in_2 : f32 loc(#loc54)
          linalg.yield %60 : f32 loc(#loc54)
        } -> tensor<128xf32> loc(#loc54)
        %50 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg14, %48, %49 : tensor<128xi1>, tensor<128xf32>, tensor<128xf32>) outs(%48 : tensor<128xf32>) {
        ^bb0(%in: i1 loc("stop"(#loc51)), %in_2: f32 loc("cur_val"(#loc20)), %in_3: f32 loc("new_val"(#loc21)), %out: f32 loc("cur_val"(#loc20))):
          %60 = arith.select %in, %in_2, %in_3 : f32 loc(#loc55)
          linalg.yield %60 : f32 loc(#loc55)
        } -> tensor<128xf32> loc(#loc55)
        %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32> loc(#loc56)
        %51 = scf.for %arg15 = %c0 to %c128 step %c1 iter_args(%arg16 = %30) -> (tensor<128xf32>) {
          %extracted_2 = tensor.extract %29[%arg15] : tensor<128xi64> loc(#loc56)
          %60 = arith.index_cast %extracted_2 : i64 to index loc(#loc56)
          %extracted_3 = tensor.extract %48[%arg15] : tensor<128xf32> loc(#loc56)
          %extracted_4 = tensor.extract %50[%arg15] : tensor<128xf32> loc(#loc56)
          %intptr = memref.extract_aligned_pointer_as_index %cast_1 : memref<?xf32> -> index loc(#loc56)
          %61 = arith.muli %60, %c4 : index loc(#loc56)
          %62 = arith.addi %intptr, %61 : index loc(#loc56)
          %63 = func.call @__triton_shared_atomic_cas_relaxed(%62, %extracted_3, %extracted_4) : (index, f32, f32) -> f32 loc(#loc56)
          %inserted = tensor.insert %63 into %arg16[%arg15] : tensor<128xf32> loc(#loc56)
          scf.yield %inserted : tensor<128xf32> loc(#loc56)
        } loc(#loc56)
        %52 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%48 : tensor<128xf32>) outs(%0 : tensor<128xi32>) {
        ^bb0(%in: f32 loc("cur_val"(#loc20)), %out: i32 loc("stop"(#loc24))):
          %60 = arith.bitcast %in : f32 to i32 loc(#loc57)
          linalg.yield %60 : i32 loc(#loc57)
        } -> tensor<128xi32> loc(#loc57)
        %53 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%51 : tensor<128xf32>) outs(%0 : tensor<128xi32>) {
        ^bb0(%in: f32 loc("cas_res"(#loc23)), %out: i32 loc("stop"(#loc25))):
          %60 = arith.bitcast %in : f32 to i32 loc(#loc58)
          linalg.yield %60 : i32 loc(#loc58)
        } -> tensor<128xi32> loc(#loc58)
        %54 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%52, %53 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi1>) {
        ^bb0(%in: i32 loc("stop"(#loc24)), %in_2: i32 loc("stop"(#loc25)), %out: i1 loc("stop"(#loc24))):
          %60 = arith.cmpi eq, %in, %in_2 : i32 loc(#loc57)
          linalg.yield %60 : i1 loc(#loc57)
        } -> tensor<128xi1> loc(#loc57)
        %55 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg14, %54 : tensor<128xi1>, tensor<128xi1>) outs(%arg14 : tensor<128xi1>) {
        ^bb0(%in: i1 loc("stop"(#loc51)), %in_2: i1 loc("stop"(#loc24)), %out: i1 loc("stop"(#loc51))):
          %60 = arith.ori %in, %in_2 : i1 loc(#loc59)
          linalg.yield %60 : i1 loc(#loc59)
        } -> tensor<128xi1> loc(#loc59)
        %56 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%55 : tensor<128xi1>) outs(%0 : tensor<128xi32>) {
        ^bb0(%in: i1 loc("stop"(#loc26)), %out: i32 loc("block_stop"(#loc27))):
          %60 = arith.extui %in : i1 to i32 loc(#loc60)
          linalg.yield %60 : i32 loc(#loc60)
        } -> tensor<128xi32> loc(#loc60)
        %57 = tensor.empty() : tensor<i32> loc(#loc66)
        %58 = linalg.fill ins(%c0_i32 : i32) outs(%57 : tensor<i32>) -> tensor<i32> loc(#loc66)
        %reduced = linalg.reduce ins(%56 : tensor<128xi32>) outs(%58 : tensor<i32>) dimensions = [0] 
          (%in: i32 loc(callsite(#loc28 at #loc61)), %init: i32 loc(callsite(#loc28 at #loc61))) {
            %60 = arith.addi %in, %init : i32 loc(#loc66)
            linalg.yield %60 : i32 loc(#loc66)
          } loc(#loc66)
        %extracted = tensor.extract %reduced[] : tensor<i32> loc(#loc66)
        %59 = arith.cmpi eq, %extracted, %c128_i32 : i32 loc(#loc61)
        scf.yield %55, %59 : tensor<128xi1>, i1 loc(#loc18)
      } loc(#loc63)
    } loc(#loc3)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc3 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":119:5)
#loc5 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":135:27)
#loc9 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":141:31)
#loc10 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":120:35)
#loc18 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":140:9)
#loc19 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":140:15)
#loc22 = loc("/home/caihanyi/work/triton-riscv/python/examples/flaggems/scatter_reduce.py":142:23)
#loc39 = loc("src_val"(#loc5))
#loc43 = loc("cur_val"(#loc9))
#loc44 = loc("offsets"(#loc10))
#loc52 = loc("stop"(#loc18))
#loc55 = loc("new_val"(#loc22))
#loc62 = loc(fused[#loc43, #loc42])
#loc63 = loc("block_stop"(#loc52))
#loc65 = loc(fused[#loc53, #loc40])

