# Optimization Opportunities

This document records optimization opportunities found while inspecting the
RISC-V vector-add lowering, updating the MLIR tests, and validating the
statically linked RVV executable. It distinguishes code-generation work from
test-maintenance work so that a green test suite is not mistaken for optimal
generated code.

## 1. Current Baseline

The optimized vector-add path can generate RVV arithmetic such as
`vfadd.vv`, write the result directly to the output buffer, and avoid temporary
heap allocations in the kernel. Buffer deallocation is also enabled in the
end-to-end compiler pipeline so allocations that cannot yet be eliminated are
paired with deallocations.

This is the desired shape for simple elementwise kernels:

```text
input A -- RVV load --+
                      +-- RVV elementwise op -- RVV store --> output
input B -- RVV load --+
```

The remaining opportunities are cases where the generic lowering still
materializes tensors through temporary buffers or cannot preserve pointer
semantics through control flow.

## 2. Eliminate Load-Side Temporary Buffers

A structured load may currently lower to:

```mlir
%src = memref.reinterpret_cast %arg0
    to offset: [%offset], sizes: [4, 256], strides: [1, 6]
%tmp = memref.alloc() : memref<4x256xbf16>
memref.copy %src, %tmp
%tensor = bufferization.to_tensor %tmp restrict writable
```

The operations have the following costs:

1. `memref.alloc` may become `malloc`.
2. `memref.copy` performs a complete input copy before the computation starts.
3. The copied tensor increases memory traffic and working-set size.
4. Later bufferization has to prove the allocation can be reused or removed.

For elementwise operations, the target lowering should consume the source view
directly. Depending on the access pattern, this can become a vector transfer,
a vector load, or a loop containing scalar/vector loads:

```text
strided input view -> vector/scalar load -> computation
```

An optimization should only remove the copy when aliasing, masking, bounds,
and layout semantics are preserved. A non-contiguous view such as strides
`[1, 6]` cannot be replaced blindly with a contiguous RVV load; it may require
strided RVV loads, indexed loads, or loop restructuring.

## 3. Destination Passing and Buffer Reuse

The generic tensor path may write a result using:

```mlir
%dst = memref.reinterpret_cast %arg1 ...
bufferization.materialize_in_destination %result in writable %dst
```

If `%result` was first built in an allocated tensor buffer, this creates an
additional copy to the real output. The preferred representation passes the
output destination into the operation from the beginning:

```text
loads -> elementwise operation with output destination -> output buffer
```

The existing masked elementwise store fusion is one implementation of this
idea. It should remain operation-generic: any side-effect-free, elementwise
arithmetic DAG with compatible shapes and masks should be eligible, rather
than matching only `arith.addf`.

Important legality conditions include:

- the intermediate tensor has no users that require its original storage;
- input/output aliasing does not change observable behavior;
- mask and boundary semantics are identical;
- element types and vector shapes are supported;
- operations with side effects or non-elementwise indexing are rejected.

## 4. Prefer Direct RVV Loads and Stores

After destination fusion, vectorizable elementwise kernels should reach LLVM
as direct vector memory operations and arithmetic. For floating-point add, the
expected RISC-V instructions include:

```asm
vle32.v
vfadd.vv
vse32.v
```

The same path should cover compatible elementwise operations such as
subtraction, multiplication, min/max, comparisons, select, casts, and supported
math operations. Whether an operation maps to one RVV instruction or expands
to a sequence is a target-lowering decision; the fusion pass should be based on
semantics and legality, not a hard-coded opcode list for add.

Code-generation checks should verify both properties:

- the expected RVV operation is present where applicable;
- avoidable `malloc`, `free`, and temporary-buffer copies are absent from the
  optimized kernel.

`free` must not simply be removed together with the assertion: if an allocation
remains, it must be deallocated. The best result is to eliminate both the
allocation and its deallocation through buffer reuse.

## 5. Gather/Scatter Pointer Representation

Gather/scatter accesses are represented as a tensor of element pointers:

```mlir
tensor<8x8x!tt.ptr<f32>>
```

They must not be modeled as a single pointer to a ranked tensor:

```mlir
!tt.ptr<tensor<8x8xf32>>
```

The latter implies a single regular block, while gather/scatter addresses may
be unrelated. Optimizations for this path should preserve the tensor-of-pointers
representation and lower it to indexed loads/stores or an equivalent loop. It
must not apply the contiguous-load optimization used for regular elementwise
accesses unless pointer analysis proves contiguity.

## 6. Conditional Pointer-Tensor Lowering

The experimental pipeline still has a known issue when an `scf.if` yields a
non-contiguous pointer tensor. Current output may retain conversions such as:

```mlir
builtin.unrealized_conversion_cast
    %base : memref<*xi64> to tensor<512x!ptr.ptr<...>>
```

The observed lowering also risks losing branch-local offset calculations. This
is a correctness issue, not merely an IR cleanup opportunity. The corresponding
test remains XFAIL until the representation can preserve, merge, and lower the
complete pointer state across control-flow joins.

A complete fix should:

1. preserve base, offset, stride, and mask information in every branch;
2. define a legal joined representation for `scf.if`/`scf.for` results;
3. lower all pointer conversions to real Ptr/MemRef operations;
4. reject unsupported non-contiguous cases instead of silently dropping state;
5. leave no `unrealized_conversion_cast` in the final pipeline output.

## 7. Tensor Descriptor Test Coverage

Upstream Triton removed `tt.make_tensor_ptr`. Tests using that operation could
no longer be parsed and were removed. The current API uses operations such as:

```mlir
tt.make_tensor_descriptor
tt.descriptor_load
tt.descriptor_store
```

Removing obsolete tests restores test-suite validity but does not replace their
coverage. New tests should be added for:

- descriptor load and store;
- dynamic shapes and strides;
- non-zero and computed offsets;
- the descriptor equivalent of block-pointer advance;
- boundary checks and masks;
- invalid layout/order diagnostics;
- descriptor-to-pointer rewriting followed by the RISC-V lowering pipeline.

This is primarily a correctness and maintenance task, but it also provides the
coverage needed to optimize descriptor accesses safely.

## 8. FileCheck Quality

Many checks had drifted from the current Triton/MLIR output. Automatically
generated checks are useful for establishing a baseline, but should be reduced
to the behavior each test owns. Prefer checks for:

- the relevant type conversion;
- the intended structured operation;
- absence of an illegal or unwanted operation;
- critical data-flow relationships.

Avoid checking every constant and SSA name unless the full pipeline shape is
the purpose of the test. When generating checks, use strict SSA-name matching:

```sh
generate-test-checks.py ... --strict_name_re=true
```

This prevents a greedy `.*` in one `CHECK-SAME` directive from consuming later
function arguments.

## 9. Measurement Plan

QEMU is appropriate for correctness, but not for host-side performance
comparison. Performance work should use two complementary methods:

1. Compile an equivalent lowering for native x86 and benchmark the old and new
   bufferization strategies. Measure wall time, allocation count, and copied
   bytes across multiple tensor sizes.
2. Inspect RISC-V assembly and static metrics: kernel size, number of vector
   loads/stores, scalar fallback instructions, calls to allocation routines,
   and loop structure.

Native x86 measurements estimate the cost of allocations and extra memory
traffic, but do not predict exact RVV speedup. Final hardware measurements are
required for target-specific conclusions.

Recommended benchmark cases are:

- contiguous vector add;
- masked tail handling;
- strided elementwise copy/add;
- input/output aliasing;
- gather/scatter;
- small tensors where allocation overhead dominates;
- large tensors where memory bandwidth dominates.

## 10. Priority Order

The recommended implementation order is:

1. Keep direct destination passing for all legal elementwise DAGs and expand
   its regression tests beyond add.
2. Eliminate load-side `alloc` and `memref.copy` for provably safe contiguous
   accesses.
3. Add strided/indexed RVV lowering without assuming contiguity.
4. Fix pointer tensors across conditional and loop control flow.
5. Restore block-access coverage using current tensor descriptors.
6. Reduce autogenerated FileCheck assertions to stable semantic checks.
7. Benchmark native allocation/copy overhead and validate final RVV assembly
   on real RISC-V hardware.
