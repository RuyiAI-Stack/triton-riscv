#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
  let options = [
      Option<"enableTensorFirstVectorCpu", "enable-tensor-first-vector-cpu", "bool", /*default*/"false",
             "Enable tensor-first fast path for simple 1D contiguous load/store on CPU">,
      Option<"cpuVectorWidth", "cpu-vector-width", "int64_t", /*default*/"16",
             "Lane width for fixed-width vector<WxT> in 1D masked copy loops (16=AVX-512). "
             "Use 1 on targets where LLVM fixed-width vectors miscompile (e.g. riscv64 llc); "
             "RVV matmul uses buddy matmul-vectorization separately.">
  ];
}

#endif
