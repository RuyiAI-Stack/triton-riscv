//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgExperimental : Pass<"triton-to-linalg-experimental", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "triton::createTritonToLinalgExperimentalPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"enableCollapseShape", "enable-collapse-shape", "bool", /*default*/"false",
             "Enable collapse shape pass">,
      Option<"structuredLdstMode", "structured-ldst-mode", "std::string", /*default*/"\"legacy_memref_bridge\"",
             "Structured load/store lowering mode: legacy_memref_bridge | tensor-first-vector-cpu">,
      Option<"cpuVectorWidth", "cpu-vector-width", "int64_t", /*default*/"16",
             "Forwarded to structured-to-memref cpu-vector-width (see that pass for semantics)">,
  ];
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let constructor = "triton::createReconcilePtrCastsPass()";
}

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
  let constructor = "triton::createTritonToPtrPass()";
}

def CollapseShape : Pass</*cli-arg*/"collapse-shape", /*Op*/"mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,transpose,fill,reduce} efficiency";
  let constructor = "triton::createCollapseShapePass()";
}
#endif
