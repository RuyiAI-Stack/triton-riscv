//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// Unstructured gather/scatter-style ops to MemRef lowering.
//
// Inputs: tts.gather / tts.scatter produced when structured lowering cannot
// prove a dense access.
//
// Outputs: memref forms that still carry contract buffer annotations where
// possible.
//
// When used: the fallback path of the shared pipeline after
// TritonToUnstructured.
//

#ifndef TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H
#define TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "triton-shared/Conversion/TritonLoweringConfig.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createUnstructuredToMemrefPass(TritonLoweringConfig config = {});
std::unique_ptr<OperationPass<ModuleOp>> createLowerAtomicCASToLLVMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H
