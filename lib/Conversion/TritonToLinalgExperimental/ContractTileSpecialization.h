//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// Contract tile specialization for BOSCAME AOT GEMM/DMA shapes.
//
// Inputs: linalg matmul (or similar) plus kernel contract tile hints from the
// frontend.
//
// Outputs: specialized loops/tiles that match SPM/capacity facts without
// picking a GPU backend.
//
// When used: emitContract=true on TritonToLinalgExperimental.
//

#ifndef BUDDY_CONTRACTTILESPECIALIZATION_H
#define BUDDY_CONTRACTTILESPECIALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton {
/// Split proven partial last loop iterations and specialize rectangular
/// load/store bounds while accumulators are tensors, before graph annotation.
LogicalResult specializeContractTiles(ModuleOp module);
} // namespace mlir::triton

#endif
