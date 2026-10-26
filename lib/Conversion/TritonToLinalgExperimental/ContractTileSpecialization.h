//===- ContractTileSpecialization.h - Rectangular tiles
//---------------------===//

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
