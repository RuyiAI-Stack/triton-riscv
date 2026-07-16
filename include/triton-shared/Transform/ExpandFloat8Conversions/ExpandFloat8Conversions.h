#ifndef TRITON_SHARED_TRANSFORM_EXPAND_FLOAT8_CONVERSIONS_H
#define TRITON_SHARED_TRANSFORM_EXPAND_FLOAT8_CONVERSIONS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createExpandFloat8ConversionsPass();

} // namespace triton
} // namespace mlir

#endif
