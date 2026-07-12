#ifndef EXPAND_FLOAT8_CONVERSIONS_TRANSFORM_PASSES
#define EXPAND_FLOAT8_CONVERSIONS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ExpandFloat8Conversions
    : Pass<"expand-float8-conversions", "mlir::ModuleOp"> {
  let summary = "Expand f8E4M3FN conversions to integer arithmetic";
  let constructor = "triton::createExpandFloat8ConversionsPass()";
}

#endif
