#ifndef EXPAND_FLOAT8_CONVERSIONS_TRANSFORM_PASSES_H
#define EXPAND_FLOAT8_CONVERSIONS_TRANSFORM_PASSES_H

#include "triton-shared/Transform/ExpandFloat8Conversions/ExpandFloat8Conversions.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/ExpandFloat8Conversions/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
