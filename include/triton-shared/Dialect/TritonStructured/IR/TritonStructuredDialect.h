#ifndef MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_
#define MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace tts {
namespace utils {
mlir::Value getScalarValue(mlir::Value operand, mlir::Location loc,
                           mlir::OpBuilder &builder);
}
} // namespace tts
} // namespace mlir

// The DynamicIndexList assembly format no longer has helpers in MLIR, so the
// generated parsers and printers resolve these instead.
namespace mlir {
namespace tts {
ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicOperands,
    DenseI64ArrayAttr &staticOperands);
void printDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                           OperandRange dynamicOperands,
                           DenseI64ArrayAttr staticOperands);
} // namespace tts
} // namespace mlir

//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

#endif
