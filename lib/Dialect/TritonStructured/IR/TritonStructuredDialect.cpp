#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace mlir::tts;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DynamicIndexList assembly helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tts {

ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicOperands,
    DenseI64ArrayAttr &staticOperands) {
  SmallVector<int64_t> staticValues;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseLSquare())
    return failure();
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t staticValue = 0;
      OptionalParseResult parsedInteger =
          parser.parseOptionalInteger(staticValue);
      if (parsedInteger.has_value()) {
        if (failed(*parsedInteger))
          return failure();
        staticValues.push_back(staticValue);
        continue;
      }
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand))
        return failure();
      staticValues.push_back(ShapedType::kDynamic);
      operands.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();
  }
  dynamicOperands.assign(operands.begin(), operands.end());
  staticOperands = DenseI64ArrayAttr::get(parser.getContext(), staticValues);
  return success();
}

void printDynamicIndexList(OpAsmPrinter &printer, Operation *op,
                           OperandRange dynamicOperands,
                           DenseI64ArrayAttr staticOperands) {
  unsigned dynamicIndex = 0;
  printer << '[';
  llvm::interleaveComma(staticOperands.asArrayRef(), printer, [&](int64_t v) {
    if (v == ShapedType::kDynamic)
      printer << dynamicOperands[dynamicIndex++];
    else
      printer << v;
  });
  printer << ']';
}

} // namespace tts
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
