//
// Structured Triton dialect to MemRef lowering.
//
// Inputs: pointer/mask analysis results on blocked loads and stores.
//
// Outputs: memref.load/store (and related ops) that BOSCAME / buddy passes can
// consume.
//
// When used: after TritonArithToLinalg when the access is proven contiguous.
//

#ifndef TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
#define TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonLoweringConfig.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"

void populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    bool enableTensorFirstVectorCpu = false, TritonLoweringConfig config = {});

void populateStructuredToMemrefPreConversionPatterns(
    RewritePatternSet &patterns, bool enableTensorFirstVectorCpu = false);

std::unique_ptr<OperationPass<ModuleOp>>
createStructuredToMemrefPass(bool enableTensorFirstVectorCpu = false,
                             TritonLoweringConfig config = {});

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_STRUCTUREDTOMEMREF_STRUCTUREDTOMEMREF_H
