//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONPTRTOMEMREF
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

namespace {

static Type getMemRefElementTypeForPointer(triton::PointerType ptrType) {
  Type pointeeType = ptrType.getPointeeType();
  if (auto shapedType = dyn_cast<ShapedType>(pointeeType)) {
    return shapedType.getElementType();
  }
  return pointeeType;
}

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(getMemRefElementTypeForPointer(ptrType),
                                     /*memorySpace=*/0);
    });
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(), ptrType.getPointeeType());
      }
      return std::nullopt;
    });

    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
  }
};

static bool
isFunctionEntryBlockArgument(Value value,
                             const llvm::DenseSet<Block *> &entryBlocks) {
  auto blockArgument = dyn_cast<BlockArgument>(value);
  return blockArgument && entryBlocks.contains(blockArgument.getOwner());
}

// A uniform pointer choice can remain a memref choice. Restrict this to
// function pointer arguments so derived pointer values continue through the
// ptr dialect, which is responsible for their byte-offset arithmetic.
static bool
isScalarPointerBaseSelection(Value value,
                             const llvm::DenseSet<Block *> &entryBlocks) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1 && cast->getNumResults() == 1) {
      return isScalarPointerBaseSelection(cast.getInputs().front(),
                                          entryBlocks);
    }
  }

  if (isFunctionEntryBlockArgument(value, entryBlocks)) {
    return true;
  }

  if (!isa<triton::PointerType>(value.getType())) {
    return false;
  }

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    return isScalarPointerBaseSelection(select.getTrueValue(), entryBlocks) &&
           isScalarPointerBaseSelection(select.getFalseValue(), entryBlocks);
  }

  auto ifOp = value.getDefiningOp<scf::IfOp>();
  if (!ifOp || ifOp.getThenRegion().empty() || ifOp.getElseRegion().empty()) {
    return false;
  }

  unsigned resultIndex = cast<OpResult>(value).getResultNumber();
  auto isBaseYield = [&](Region &region) {
    auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
    return yield && resultIndex < yield.getNumOperands() &&
           isScalarPointerBaseSelection(yield.getOperand(resultIndex),
                                        entryBlocks);
  };
  return isBaseYield(ifOp.getThenRegion()) && isBaseYield(ifOp.getElseRegion());
}

static bool
hasScalarPointerBaseSelection(scf::IfOp ifOp,
                              const llvm::DenseSet<Block *> &entryBlocks) {
  bool hasPointerResult = false;
  for (Value result : ifOp.getResults()) {
    if (!triton::isPtrTypeLike(result.getType())) {
      continue;
    }

    // ScalarPointerIfConverter rebuilds every result of the scf.if, but only
    // supports selecting scalar pointer bases. Leave the whole operation
    // untouched if it also returns another pointer-like type.
    if (!isa<triton::PointerType>(result.getType())) {
      return false;
    }

    hasPointerResult = true;
    if (!isScalarPointerBaseSelection(result, entryBlocks)) {
      return false;
    }
  }
  return hasPointerResult;
}

static bool
needsScalarPointerYieldConversion(scf::YieldOp yield,
                                  const llvm::DenseSet<Block *> &entryBlocks) {
  auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp());
  if (!ifOp) {
    return false;
  }

  if (hasScalarPointerBaseSelection(ifOp, entryBlocks)) {
    return true;
  }

  for (auto [result, operand] :
       llvm::zip(ifOp.getResults(), yield.getOperands())) {
    if (isa<UnrankedMemRefType>(result.getType()) &&
        isa<triton::PointerType>(operand.getType())) {
      return true;
    }
  }
  return false;
}

struct ScalarPointerSelectConverter
    : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<triton::PointerType>(op.getType())) {
      return failure();
    }

    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || adaptor.getTrueValue().getType() != resultType ||
        adaptor.getFalseValue().getType() != resultType) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, resultType, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

struct ScalarPointerIfConverter : public OpConversionPattern<scf::IfOp> {
  ScalarPointerIfConverter(const TypeConverter &typeConverter,
                           MLIRContext *context,
                           const llvm::DenseSet<Block *> &entryBlocks)
      : OpConversionPattern(typeConverter, context), entryBlocks(entryBlocks) {}

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasScalarPointerBaseSelection(op, entryBlocks)) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      Type convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType) {
        return failure();
      }
      resultTypes.push_back(convertedType);
    }

    auto newIf = scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                   adaptor.getCondition(), true);
    newIf->setAttrs(op->getAttrs());
    rewriter.eraseBlock(newIf.elseBlock());
    rewriter.eraseBlock(newIf.thenBlock());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().end());
    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }

private:
  const llvm::DenseSet<Block *> &entryBlocks;
};

struct ScalarPointerYieldConverter : public OpConversionPattern<scf::YieldOp> {
  ScalarPointerYieldConverter(const TypeConverter &typeConverter,
                              MLIRContext *context,
                              const llvm::DenseSet<Block *> &entryBlocks)
      : OpConversionPattern(typeConverter, context), entryBlocks(entryBlocks) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsScalarPointerYieldConversion(op, entryBlocks)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }

private:
  const llvm::DenseSet<Block *> &entryBlocks;
};

class TritonPtrToMemrefPass
    : public ::impl::TritonPtrToMemrefBase<TritonPtrToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;
    llvm::DenseSet<Block *> entryBlocks;
    auto collectEntryBlock = [&](auto funcOp) {
      if (!funcOp.isExternal()) {
        entryBlocks.insert(&funcOp.getFunctionBody().front());
      }
    };
    moduleOp.walk([&](func::FuncOp funcOp) { collectEntryBlock(funcOp); });
    moduleOp.walk([&](triton::FuncOp funcOp) { collectEntryBlock(funcOp); });

    // Update function signature and call ops to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return typeConverter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    target.addDynamicallyLegalOp<arith::SelectOp>([&](arith::SelectOp op) {
      return !isScalarPointerBaseSelection(op.getResult(), entryBlocks);
    });

    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return !hasScalarPointerBaseSelection(op, entryBlocks);
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      return !needsScalarPointerYieldConversion(op, entryBlocks);
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ScalarPointerSelectConverter>(typeConverter,
                                               patterns.getContext());
    patterns.add<ScalarPointerIfConverter, ScalarPointerYieldConverter>(
        typeConverter, patterns.getContext(), entryBlocks);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToMemrefPass() {
  return std::make_unique<TritonPtrToMemrefPass>();
}
