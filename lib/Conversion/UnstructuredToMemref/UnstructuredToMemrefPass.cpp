//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "unstructured-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_UNSTRUCTUREDTOMEMREF
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

namespace {

static Type getMemRefElementTypeForPointer(triton::PointerType ptrType) {
  Type pointeeType = ptrType.getPointeeType();
  if (auto shapedType = dyn_cast<ShapedType>(pointeeType)) {
    return shapedType.getElementType();
  }
  return pointeeType;
}

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(getMemRefElementTypeForPointer(ptrType),
                                     0);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

static MemRefType getMemrefTypeForScalarPtr(triton::PointerType ptrType,
                                            MLIRContext *context) {
  SmallVector<int64_t> strides{1};
  auto layout = StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);
  auto elemType = ptrType.getPointeeType();
  auto memrefType = MemRefType::get({1}, elemType, layout);
  return memrefType;
}

struct ScalarLoadConverter : public OpConversionPattern<tts::GatherOp> {
  using OpConversionPattern<tts::GatherOp>::OpConversionPattern;

  ScalarLoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(typeConverter, context) {}

  ScalarLoadConverter(MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!gatherOp.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = gatherOp->getLoc();

    auto basePtr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    Value loadIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offset);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(gatherOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(loadIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    auto scalarLoadOp = rewriter.create<affine::AffineLoadOp>(
        loc, memref, zeroMap, ValueRange{});

    rewriter.replaceOp(gatherOp, scalarLoadOp.getResult());

    return success();
  }
};

struct ScalarStoreConverter : public OpConversionPattern<tts::ScatterOp> {
  using OpConversionPattern<tts::ScatterOp>::OpConversionPattern;

  ScalarStoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(typeConverter, context) {}

  ScalarStoreConverter(MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!scatterOp.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = scatterOp->getLoc();

    auto basePtr = adaptor.getPtr();
    auto offset = adaptor.getOffset();

    Value storeIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offset);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(scatterOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(storeIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto storeVal = scatterOp.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    rewriter.create<affine::AffineStoreOp>(loc, storeVal, memref, zeroMap,
                                           ValueRange{});
    rewriter.eraseOp(scatterOp);

    return success();
  }
};

// Lowering an unstructured load op (gather) into a linalg.generic op.
struct GatherConverter : public OpConversionPattern<tts::GatherOp> {
  using OpConversionPattern<tts::GatherOp>::OpConversionPattern;

  GatherConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(typeConverter, context) {}

  GatherConverter(MLIRContext *context)
      : OpConversionPattern<tts::GatherOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = gatherOp->getLoc();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = adaptor.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar load, skip processing.
    if (!offsetType) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(gatherOp.getResult().getType());

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref = rewriter
                          .create<memref::CastOp>(
                              loc,
                              MemRefType::get({ShapedType::kDynamic},
                                              resultType.getElementType()),
                              ptr)
                          .getResult();

    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    SmallVector<Value> loopBounds;
    loopBounds.reserve(resultType.getRank());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        loopBounds.push_back(
            rewriter.create<tensor::DimOp>(loc, offsetTensor, i));
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, resultType.getShape()[i]));
      }
    }

    Value fallback = gatherOp.getOther();
    if (!fallback) {
      auto elemType = resultType.getElementType();
      auto zeroAttr = rewriter.getZeroAttr(elemType);
      assert(zeroAttr && "unexpected element type");
      fallback = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    }

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, c0, loopBounds[0], c1,
                                            ValueRange{resultTensor});
    auto buildLoopNest = [&](auto &&self, OpBuilder &b, Location nestedLoc,
                             int64_t dim, SmallVector<Value> &indices,
                             Value tensorAcc) -> Value {
      if (dim == resultType.getRank()) {
        auto offset =
            b.create<tensor::ExtractOp>(nestedLoc, offsetTensor, indices);
        Value index0 =
            b.create<arith::IndexCastOp>(nestedLoc, b.getIndexType(), offset);
        Value loadValue =
            b.create<memref::LoadOp>(nestedLoc, baseMemref, ValueRange{index0});
        Value value = loadValue;
        if (gatherOp.getMask()) {
          Value mask = b.create<tensor::ExtractOp>(nestedLoc,
                                                   gatherOp.getMask(), indices);
          value =
              b.create<arith::SelectOp>(nestedLoc, mask, loadValue, fallback)
                  .getResult();
        }
        return b.create<tensor::InsertOp>(nestedLoc, value, tensorAcc, indices)
            .getResult();
      }

      auto nestedLoop = b.create<scf::ForOp>(nestedLoc, c0, loopBounds[dim], c1,
                                             ValueRange{tensorAcc});
      auto *body = nestedLoop.getBody();
      OpBuilder nestedBuilder = OpBuilder::atBlockBegin(body);
      indices.push_back(nestedLoop.getInductionVar());
      Value nextTensor = self(self, nestedBuilder, nestedLoc, dim + 1, indices,
                              nestedLoop.getRegionIterArgs()[0]);
      indices.pop_back();
      nestedBuilder.create<scf::YieldOp>(nestedLoc, nextTensor);
      return nestedLoop.getResult(0);
    };

    rewriter.setInsertionPointToStart(loop.getBody());
    SmallVector<Value> indices{loop.getInductionVar()};
    Value nextTensor = buildLoopNest(buildLoopNest, rewriter, loc, 1, indices,
                                     loop.getRegionIterArgs()[0]);
    rewriter.create<scf::YieldOp>(loc, nextTensor);

    rewriter.replaceOp(gatherOp, loop.getResult(0));

    return success();
  }
};

// Lowering an unstructured store op (scatter) into a linalg.generic op.
struct ScatterConverter : public OpConversionPattern<tts::ScatterOp> {
  using OpConversionPattern<tts::ScatterOp>::OpConversionPattern;

  ScatterConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(typeConverter, context) {}

  ScatterConverter(MLIRContext *context)
      : OpConversionPattern<tts::ScatterOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = scatterOp->getLoc();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = adaptor.getOffset();
    auto valueTensor = adaptor.getValue();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar store, skip processing.
    if (!offsetType) {
      return failure();
    }

    auto valueType = dyn_cast<RankedTensorType>(scatterOp.getValue().getType());

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref =
        rewriter
            .create<memref::CastOp>(loc,
                                    MemRefType::get({ShapedType::kDynamic},
                                                    valueType.getElementType()),
                                    ptr)
            .getResult();

    // The linalg.generic op should have the following inputs:
    // - the offset tensor.
    // - the value tensor.
    // - an optional mask tensor if the scatter op contains mask.
    SmallVector<Value> inputs{offsetTensor, valueTensor};

    if (scatterOp.getMask()) {
      inputs.push_back(scatterOp.getMask());
    }

    // Affine maps for the inputs.
    SmallVector<AffineMap> affineMaps(
        inputs.size(), rewriter.getMultiDimIdentityMap(valueType.getRank()));

    // All iterator types are parallel.
    SmallVector<utils::IteratorType> iteratorTypes(
        valueType.getRank(), utils::IteratorType::parallel);

    rewriter.setInsertionPoint(scatterOp);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, inputs, ValueRange{}, affineMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto storeValueAtIndex = [baseMemref](OpBuilder &b, Location loc,
                                                Value index, Value value) {
            Value index0 =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), index);

            b.create<memref::StoreOp>(loc, value, baseMemref,
                                      ValueRange{index0});
          };

          auto offset = args[0];
          auto value = args[1];

          if (!scatterOp.getMask()) {
            // If there is no mask, simply insert the current value to the
            // base memref using its offset.
            storeValueAtIndex(b, loc, offset, value);
          } else {
            // Keep the linalg.generic body single-block; mask false preserves
            // the existing destination value.
            auto mask = args[2];
            Value index0 =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), offset);
            Value oldValue =
                b.create<memref::LoadOp>(loc, baseMemref, ValueRange{index0});
            Value selected =
                b.create<arith::SelectOp>(loc, mask, value, oldValue);
            b.create<memref::StoreOp>(loc, selected, baseMemref,
                                      ValueRange{index0});
          }

          b.create<linalg::YieldOp>(loc);
        });

    rewriter.eraseOp(scatterOp);

    return success();
  }
};

class UnstructuredToMemrefPass
    : public ::impl::UnstructuredToMemrefBase<UnstructuredToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    target.addIllegalOp<tts::GatherOp, tts::ScatterOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<GatherConverter, ScatterConverter, ScalarLoadConverter,
                 ScalarStoreConverter>(typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createUnstructuredToMemrefPass() {
  return std::make_unique<UnstructuredToMemrefPass>();
}
