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

#define GEN_PASS_CLASSES
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

static Value getFlatMemref(Location loc, Value ptr, Type elementType,
                           ConversionPatternRewriter &rewriter) {
  return rewriter
      .create<memref::CastOp>(
          loc, MemRefType::get({ShapedType::kDynamic}, elementType), ptr)
      .getResult();
}

static Value createAtomicRMWValue(Location loc, triton::RMWOp rmwOp,
                                  Value oldValue, Value newValue,
                                  OpBuilder &builder) {
  Type type = oldValue.getType();
  switch (rmwOp) {
  case triton::RMWOp::AND:
    return builder.create<arith::AndIOp>(loc, oldValue, newValue);
  case triton::RMWOp::OR:
    return builder.create<arith::OrIOp>(loc, oldValue, newValue);
  case triton::RMWOp::XOR:
    return builder.create<arith::XOrIOp>(loc, oldValue, newValue);
  case triton::RMWOp::ADD:
    if (isa<FloatType>(type))
      return builder.create<arith::AddFOp>(loc, oldValue, newValue);
    return builder.create<arith::AddIOp>(loc, oldValue, newValue);
  case triton::RMWOp::FADD:
    return builder.create<arith::AddFOp>(loc, oldValue, newValue);
  case triton::RMWOp::MAX:
    if (isa<FloatType>(type))
      return builder.create<arith::MaximumFOp>(loc, oldValue, newValue);
    return builder.create<arith::MaxSIOp>(loc, oldValue, newValue);
  case triton::RMWOp::MIN:
    if (isa<FloatType>(type))
      return builder.create<arith::MinimumFOp>(loc, oldValue, newValue);
    return builder.create<arith::MinSIOp>(loc, oldValue, newValue);
  case triton::RMWOp::UMAX:
    return builder.create<arith::MaxUIOp>(loc, oldValue, newValue);
  case triton::RMWOp::UMIN:
    return builder.create<arith::MinUIOp>(loc, oldValue, newValue);
  case triton::RMWOp::XCHG:
    return newValue;
  }
  llvm_unreachable("unexpected atomic rmw op");
}

static Value createAtomicCASValue(Location loc, Value oldValue, Value cmpValue,
                                  Value newValue, OpBuilder &builder) {
  Type type = oldValue.getType();
  Value matched;
  if (isa<FloatType>(type)) {
    matched = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                            oldValue, cmpValue);
  } else {
    matched = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            oldValue, cmpValue);
  }
  return builder.create<arith::SelectOp>(loc, matched, newValue, oldValue);
}

static Value createZeroValue(Location loc, Type type, OpBuilder &builder) {
  auto zeroAttr = builder.getZeroAttr(type);
  assert(zeroAttr && "unexpected element type");
  return builder.create<arith::ConstantOp>(loc, zeroAttr);
}

static Value createMaskedLoadOrFallback(Location loc, Value mask,
                                        Value fallback, Value memref,
                                        Value index, OpBuilder &builder) {
  if (!mask) {
    return builder.create<memref::LoadOp>(loc, memref, ValueRange{index});
  }

  auto ifOp = builder.create<scf::IfOp>(loc, fallback.getType(), mask,
                                        /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value loaded =
        builder.create<memref::LoadOp>(loc, memref, ValueRange{index});
    builder.create<scf::YieldOp>(loc, loaded);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, fallback);
  }
  return ifOp.getResult(0);
}

template <typename StoreBuilder>
static void createOptionalMaskedStore(Location loc, Value mask,
                                      StoreBuilder storeBuilder,
                                      OpBuilder &builder) {
  if (!mask) {
    storeBuilder(builder);
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  auto ifOp = builder.create<scf::IfOp>(loc, mask);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  storeBuilder(builder);
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

    Value resultTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                         resultType.getElementType());
    SmallVector<Value> loopBounds;
    loopBounds.reserve(resultType.getRank());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        loopBounds.push_back(rewriter.create<tensor::DimOp>(loc, offsetTensor, i));
      } else {
        loopBounds.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, resultType.getShape()[i]));
      }
    }

    Value fallback = gatherOp.getOther();
    if (!fallback) {
      fallback = createZeroValue(loc, resultType.getElementType(), rewriter);
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
        Value value;
        if (gatherOp.getMask()) {
          Value mask = b.create<tensor::ExtractOp>(nestedLoc,
                                                   gatherOp.getMask(), indices);
          value = createMaskedLoadOrFallback(nestedLoc, mask, fallback,
                                             baseMemref, index0, b);
        } else {
          value = b.create<memref::LoadOp>(nestedLoc, baseMemref,
                                           ValueRange{index0});
        }
        return b.create<tensor::InsertOp>(nestedLoc, value, tensorAcc, indices)
            .getResult();
      }

      auto nestedLoop = b.create<scf::ForOp>(
          nestedLoc, c0, loopBounds[dim], c1, ValueRange{tensorAcc});
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

    auto offsetRankedType = dyn_cast<RankedTensorType>(offsetTensor.getType());
    auto valueType = dyn_cast<RankedTensorType>(valueTensor.getType());
    if (!offsetRankedType || !valueType ||
        offsetRankedType.getRank() != valueType.getRank()) {
      return failure();
    }

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref =
        rewriter
            .create<memref::CastOp>(loc,
                                    MemRefType::get({ShapedType::kDynamic},
                                                    valueType.getElementType()),
                                    ptr)
            .getResult();

    Value maskTensor = scatterOp.getMask() ? adaptor.getMask() : Value();
    if (maskTensor) {
      auto maskType = dyn_cast<RankedTensorType>(maskTensor.getType());
      if (!maskType || maskType.getRank() != valueType.getRank())
        return failure();
    }

    SmallVector<Value> loopBounds;
    loopBounds.reserve(offsetRankedType.getRank());
    for (int64_t i = 0, e = offsetRankedType.getRank(); i < e; ++i) {
      if (offsetRankedType.isDynamicDim(i)) {
        loopBounds.push_back(
            rewriter.create<tensor::DimOp>(loc, offsetTensor, i));
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, offsetRankedType.getShape()[i]));
      }
    }

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto buildLoopNest = [&](auto &&self, OpBuilder &b, Location nestedLoc,
                             int64_t dim, SmallVector<Value> &indices) -> void {
      if (dim == offsetRankedType.getRank()) {
        Value offsetValue =
            b.create<tensor::ExtractOp>(nestedLoc, offsetTensor, indices);
        Value index = b.create<arith::IndexCastOp>(nestedLoc, b.getIndexType(),
                                                   offsetValue);
        Value value =
            b.create<tensor::ExtractOp>(nestedLoc, valueTensor, indices);
        Value maskValue =
            maskTensor
                ? b.create<tensor::ExtractOp>(nestedLoc, maskTensor, indices)
                      .getResult()
                : Value();
        createOptionalMaskedStore(
            nestedLoc, maskValue,
            [&](OpBuilder &storeBuilder) {
              storeBuilder.create<memref::StoreOp>(nestedLoc, value, baseMemref,
                                                   ValueRange{index});
            },
            b);
        return;
      }

      auto loop = b.create<scf::ForOp>(nestedLoc, c0, loopBounds[dim], c1);
      OpBuilder nestedBuilder = OpBuilder::atBlockBegin(loop.getBody());
      indices.push_back(loop.getInductionVar());
      self(self, nestedBuilder, nestedLoc, dim + 1, indices);
      indices.pop_back();
    };

    SmallVector<Value> indices;
    buildLoopNest(buildLoopNest, rewriter, loc, 0, indices);

    rewriter.eraseOp(scatterOp);

    return success();
  }
};

struct AtomicRMWConverter : public OpConversionPattern<tts::AtomicRMWOp> {
  using OpConversionPattern<tts::AtomicRMWOp>::OpConversionPattern;

  AtomicRMWConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::AtomicRMWOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::AtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = atomicOp->getLoc();
    auto ptr = adaptor.getPtr();
    auto offset = adaptor.getOffset();
    auto value = adaptor.getVal();
    auto mask = adaptor.getMask();
    auto offsetType = dyn_cast<ShapedType>(offset.getType());

    if (!offsetType) {
      auto resultType = atomicOp.getResult().getType();
      Value baseMemref = getFlatMemref(loc, ptr, resultType, rewriter);
      Value index = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offset);
      Value fallback = createZeroValue(loc, resultType, rewriter);
      Value oldValue = createMaskedLoadOrFallback(loc, mask, fallback,
                                                  baseMemref, index, rewriter);
      Value storedValue = createAtomicRMWValue(loc, atomicOp.getAtomicRmwOp(),
                                               oldValue, value, rewriter);
      createOptionalMaskedStore(
          loc, mask,
          [&](OpBuilder &b) {
            b.create<memref::StoreOp>(loc, storedValue, baseMemref,
                                      ValueRange{index});
          },
          rewriter);
      rewriter.replaceOp(atomicOp, oldValue);
      return success();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(atomicOp.getResult().getType());
    auto valueType = dyn_cast<RankedTensorType>(atomicOp.getVal().getType());
    if (!resultType || !valueType) {
      return failure();
    }

    Value baseMemref =
        getFlatMemref(loc, ptr, resultType.getElementType(), rewriter);
    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    SmallVector<Value> loopBounds;
    loopBounds.reserve(resultType.getRank());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        loopBounds.push_back(rewriter.create<tensor::DimOp>(loc, value, i));
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, resultType.getShape()[i]));
      }
    }

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto buildLoopNest = [&](auto &&self, OpBuilder &b, Location nestedLoc,
                             int64_t dim, SmallVector<Value> &indices,
                             Value tensorAcc) -> Value {
      if (dim == resultType.getRank()) {
        Value offsetValue =
            b.create<tensor::ExtractOp>(nestedLoc, offset, indices);
        Value index = b.create<arith::IndexCastOp>(nestedLoc, b.getIndexType(),
                                                   offsetValue);
        Value maskValue =
            mask ? b.create<tensor::ExtractOp>(nestedLoc, mask, indices)
                       .getResult()
                 : Value();
        Value fallback =
            createZeroValue(nestedLoc, resultType.getElementType(), b);
        Value oldValue = createMaskedLoadOrFallback(
            nestedLoc, maskValue, fallback, baseMemref, index, b);
        Value newValue = b.create<tensor::ExtractOp>(nestedLoc, value, indices);
        Value storedValue = createAtomicRMWValue(
            nestedLoc, atomicOp.getAtomicRmwOp(), oldValue, newValue, b);
        createOptionalMaskedStore(
            nestedLoc, maskValue,
            [&](OpBuilder &storeBuilder) {
              storeBuilder.create<memref::StoreOp>(
                  nestedLoc, storedValue, baseMemref, ValueRange{index});
            },
            b);
        return b
            .create<tensor::InsertOp>(nestedLoc, oldValue, tensorAcc, indices)
            .getResult();
      }

      auto loop = b.create<scf::ForOp>(nestedLoc, c0, loopBounds[dim], c1,
                                       ValueRange{tensorAcc});
      auto *body = loop.getBody();
      OpBuilder nestedBuilder = OpBuilder::atBlockBegin(body);
      indices.push_back(loop.getInductionVar());
      Value nextTensor = self(self, nestedBuilder, nestedLoc, dim + 1, indices,
                              loop.getRegionIterArgs()[0]);
      indices.pop_back();
      nestedBuilder.create<scf::YieldOp>(nestedLoc, nextTensor);
      return loop.getResult(0);
    };

    SmallVector<Value> indices;
    Value result =
        buildLoopNest(buildLoopNest, rewriter, loc, 0, indices, resultTensor);
    rewriter.replaceOp(atomicOp, result);
    return success();
  }
};

struct AtomicCASConverter : public OpConversionPattern<tts::AtomicCASOp> {
  using OpConversionPattern<tts::AtomicCASOp>::OpConversionPattern;

  AtomicCASConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::AtomicCASOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::AtomicCASOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = atomicOp->getLoc();
    auto ptr = adaptor.getPtr();
    auto offset = adaptor.getOffset();
    auto cmp = adaptor.getCmp();
    auto value = adaptor.getVal();
    auto offsetType = dyn_cast<ShapedType>(offset.getType());

    if (!offsetType) {
      auto resultType = atomicOp.getResult().getType();
      Value baseMemref = getFlatMemref(loc, ptr, resultType, rewriter);
      Value index = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offset);
      Value oldValue =
          rewriter.create<memref::LoadOp>(loc, baseMemref, ValueRange{index});
      Value storedValue =
          createAtomicCASValue(loc, oldValue, cmp, value, rewriter);
      rewriter.create<memref::StoreOp>(loc, storedValue, baseMemref,
                                       ValueRange{index});
      rewriter.replaceOp(atomicOp, oldValue);
      return success();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(atomicOp.getResult().getType());
    auto valueType = dyn_cast<RankedTensorType>(atomicOp.getVal().getType());
    if (!resultType || !valueType) {
      return failure();
    }

    Value baseMemref =
        getFlatMemref(loc, ptr, resultType.getElementType(), rewriter);
    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    SmallVector<Value> loopBounds;
    loopBounds.reserve(resultType.getRank());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        loopBounds.push_back(rewriter.create<tensor::DimOp>(loc, value, i));
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, resultType.getShape()[i]));
      }
    }

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto buildLoopNest = [&](auto &&self, OpBuilder &b, Location nestedLoc,
                             int64_t dim, SmallVector<Value> &indices,
                             Value tensorAcc) -> Value {
      if (dim == resultType.getRank()) {
        Value offsetValue =
            b.create<tensor::ExtractOp>(nestedLoc, offset, indices);
        Value index = b.create<arith::IndexCastOp>(nestedLoc, b.getIndexType(),
                                                   offsetValue);
        Value oldValue =
            b.create<memref::LoadOp>(nestedLoc, baseMemref, ValueRange{index});
        Value cmpValue = b.create<tensor::ExtractOp>(nestedLoc, cmp, indices);
        Value newValue = b.create<tensor::ExtractOp>(nestedLoc, value, indices);
        Value storedValue =
            createAtomicCASValue(nestedLoc, oldValue, cmpValue, newValue, b);
        b.create<memref::StoreOp>(nestedLoc, storedValue, baseMemref,
                                  ValueRange{index});
        return b
            .create<tensor::InsertOp>(nestedLoc, oldValue, tensorAcc, indices)
            .getResult();
      }

      auto loop = b.create<scf::ForOp>(nestedLoc, c0, loopBounds[dim], c1,
                                       ValueRange{tensorAcc});
      auto *body = loop.getBody();
      OpBuilder nestedBuilder = OpBuilder::atBlockBegin(body);
      indices.push_back(loop.getInductionVar());
      Value nextTensor = self(self, nestedBuilder, nestedLoc, dim + 1, indices,
                              loop.getRegionIterArgs()[0]);
      indices.pop_back();
      nestedBuilder.create<scf::YieldOp>(nestedLoc, nextTensor);
      return loop.getResult(0);
    };

    SmallVector<Value> indices;
    Value result =
        buildLoopNest(buildLoopNest, rewriter, loc, 0, indices, resultTensor);
    rewriter.replaceOp(atomicOp, result);
    return success();
  }
};

class UnstructuredToMemrefPass
    : public UnstructuredToMemrefBase<UnstructuredToMemrefPass> {

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

    target.addIllegalOp<tts::GatherOp, tts::ScatterOp, tts::AtomicRMWOp,
                        tts::AtomicCASOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<GatherConverter, ScatterConverter, ScalarLoadConverter,
                 ScalarStoreConverter, AtomicRMWConverter, AtomicCASConverter>(
        typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createUnstructuredToMemrefPass() {
  return std::make_unique<UnstructuredToMemrefPass>();
}
