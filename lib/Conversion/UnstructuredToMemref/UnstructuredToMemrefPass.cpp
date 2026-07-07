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
#include <functional>

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

// Lowering an unstructured store op (scatter) into explicit scalar stores.
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

    rewriter.setInsertionPoint(scatterOp);
    SmallVector<Value> loopBounds;
    loopBounds.reserve(valueType.getRank());
    for (int64_t i = 0, e = valueType.getRank(); i < e; ++i) {
      if (valueType.isDynamicDim(i)) {
        loopBounds.push_back(
            rewriter.create<tensor::DimOp>(loc, valueTensor, i));
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, valueType.getShape()[i]));
      }
    }

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto emitStore = [baseMemref](OpBuilder &b, Location nestedLoc,
                                  Value offset, Value value) {
      Value index0 =
          b.create<arith::IndexCastOp>(nestedLoc, b.getIndexType(), offset);
      b.create<memref::StoreOp>(nestedLoc, value, baseMemref,
                                ValueRange{index0});
    };

    std::function<void(OpBuilder &, Location, int64_t, SmallVector<Value> &)>
        buildLoopNest;
    buildLoopNest = [&](OpBuilder &b, Location nestedLoc, int64_t dim,
                        SmallVector<Value> &indices) {
      if (dim == valueType.getRank()) {
        Value offset =
            b.create<tensor::ExtractOp>(nestedLoc, offsetTensor, indices);
        Value value =
            b.create<tensor::ExtractOp>(nestedLoc, valueTensor, indices);
        if (!scatterOp.getMask()) {
          emitStore(b, nestedLoc, offset, value);
          return;
        }

        Value mask = b.create<tensor::ExtractOp>(nestedLoc, scatterOp.getMask(),
                                                 indices);
        auto ifOp = b.create<scf::IfOp>(nestedLoc, mask,
                                        /*withElseRegion=*/false);
        OpBuilder thenBuilder =
            OpBuilder::atBlockBegin(ifOp.thenBlock(), b.getListener());
        emitStore(thenBuilder, nestedLoc, offset, value);
        return;
      }

      auto loop = b.create<scf::ForOp>(nestedLoc, c0, loopBounds[dim], c1);
      OpBuilder bodyBuilder =
          OpBuilder::atBlockBegin(loop.getBody(), b.getListener());
      indices.push_back(loop.getInductionVar());
      buildLoopNest(bodyBuilder, nestedLoc, dim + 1, indices);
      indices.pop_back();
    };

    SmallVector<Value> indices;
    buildLoopNest(rewriter, loc, 0, indices);

    rewriter.eraseOp(scatterOp);

    return success();
  }
};

struct AtomicRMWConverter : public OpConversionPattern<tts::AtomicRMWOp> {
  using OpConversionPattern<tts::AtomicRMWOp>::OpConversionPattern;

  AtomicRMWConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::AtomicRMWOp>(typeConverter, context) {}

  AtomicRMWConverter(MLIRContext *context)
      : OpConversionPattern<tts::AtomicRMWOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::AtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rmw = atomicOp.getAtomicRmwOp();
    auto isSignedInt = [](Type type) { return type.isSignlessIntOrIndex(); };
    auto isFloat = [](Type type) { return isa<FloatType>(type); };
    auto isSupportedRMW = [&](Type elemType) {
      switch (rmw) {
      case triton::RMWOp::ADD:
        return elemType.isIntOrIndex();
      case triton::RMWOp::FADD:
        return isFloat(elemType);
      case triton::RMWOp::MAX:
      case triton::RMWOp::MIN:
        return isSignedInt(elemType);
      case triton::RMWOp::UMAX:
      case triton::RMWOp::UMIN:
        return elemType.isIntOrIndex();
      default:
        return false;
      }
    };
    auto emitUpdatedValue = [&](OpBuilder &builder, Location nestedLoc,
                                Type elemType, Value oldValue,
                                Value update) -> Value {
      switch (rmw) {
      case triton::RMWOp::ADD:
        return builder.create<arith::AddIOp>(nestedLoc, oldValue, update);
      case triton::RMWOp::FADD:
        return builder.create<arith::AddFOp>(nestedLoc, oldValue, update);
      case triton::RMWOp::MAX:
        return builder.create<arith::MaxSIOp>(nestedLoc, oldValue, update);
      case triton::RMWOp::MIN:
        return builder.create<arith::MinSIOp>(nestedLoc, oldValue, update);
      case triton::RMWOp::UMAX:
        return builder.create<arith::MaxUIOp>(nestedLoc, oldValue, update);
      case triton::RMWOp::UMIN:
        return builder.create<arith::MinUIOp>(nestedLoc, oldValue, update);
      default:
        llvm_unreachable("unsupported atomic_rmw op");
      }
    };
    auto loc = atomicOp->getLoc();
    auto ptr = adaptor.getPtr();
    auto offsetTensor = adaptor.getOffset();
    auto valueTensor = adaptor.getValue();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    if (!offsetType) {
      Type elemType = atomicOp.getValue().getType();
      if (!elemType.isIntOrIndexOrFloat()) {
        return rewriter.notifyMatchFailure(atomicOp,
                                           "expected scalar atomic_rmw value");
      }
      if (!isSupportedRMW(elemType)) {
        return rewriter.notifyMatchFailure(
            atomicOp, "unsupported scalar atomic_rmw op/type");
      }

      Value index0 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offsetTensor);
      auto memref = rewriter.create<memref::ReinterpretCastOp>(
          loc,
          getMemrefTypeForScalarPtr(
              cast<triton::PointerType>(atomicOp.getPtr().getType()),
              rewriter.getContext()),
          ptr, getAsOpFoldResult(index0) /*offset*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

      auto emitAtomicAdd = [&](OpBuilder &builder,
                               Location nestedLoc) -> Value {
        Value oldValue = builder.create<affine::AffineLoadOp>(
            nestedLoc, memref, zeroMap, ValueRange{});
        Value newValue = emitUpdatedValue(builder, nestedLoc, elemType,
                                          oldValue, valueTensor);
        builder.create<affine::AffineStoreOp>(nestedLoc, newValue, memref,
                                              zeroMap, ValueRange{});
        return oldValue;
      };

      Value oldValue;
      if (atomicOp.getMask()) {
        Value zero = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(elemType));
        auto ifOp =
            rewriter.create<scf::IfOp>(loc, elemType, atomicOp.getMask(), true);

        OpBuilder thenBuilder =
            OpBuilder::atBlockBegin(ifOp.thenBlock(), rewriter.getListener());
        Value thenOld = emitAtomicAdd(thenBuilder, loc);
        thenBuilder.create<scf::YieldOp>(loc, thenOld);

        OpBuilder elseBuilder =
            OpBuilder::atBlockBegin(ifOp.elseBlock(), rewriter.getListener());
        elseBuilder.create<scf::YieldOp>(loc, zero);

        oldValue = ifOp.getResult(0);
      } else {
        oldValue = emitAtomicAdd(rewriter, loc);
      }

      rewriter.replaceOp(atomicOp, oldValue);
      return success();
    }

    auto valueType = dyn_cast<RankedTensorType>(atomicOp.getValue().getType());
    if (!valueType) {
      return rewriter.notifyMatchFailure(atomicOp,
                                         "expected tensor atomic_rmw value");
    }

    auto elemType = valueType.getElementType();
    if (!isSupportedRMW(elemType)) {
      return rewriter.notifyMatchFailure(
          atomicOp, "unsupported tensor atomic_rmw op/type");
    }

    auto baseMemref =
        rewriter
            .create<memref::CastOp>(
                loc, MemRefType::get({ShapedType::kDynamic}, elemType), ptr)
            .getResult();

    Value resultTensor =
        rewriter.create<tensor::EmptyOp>(loc, valueType.getShape(), elemType);
    auto zeroAttr = rewriter.getZeroAttr(elemType);
    assert(zeroAttr && "unexpected atomic result element type");
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    resultTensor = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{zero},
                                               ValueRange{resultTensor})
                       .getResult(0);

    if (offsetType.getRank() != valueType.getRank()) {
      return rewriter.notifyMatchFailure(
          atomicOp, "atomic_rmw offset/value rank mismatch");
    }

    auto mask = atomicOp.getMask();
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      if (!maskType || maskType.getRank() != valueType.getRank()) {
        return rewriter.notifyMatchFailure(
            atomicOp, "expected ranked tensor mask matching atomic_rmw rank");
      }
    }

    // CPU backend correctness fallback: serialize each tensor lane in program
    // order so repeated offsets inside one block observe prior lane updates.
    // This is not a parallel-performance atomic implementation.
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto getDim = [&](unsigned dim) -> Value {
      int64_t staticDim = valueType.getDimSize(dim);
      if (staticDim != ShapedType::kDynamic) {
        return rewriter.create<arith::ConstantIndexOp>(loc, staticDim);
      }
      return rewriter.create<tensor::DimOp>(loc, valueTensor, dim);
    };

    auto emitAtomicAdd = [&](OpBuilder &builder, Location nestedLoc,
                             Value offset, Value update) -> Value {
      Value index0 = builder.create<arith::IndexCastOp>(
          nestedLoc, builder.getIndexType(), offset);
      Value oldValue = builder.create<memref::LoadOp>(nestedLoc, baseMemref,
                                                      ValueRange{index0});

      Value newValue =
          emitUpdatedValue(builder, nestedLoc, elemType, oldValue, update);

      builder.create<memref::StoreOp>(nestedLoc, newValue, baseMemref,
                                      ValueRange{index0});
      return oldValue;
    };

    std::function<Value(Value, unsigned, SmallVector<Value> &)> buildLoops =
        [&](Value currentTensor, unsigned dim,
            SmallVector<Value> &indices) -> Value {
      if (dim == valueType.getRank()) {
        Value offset = rewriter.create<tensor::ExtractOp>(loc, offsetTensor,
                                                          ValueRange{indices});
        Value update = rewriter.create<tensor::ExtractOp>(loc, valueTensor,
                                                          ValueRange{indices});

        Value oldValue;
        if (mask) {
          Value laneMask = rewriter.create<tensor::ExtractOp>(
              loc, mask, ValueRange{indices});
          auto ifOp = rewriter.create<scf::IfOp>(loc, elemType, laneMask, true);

          OpBuilder thenBuilder =
              OpBuilder::atBlockBegin(ifOp.thenBlock(), rewriter.getListener());
          Value thenOld = emitAtomicAdd(thenBuilder, loc, offset, update);
          thenBuilder.create<scf::YieldOp>(loc, thenOld);

          OpBuilder elseBuilder =
              OpBuilder::atBlockBegin(ifOp.elseBlock(), rewriter.getListener());
          elseBuilder.create<scf::YieldOp>(loc, zero);

          oldValue = ifOp.getResult(0);
          rewriter.setInsertionPointAfter(ifOp);
        } else {
          oldValue = emitAtomicAdd(rewriter, loc, offset, update);
        }

        return rewriter.create<tensor::InsertOp>(loc, oldValue, currentTensor,
                                                 ValueRange{indices});
      }

      auto forOp = rewriter.create<scf::ForOp>(loc, c0, getDim(dim), c1,
                                               ValueRange{currentTensor});
      rewriter.setInsertionPointToStart(forOp.getBody());

      indices.push_back(forOp.getInductionVar());
      Value yieldedTensor =
          buildLoops(forOp.getRegionIterArg(0), dim + 1, indices);
      indices.pop_back();

      rewriter.create<scf::YieldOp>(loc, yieldedTensor);
      rewriter.setInsertionPointAfter(forOp);
      return forOp.getResult(0);
    };

    SmallVector<Value> indices;
    Value replacement = buildLoops(resultTensor, 0, indices);

    rewriter.replaceOp(atomicOp, ValueRange{replacement});
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

    target.addIllegalOp<tts::GatherOp, tts::ScatterOp, tts::AtomicRMWOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<GatherConverter, ScatterConverter, AtomicRMWConverter,
                 ScalarLoadConverter, ScalarStoreConverter>(
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
