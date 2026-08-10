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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
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
#define GEN_PASS_DEF_LOWERATOMICCASTOLLVM
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

namespace {

constexpr StringLiteral kAtomicCASHelperPrefix = "__triton_shared_atomic_cas_";

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

static FailureOr<arith::AtomicRMWKind> getAtomicRMWKind(triton::RMWOp rmwOp,
                                                        Type valueType) {
  switch (rmwOp) {
  case triton::RMWOp::AND:
    return arith::AtomicRMWKind::andi;
  case triton::RMWOp::OR:
    return arith::AtomicRMWKind::ori;
  case triton::RMWOp::XOR:
    return arith::AtomicRMWKind::xori;
  case triton::RMWOp::ADD:
    return isa<FloatType>(valueType) ? arith::AtomicRMWKind::addf
                                     : arith::AtomicRMWKind::addi;
  case triton::RMWOp::FADD:
    return arith::AtomicRMWKind::addf;
  case triton::RMWOp::MAX:
    return isa<FloatType>(valueType) ? arith::AtomicRMWKind::maximumf
                                     : arith::AtomicRMWKind::maxs;
  case triton::RMWOp::MIN:
    return isa<FloatType>(valueType) ? arith::AtomicRMWKind::minimumf
                                     : arith::AtomicRMWKind::mins;
  case triton::RMWOp::UMAX:
    return arith::AtomicRMWKind::maxu;
  case triton::RMWOp::UMIN:
    return arith::AtomicRMWKind::minu;
  case triton::RMWOp::XCHG:
    return arith::AtomicRMWKind::assign;
  }
  return failure();
}

static Value createAtomicRMWOp(Location loc, triton::RMWOp rmwOp, Value memref,
                               Value index, Value newValue,
                               OpBuilder &builder) {
  auto kind = getAtomicRMWKind(rmwOp, newValue.getType());
  assert(succeeded(kind) && "unexpected atomic rmw op");
  return builder
      .create<memref::AtomicRMWOp>(loc, *kind, newValue, memref,
                                   ValueRange{index})
      .getResult();
}

static FlatSymbolRefAttr
getOrCreateAtomicCASHelper(ModuleOp module, triton::MemSemantic semantic,
                           Type addressType, Type valueType,
                           OpBuilder &builder) {
  auto functionType = builder.getFunctionType(
      TypeRange{addressType, valueType, valueType}, TypeRange{valueType});
  std::string baseName =
      (kAtomicCASHelperPrefix + stringifyMemSemantic(semantic)).str();

  for (unsigned suffix = 0;; ++suffix) {
    std::string name = baseName;
    if (suffix != 0)
      name += "_" + std::to_string(suffix);
    if (auto func = module.lookupSymbol<func::FuncOp>(name)) {
      if (func.getFunctionType() == functionType)
        return SymbolRefAttr::get(builder.getContext(), name);
      continue;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), name, functionType);
    func.setPrivate();
    return SymbolRefAttr::get(builder.getContext(), name);
  }
}

static Value createAtomicCASCall(Location loc, ModuleOp module, Value memref,
                                 Value index, Value cmpValue, Value newValue,
                                 triton::MemSemantic semantic,
                                 OpBuilder &builder) {
  Type valueType = cmpValue.getType();
  unsigned byteWidth = (valueType.getIntOrFloatBitWidth() + 7) / 8;
  Value baseAddress =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value byteWidthValue = builder.create<arith::ConstantIndexOp>(loc, byteWidth);
  Value byteOffset = builder.create<arith::MulIOp>(loc, index, byteWidthValue);
  Value address = builder.create<arith::AddIOp>(loc, baseAddress, byteOffset);

  auto helper = getOrCreateAtomicCASHelper(module, semantic, address.getType(),
                                           valueType, builder);
  return builder
      .create<func::CallOp>(loc, helper, valueType,
                            ValueRange{address, cmpValue, newValue})
      .getResult(0);
}

static Value createLLVMAtomicCASOp(Location loc, Value address, Value cmpValue,
                                   Value newValue, triton::MemSemantic semantic,
                                   OpBuilder &builder) {
  Type valueType = cmpValue.getType();
  Value addressI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), address);
  Value pointer = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), addressI64);

  Value cmpBits = cmpValue;
  Value newBits = newValue;
  if (auto floatType = dyn_cast<FloatType>(valueType)) {
    Type integerType = builder.getIntegerType(floatType.getWidth());
    cmpBits = builder.create<LLVM::BitcastOp>(loc, integerType, cmpValue);
    newBits = builder.create<LLVM::BitcastOp>(loc, integerType, newValue);
  }

  LLVM::AtomicOrdering successOrdering;
  LLVM::AtomicOrdering failureOrdering;
  switch (semantic) {
  case triton::MemSemantic::RELAXED:
    successOrdering = LLVM::AtomicOrdering::monotonic;
    failureOrdering = LLVM::AtomicOrdering::monotonic;
    break;
  case triton::MemSemantic::ACQUIRE:
    successOrdering = LLVM::AtomicOrdering::acquire;
    failureOrdering = LLVM::AtomicOrdering::acquire;
    break;
  case triton::MemSemantic::RELEASE:
    successOrdering = LLVM::AtomicOrdering::release;
    failureOrdering = LLVM::AtomicOrdering::monotonic;
    break;
  case triton::MemSemantic::ACQUIRE_RELEASE:
    successOrdering = LLVM::AtomicOrdering::acq_rel;
    failureOrdering = LLVM::AtomicOrdering::acquire;
    break;
  }

  auto cmpxchg = builder.create<LLVM::AtomicCmpXchgOp>(
      loc, pointer, cmpBits, newBits, successOrdering, failureOrdering);
  Value oldBits = builder.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
  if (isa<FloatType>(valueType))
    return builder.create<LLVM::BitcastOp>(loc, valueType, oldBits);
  return oldBits;
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
      Value result;
      if (!mask) {
        result = createAtomicRMWOp(loc, atomicOp.getAtomicRmwOp(), baseMemref,
                                   index, value, rewriter);
      } else {
        auto ifOp = rewriter.create<scf::IfOp>(loc, resultType, mask,
                                               /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value thenResult =
              createAtomicRMWOp(loc, atomicOp.getAtomicRmwOp(), baseMemref,
                                index, value, rewriter);
          rewriter.create<scf::YieldOp>(loc, thenResult);
        }
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          rewriter.create<scf::YieldOp>(loc, fallback);
        }
        result = ifOp.getResult(0);
      }
      rewriter.replaceOp(atomicOp, result);
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

    SmallVector<Value> loopBounds;
    SmallVector<Value> dynamicResultDims;
    loopBounds.reserve(resultType.getRank());
    dynamicResultDims.reserve(resultType.getNumDynamicDims());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, value, i);
        loopBounds.push_back(dim);
        dynamicResultDims.push_back(dim);
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, resultType.getShape()[i]));
      }
    }
    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(),
        dynamicResultDims);

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
        Value newValue = b.create<tensor::ExtractOp>(nestedLoc, value, indices);
        Value oldValue;
        if (!maskValue) {
          oldValue = createAtomicRMWOp(nestedLoc, atomicOp.getAtomicRmwOp(),
                                       baseMemref, index, newValue, b);
        } else {
          Type elementType = resultType.getElementType();
          Value fallback = createZeroValue(nestedLoc, elementType, b);
          auto ifOp = b.create<scf::IfOp>(nestedLoc, elementType, maskValue,
                                          /*withElseRegion=*/true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&ifOp.getThenRegion().front());
            Value thenResult =
                createAtomicRMWOp(nestedLoc, atomicOp.getAtomicRmwOp(),
                                  baseMemref, index, newValue, b);
            b.create<scf::YieldOp>(nestedLoc, thenResult);
          }
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&ifOp.getElseRegion().front());
            b.create<scf::YieldOp>(nestedLoc, fallback);
          }
          oldValue = ifOp.getResult(0);
        }
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
    auto module = atomicOp->getParentOfType<ModuleOp>();

    if (!offsetType) {
      auto resultType = atomicOp.getResult().getType();
      Value baseMemref = getFlatMemref(loc, ptr, resultType, rewriter);
      Value index = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offset);
      Value oldValue = createAtomicCASCall(loc, module, baseMemref, index, cmp,
                                           value, atomicOp.getSem(), rewriter);
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

    SmallVector<Value> loopBounds;
    SmallVector<Value> dynamicResultDims;
    loopBounds.reserve(resultType.getRank());
    dynamicResultDims.reserve(resultType.getNumDynamicDims());
    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value dim = rewriter.create<tensor::DimOp>(loc, value, i);
        loopBounds.push_back(dim);
        dynamicResultDims.push_back(dim);
      } else {
        loopBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, resultType.getShape()[i]));
      }
    }
    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(),
        dynamicResultDims);

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
        Value cmpValue = b.create<tensor::ExtractOp>(nestedLoc, cmp, indices);
        Value newValue = b.create<tensor::ExtractOp>(nestedLoc, value, indices);
        Value oldValue =
            createAtomicCASCall(nestedLoc, module, baseMemref, index, cmpValue,
                                newValue, atomicOp.getSem(), b);
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

static FailureOr<triton::MemSemantic>
getAtomicCASSemantic(StringRef helperName) {
  if (!helperName.consume_front(kAtomicCASHelperPrefix))
    return failure();

  for (triton::MemSemantic semantic :
       {triton::MemSemantic::RELAXED, triton::MemSemantic::ACQUIRE,
        triton::MemSemantic::RELEASE, triton::MemSemantic::ACQUIRE_RELEASE}) {
    StringRef name = stringifyMemSemantic(semantic);
    if (helperName == name ||
        (helperName.consume_front(name) && helperName.starts_with("_")))
      return semantic;
  }
  return failure();
}

class LowerAtomicCASToLLVMPass
    : public ::impl::LowerAtomicCASToLLVMBase<LowerAtomicCASToLLVMPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (call.getCallee().starts_with(kAtomicCASHelperPrefix))
        calls.push_back(call);
    });

    for (func::CallOp call : calls) {
      FailureOr<triton::MemSemantic> semantic =
          getAtomicCASSemantic(call.getCallee());
      if (failed(semantic) || call.getNumOperands() != 3 ||
          call.getNumResults() != 1 ||
          !call.getOperand(0).getType().isIndex() ||
          call.getOperand(1).getType() != call.getResult(0).getType() ||
          call.getOperand(2).getType() != call.getResult(0).getType()) {
        call.emitError("invalid atomic CAS helper call");
        signalPassFailure();
        return;
      }

      OpBuilder builder(call);
      Value oldValue = createLLVMAtomicCASOp(
          call.getLoc(), call.getOperand(0), call.getOperand(1),
          call.getOperand(2), *semantic, builder);
      call.getResult(0).replaceAllUsesWith(oldValue);
      call.erase();
    }

    for (func::FuncOp func :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (func.isExternal() &&
          func.getName().starts_with(kAtomicCASHelperPrefix))
        func.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createUnstructuredToMemrefPass() {
  return std::make_unique<UnstructuredToMemrefPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createLowerAtomicCASToLLVMPass() {
  return std::make_unique<LowerAtomicCASToLLVMPass>();
}
