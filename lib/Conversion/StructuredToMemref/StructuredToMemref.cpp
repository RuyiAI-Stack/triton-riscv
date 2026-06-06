//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";
static const std::string WRAP_LINEAR = "wrap_linear";

static bool hasUnitStride1DLayout(MemRefType memrefType) {
  if (memrefType.getRank() != 1) {
    return false;
  }
  auto stridesAndOffset = memrefType.getStridesAndOffset();
  int64_t stride = stridesAndOffset.first[0];
  return stride == 1;
}

static bool staticSizeCompatible1D(RankedTensorType tensorType,
                                   MemRefType memrefType) {
  if (tensorType.getRank() != 1 || memrefType.getRank() != 1) {
    return false;
  }
  int64_t tensorSize = tensorType.getShape()[0];
  int64_t memrefSize = memrefType.getShape()[0];
  return tensorSize == ShapedType::kDynamic || memrefSize == ShapedType::kDynamic ||
         tensorSize == memrefSize;
}

static bool staticShapeCompatible(RankedTensorType tensorType,
                                  MemRefType memrefType) {
  if (tensorType.getRank() != memrefType.getRank()) {
    return false;
  }

  for (auto [tensorDim, memrefDim] :
       llvm::zip(tensorType.getShape(), memrefType.getShape())) {
    if (!ShapedType::isDynamic(tensorDim) && !ShapedType::isDynamic(memrefDim) &&
        tensorDim != memrefDim) {
      return false;
    }
  }
  return true;
}

static memref::SubViewOp createFullSubview1D(Location loc, Value source,
                                             OpBuilder &b) {
  auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  auto size = b.create<memref::DimOp>(loc, source, 0).getResult();
  SmallVector<OpFoldResult> offsets = {c0.getResult()};
  SmallVector<OpFoldResult> sizes = {size};
  SmallVector<OpFoldResult> strides = {c1.getResult()};
  auto srcType = cast<MemRefType>(source.getType());
  auto subviewType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(subviewType), source,
                                     offsets, sizes, strides);
}

static FailureOr<Value> ensureRankedMemRef(Value source, int64_t rank,
                                           Type elementType, Location loc,
                                           OpBuilder &rewriter) {
  if (isa<MemRefType>(source.getType())) {
    return source;
  }

  auto unrankedType = dyn_cast<UnrankedMemRefType>(source.getType());
  if (!unrankedType || unrankedType.getElementType() != elementType) {
    return failure();
  }

  SmallVector<int64_t> dynamicShape(rank, ShapedType::kDynamic);
  SmallVector<int64_t> dynamicStrides(rank, ShapedType::kDynamic);
  auto dynamicLayout = StridedLayoutAttr::get(rewriter.getContext(),
                                              ShapedType::kDynamic,
                                              dynamicStrides);
  auto rankedType = MemRefType::get(dynamicShape, elementType, dynamicLayout,
                                    unrankedType.getMemorySpace());
  return rewriter.create<memref::CastOp>(loc, rankedType, source).getResult();
}

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

static void emit1DMemrefToMemrefCopyLoop(Location loc, Value srcSubview,
                                         Value dstSubview, Value upperBound,
                                         ConversionPatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.getContext()->loadDialect<vector::VectorDialect>();
  auto srcType = dyn_cast<MemRefType>(srcSubview.getType());
  auto dstType = dyn_cast<MemRefType>(dstSubview.getType());

  if (srcType && dstType && srcType.getRank() == 1 && dstType.getRank() == 1 &&
      hasUnitStride1DLayout(srcType) && hasUnitStride1DLayout(dstType)) {
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cVec = rewriter.create<arith::ConstantIndexOp>(loc, 16);

    // Vectorized main body: [0, floor(upperBound / VL) * VL) with step VL.
    Value vecIters = rewriter.create<arith::DivUIOp>(loc, upperBound, cVec);
    Value vecUpper = rewriter.create<arith::MulIOp>(loc, vecIters, cVec);
    auto vecType = VectorType::get({16}, srcType.getElementType());
    auto vecLoop = rewriter.create<scf::ForOp>(loc, c0, vecUpper, cVec);
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value vec =
        rewriter.create<vector::LoadOp>(loc, vecType, srcSubview, ValueRange{ivVec});
    rewriter.create<vector::StoreOp>(loc, vec, dstSubview, ValueRange{ivVec});

    // Scalar tail: [vecUpper, upperBound).
    rewriter.setInsertionPointAfter(vecLoop);
    auto tailLoop = rewriter.create<scf::ForOp>(loc, vecUpper, upperBound, c1);
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    Value iv = tailLoop.getInductionVar();
    Value v = rewriter.create<memref::LoadOp>(loc, srcSubview, ValueRange{iv});
    rewriter.create<memref::StoreOp>(loc, v, dstSubview, ValueRange{iv});
    return;
  }

  // Conservative fallback when we cannot build a fixed-size vector load/store.
  auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = rewriter.create<scf::ForOp>(loc, c0, upperBound, c1);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();
  Value v = rewriter.create<memref::LoadOp>(loc, srcSubview, ValueRange{iv});
  rewriter.create<memref::StoreOp>(loc, v, dstSubview, ValueRange{iv});
}

static void emit1DTensorToMemrefStoreLoop(Location loc, Value srcTensor,
                                          Value dstSubview, Value upperBound,
                                          ConversionPatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.getContext()->loadDialect<vector::VectorDialect>();
  auto srcType = dyn_cast<RankedTensorType>(srcTensor.getType());
  auto dstType = dyn_cast<MemRefType>(dstSubview.getType());
  if (srcType && dstType && srcType.getRank() == 1 && dstType.getRank() == 1 &&
      hasUnitStride1DLayout(dstType)) {
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto cVec = rewriter.create<arith::ConstantIndexOp>(loc, 16);

    // Vectorized main body: [0, floor(upperBound / VL) * VL) with step VL.
    Value vecIters = rewriter.create<arith::DivUIOp>(loc, upperBound, cVec);
    Value vecUpper = rewriter.create<arith::MulIOp>(loc, vecIters, cVec);
    auto elemType = srcType.getElementType();
    auto vecType = VectorType::get({16}, elemType);
    auto vecLoop = rewriter.create<scf::ForOp>(loc, c0, vecUpper, cVec);
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value padding = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    auto identityMap = AffineMap::getMinorIdentityMap(
        /*numDims=*/1, /*numResults=*/1, rewriter.getContext());
    const bool inBoundsArr[] = {true};
    Value vec = rewriter.create<vector::TransferReadOp>(
        loc, vecType, srcTensor, ValueRange{ivVec},
        std::optional<Value>(padding), identityMap,
        std::optional<ArrayRef<bool>>(ArrayRef<bool>(inBoundsArr)));
    rewriter.create<vector::StoreOp>(loc, vec, dstSubview, ValueRange{ivVec});

    // Scalar tail: [vecUpper, upperBound).
    rewriter.setInsertionPointAfter(vecLoop);
    auto tailLoop = rewriter.create<scf::ForOp>(loc, vecUpper, upperBound, c1);
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    Value iv = tailLoop.getInductionVar();
    Value v = rewriter.create<tensor::ExtractOp>(loc, srcTensor, ValueRange{iv});
    rewriter.create<memref::StoreOp>(loc, v, dstSubview, ValueRange{iv});
    return;
  }

  auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = rewriter.create<scf::ForOp>(loc, c0, upperBound, c1);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();
  Value v = rewriter.create<tensor::ExtractOp>(loc, srcTensor, ValueRange{iv});
  rewriter.create<memref::StoreOp>(loc, v, dstSubview, ValueRange{iv});
}

static SmallVector<Value> getDynamicTensorDims(Location loc,
                                               RankedTensorType tensorType,
                                               Value sourceMemRef,
                                               OpBuilder &b) {
  SmallVector<Value> dynamicDims;
  dynamicDims.reserve(tensorType.getNumDynamicDims());
  for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i) {
    if (!tensorType.isDynamicDim(i)) {
      continue;
    }
    dynamicDims.push_back(b.create<memref::DimOp>(loc, sourceMemRef, i));
  }
  return dynamicDims;
}

static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
  assert(!op.isBlockPtr());
  // tensor<1024x!tt.ptr<f32>>
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return ptrType.getPointeeType();
}

static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
  assert(op.isBlockPtr());
  // !tt.ptr<tensor<128x64xbf16>, 1>
  auto shapedType = cast<ShapedType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  return shapedType.getElementType();
}

static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
  Type elemType;
  if (op.isBlockPtr()) {
    elemType = getElementTypeBlockPtr(op);
  } else {
    elemType = getElementTypeStructuredPtr(op);
  }
  return MemRefType::get(resultShape, elemType, layout);
}

static MemRefType getResultMemrefType(tts::MakeGatherScatterTensorPtrOp op,
                                      int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);

  auto ptrType = cast<triton::PointerType>(op.getType());
  Type elemType = ptrType.getPointeeType();

  Type realEltTy = cast<RankedTensorType>(elemType).getElementType();
  return MemRefType::get(resultShape, realEltTy, layout);
}

// If there are dimensions with size 1 and stride 0, replace 0 stride with
// the product of sizes of all lower dimensions. This avoids creating memref
// with zero stride.
template <class OpType>
llvm::SmallVector<OpFoldResult> getMixedStridesForMemref(OpType op,
                                                         OpBuilder &b) {
  llvm::SmallVector<OpFoldResult> strides;
  auto accumulate = 1;
  for (auto [size, stride] :
       llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
    auto strideIntAttr = getIntAttr(stride);
    if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
      strides.push_back(b.getIndexAttr(accumulate));
    } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
      OpFoldResult result = getAsOpFoldResult(v);
      strides.push_back(result);
    } else {
      strides.push_back(stride);
    }
    accumulate *= size;
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : offsets) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> strides,
                                           int gatherDim, OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (int i = 0; i < offsets.size(); i++) {

    OpFoldResult offset = offsets[i];
    // If this is the gather dimension, multiply the offset by the stride.
    // Non-gather dimensions are already multiplied by the stride
    // in the offsets in PtrAnalysis.
    if (i == gatherDim) {
      OpFoldResult stride = strides[i];
      offset = mulOFRs(offset, stride, loc, b);
    }
    targetOffset = addOFRs(targetOffset, offset, loc, b);
  }
  return targetOffset;
}

static SmallVector<Value, 2> computeRank1SplitSizes(Location loc, Value start,
                                                    Value size, Value modulo,
                                                    Value stride,
                                                    OpBuilder &b) {
  // start and modulo are linear offsets; reinterpret_cast sizes are elements.
  Value remaining = b.create<arith::SubIOp>(loc, modulo, start);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value strideMinusOne = b.create<arith::SubIOp>(loc, stride, one);
  Value numerator = b.create<arith::AddIOp>(loc, remaining, strideMinusOne);
  Value elementsUntilWrap = b.create<arith::DivSIOp>(loc, numerator, stride);
  Value d1 = b.create<arith::MinSIOp>(loc, size, elementsUntilWrap);
  Value d2 = b.create<arith::SubIOp>(loc, size, d1);
  return {d1, d2};
}

static FailureOr<Value> materializeStructuredTPtrMemRef(tts::MakeTensorPtrOp op,
                                                        Location loc,
                                                        OpBuilder &rewriter) {
  if (!op.isStructuredPtr()) {
    return failure();
  }

  Value base = op.getBase();
  auto elementType = getElementTypeStructuredPtr(op);
  if (!isa<MemRefType, UnrankedMemRefType>(base.getType())) {
    if (!isa<triton::PointerType>(base.getType())) {
      return failure();
    }
    auto unrankedType = UnrankedMemRefType::get(elementType, 0);
    base = rewriter
               .create<UnrealizedConversionCastOp>(loc, unrankedType, base)
               .getResult(0);
  }

  auto mixedStrides = getMixedStridesForMemref(op, rewriter);
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  auto targetOffset = accumulateTargetOffset(loc, op.getMixedOffsets(), rewriter);
  auto staticTargetOffset = getIntAttr(targetOffset);
  ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
  auto resultType = getResultMemrefType(
      op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
      resultShape);

  return rewriter
      .create<memref::ReinterpretCastOp>(loc, resultType, base, targetOffset,
                                         op.getMixedSizes(), mixedStrides)
      .getResult();
}

static FailureOr<Value> materializeSplitTPtrMemRef(tts::MakeTensorPtrOp op,
                                                   Location loc,
                                                   OpBuilder &rewriter) {
  if (!op.isSplitPtr()) {
    return failure();
  }

  Value base = op.getBase();
  auto elementType = getElementTypeStructuredPtr(op);
  if (!isa<MemRefType, UnrankedMemRefType>(base.getType())) {
    if (!isa<triton::PointerType>(base.getType())) {
      return failure();
    }
    auto unrankedType = UnrankedMemRefType::get(elementType, 0);
    base = rewriter
               .create<UnrealizedConversionCastOp>(loc, unrankedType, base)
               .getResult(0);
  }

  auto resultShape = cast<RankedTensorType>(op.getType()).getShape();
  auto targetOffset = ofrToIndexValue(
      accumulateTargetOffset(loc, op.getMixedOffsets(), rewriter), loc,
      rewriter);
  auto parentShape = op.getStaticShape();

  auto isSplitDimension = [](int64_t dim) {
    return dim == ShapedType::kDynamic || dim != 0;
  };

  SmallVector<Value> strideVals =
      ofrsToIndexValues(getMixedStridesForMemref(op, rewriter), loc, rewriter);

  if (resultShape.size() == 1) {
    if (parentShape.size() != 1) {
      return failure();
    }
    if (!isSplitDimension(parentShape[0])) {
      return materializeStructuredTPtrMemRef(op, loc, rewriter);
    }

    auto resultType = getResultMemrefType(
        op, /*offset=*/ShapedType::kDynamic,
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        SmallVector<int64_t>{ShapedType::kDynamic});
    Value size = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value modulo = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);
    Value start = rewriter.create<arith::RemSIOp>(loc, targetOffset, modulo);
    auto splitSizes = computeRank1SplitSizes(loc, start, size, modulo,
                                             strideVals[0], rewriter);
    Value d1 = splitSizes[0];
    Value d2 = splitSizes[1];
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> casts;
    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, start, ValueRange{d1},
                            ValueRange{strideVals[0]})
                        .getResult());
    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, zero, ValueRange{d2},
                            ValueRange{strideVals[0]})
                        .getResult());

    auto combinedCast =
        rewriter.create<UnrealizedConversionCastOp>(loc, op.getType(), casts);
    combinedCast->setAttr(WRAP_LINEAR, rewriter.getUnitAttr());
    return combinedCast.getResult(0);
  }

  if (resultShape.size() != 2 || parentShape.size() != 2) {
    return failure();
  }

  auto resultType = getResultMemrefType(
      op, /*offset=*/ShapedType::kDynamic,
      SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
      SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic});
  Value rowSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexAttr(op.getSizes()[0]));
  Value colSize = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexAttr(op.getSizes()[1]));

  SmallVector<Value> casts;
  StringRef wrapType;
  if (isSplitDimension(parentShape[0])) {
    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);
    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);
    d1 = rewriter.create<arith::MinSIOp>(loc, d1, rowSize);
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);

    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, targetOffset,
                            ValueRange{d1, colSize},
                            ValueRange{strideRow, strideCol})
                        .getResult());
    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, wrappedAroundOff,
                            ValueRange{d2, colSize},
                            ValueRange{strideRow, strideCol})
                        .getResult());
    wrapType = WRAP_STACKED;
  } else if (isSplitDimension(parentShape[1])) {
    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);
    Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);

    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, targetOffset,
                            ValueRange{rowSize, d1}, strideVals)
                        .getResult());
    casts.push_back(rewriter
                        .create<memref::ReinterpretCastOp>(
                            loc, resultType, base, y, ValueRange{rowSize, d2},
                            strideVals)
                        .getResult());
    wrapType = WRAP_SIDE_BY_SIDE;
  } else {
    return failure();
  }

  auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
      loc, op.getType(), casts);
  combinedCast->setAttr(wrapType, rewriter.getUnitAttr());
  return combinedCast.getResult(0);
}

static UnrealizedConversionCastOp getWraparoundCast(Value ptr) {
  auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>();
  if (!castOp) {
    return nullptr;
  }
  if (!castOp->hasAttr(WRAP_SIDE_BY_SIDE) && !castOp->hasAttr(WRAP_STACKED) &&
      !castOp->hasAttr(WRAP_LINEAR)) {
    return nullptr;
  }
  return castOp;
}

static Value rewriteGatherScatterPtrElement(
    ArrayRef<int64_t> resultShape, tts::MakeGatherScatterTensorPtrOp op,
    Value basePtr, Value gatherOffsetElt, int gatherDim,
    ConversionPatternRewriter &rewriter) {

  auto mixedStrides = getMixedStridesForMemref(op, rewriter);
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  auto offsets = op.getMixedOffsets();
  offsets[gatherDim] = gatherOffsetElt;
  auto targetOffset = accumulateTargetOffset(op.getLoc(), offsets, mixedStrides,
                                             gatherDim, rewriter);

  auto staticTargetOffset = getIntAttr(targetOffset);
  auto resultType =
      getResultMemrefType(op, staticTargetOffset.value_or(ShapedType::kDynamic),
                          staticStrides, resultShape);

  std::vector<int64_t> staticSizes = op.getSizes();
  staticSizes[gatherDim] = 1;
  SmallVector<Value> dynSizes; // sizes are always static
  auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

  auto castOp = rewriter.create<memref::ReinterpretCastOp>(
      op.getLoc(), resultType, basePtr, targetOffset, sizes, mixedStrides);

  return castOp.getResult();
}

// Fill load destination with other value for mask.
static void fillWithValue(Location loc, Value alloc, Value other,
                          ArrayRef<int64_t> shape,
                          SmallVector<OpFoldResult> &&mixedDims,
                          ArrayRef<int64_t> staticMaskDims,
                          ConversionPatternRewriter &rewriter) {
  // Fill load destination with other value
  // For each dimension check if dims[i] < shape[i], or-accumulate
  // the result
  auto accBase =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
          .getResult();
  for (size_t i = 0; i < shape.size(); i++) {
    auto shapei = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(shape[i]));

    Value dimi = dyn_cast<Value>(mixedDims[i]);
    if (!dimi) {
      dimi = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(staticMaskDims[i]));
    }

    Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               dimi, shapei);
    accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmp);
  }

  // condition the memset on the or-accumulation
  // initialize with padding prior to CopyOp
  rewriter.create<scf::IfOp>(loc, accBase, [&](OpBuilder &b, Location loc) {
    b.create<linalg::FillOp>(loc, ValueRange{other}, ValueRange{alloc});
    b.create<scf::YieldOp>(loc);
  });
}

namespace {

enum class MaskedReduceKind {
  AddF,
  AddI,
  MaximumF,
  MaxNumF,
  MaxSI,
  MaxUI,
};

static Operation *getSingleReduceBodyOp(linalg::ReduceOp op) {
  Block &block = op->getRegion(0).front();
  auto bodyOps = block.without_terminator();
  if (!llvm::hasSingleElement(bodyOps)) {
    return nullptr;
  }
  return &*bodyOps.begin();
}

static std::optional<MaskedReduceKind>
matchMaskedReduceKind(linalg::ReduceOp op) {
  Operation *bodyOp = getSingleReduceBodyOp(op);
  if (!bodyOp) {
    return std::nullopt;
  }
  if (isa<arith::AddFOp>(bodyOp)) {
    return MaskedReduceKind::AddF;
  }
  if (isa<arith::AddIOp>(bodyOp)) {
    return MaskedReduceKind::AddI;
  }
  if (isa<arith::MaximumFOp>(bodyOp)) {
    return MaskedReduceKind::MaximumF;
  }
  if (isa<arith::MaxNumFOp>(bodyOp)) {
    return MaskedReduceKind::MaxNumF;
  }
  if (isa<arith::MaxSIOp>(bodyOp)) {
    return MaskedReduceKind::MaxSI;
  }
  if (isa<arith::MaxUIOp>(bodyOp)) {
    return MaskedReduceKind::MaxUI;
  }
  return std::nullopt;
}

static Value createMaskedReduceInit(Location loc, MaskedReduceKind kind,
                                    Type resultType,
                                    PatternRewriter &rewriter) {
  switch (kind) {
  case MaskedReduceKind::AddF:
    return rewriter
        .create<arith::ConstantOp>(loc, resultType,
                                   rewriter.getFloatAttr(resultType, 0.0))
        .getResult();
  case MaskedReduceKind::AddI:
  case MaskedReduceKind::MaxUI:
    return rewriter
        .create<arith::ConstantOp>(loc, resultType,
                                   rewriter.getIntegerAttr(resultType, 0))
        .getResult();
  case MaskedReduceKind::MaximumF:
  case MaskedReduceKind::MaxNumF:
    return rewriter
        .create<arith::ConstantOp>(
            loc, resultType,
            rewriter.getFloatAttr(
                resultType, -std::numeric_limits<float>::infinity()))
        .getResult();
  case MaskedReduceKind::MaxSI:
    return rewriter
        .create<arith::ConstantOp>(
            loc, resultType,
            rewriter.getIntegerAttr(resultType,
                                    llvm::minIntN(resultType.getIntOrFloatBitWidth())))
        .getResult();
  }
  llvm_unreachable("unsupported masked reduce kind");
}

static Value castMaskedReduceInput(Location loc, Value value, Type resultType,
                                   MaskedReduceKind kind,
                                   PatternRewriter &rewriter) {
  if (value.getType() == resultType) {
    return value;
  }
  if (kind == MaskedReduceKind::AddF && isa<FloatType>(value.getType()) &&
      isa<FloatType>(resultType)) {
    return rewriter.create<arith::ExtFOp>(loc, resultType, value);
  }
  llvm_unreachable("unexpected masked reduce type mismatch");
}

static Value combineMaskedReduceValue(Location loc, MaskedReduceKind kind,
                                      Value input, Value acc, Type resultType,
                                      PatternRewriter &rewriter) {
  input = castMaskedReduceInput(loc, input, resultType, kind, rewriter);
  switch (kind) {
  case MaskedReduceKind::AddF:
    return rewriter.create<arith::AddFOp>(loc, input, acc).getResult();
  case MaskedReduceKind::AddI:
    return rewriter.create<arith::AddIOp>(loc, input, acc).getResult();
  case MaskedReduceKind::MaximumF:
    return rewriter.create<arith::MaximumFOp>(loc, input, acc).getResult();
  case MaskedReduceKind::MaxNumF:
    return rewriter.create<arith::MaxNumFOp>(loc, input, acc).getResult();
  case MaskedReduceKind::MaxSI:
    return rewriter.create<arith::MaxSIOp>(loc, input, acc).getResult();
  case MaskedReduceKind::MaxUI:
    return rewriter.create<arith::MaxUIOp>(loc, input, acc).getResult();
  }
  llvm_unreachable("unsupported masked reduce kind");
}

static Value combineMaskedReduceValue(Location loc, MaskedReduceKind kind,
                                      Value input, Value acc,
                                      PatternRewriter &rewriter) {
  return combineMaskedReduceValue(loc, kind, input, acc, acc.getType(),
                                  rewriter);
}

static std::optional<vector::CombiningKind>
getMaskedReduceCombiningKind(MaskedReduceKind kind) {
  switch (kind) {
  case MaskedReduceKind::AddF:
  case MaskedReduceKind::AddI:
    return vector::CombiningKind::ADD;
  case MaskedReduceKind::MaximumF:
    return vector::CombiningKind::MAXIMUMF;
  case MaskedReduceKind::MaxNumF:
    return vector::CombiningKind::MAXNUMF;
  case MaskedReduceKind::MaxSI:
    return vector::CombiningKind::MAXSI;
  case MaskedReduceKind::MaxUI:
    return vector::CombiningKind::MAXUI;
  }
  return std::nullopt;
}

struct MaskedReduceFusionPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getInits().size() != 1 ||
        op->getNumResults() != 1) {
      return failure();
    }
    if (op.getDimensions().size() != 1 || op.getDimensions()[0] != 0) {
      return failure();
    }

    auto kind = matchMaskedReduceKind(op);
    if (!kind) {
      return failure();
    }

    auto load = op.getInputs()[0].getDefiningOp<tts::LoadOp>();
    if (!load || !load.hasMask() || !load.getOther() || !load->hasOneUse()) {
      return failure();
    }

    auto loadType = dyn_cast<RankedTensorType>(load.getType());
    if (!loadType || loadType.getRank() != 1 || !loadType.hasStaticShape()) {
      return failure();
    }

    if (!op->getResult(0).hasOneUse()) {
      return failure();
    }
    auto extract =
        dyn_cast<tensor::ExtractOp>(*op->getResult(0).user_begin());
    if (!extract || !extract.getIndices().empty()) {
      return failure();
    }

    auto ptr = load.getPtr();
    auto makeTPtr = ptr.getDefiningOp<tts::MakeTensorPtrOp>();
    if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType())) {
      if (!makeTPtr) {
        return failure();
      }
      auto materialized =
          materializeStructuredTPtrMemRef(makeTPtr, load.getLoc(), rewriter);
      if (failed(materialized)) {
        return failure();
      }
      ptr = *materialized;
    }

    auto rankedPtr =
        ensureRankedMemRef(ptr, /*rank=*/1, loadType.getElementType(),
                           load.getLoc(), rewriter);
    if (failed(rankedPtr)) {
      return failure();
    }
    ptr = *rankedPtr;

    auto memrefType = dyn_cast<MemRefType>(ptr.getType());
    if (!memrefType || memrefType.getRank() != 1 ||
        !hasUnitStride1DLayout(memrefType)) {
      return failure();
    }

    auto vectorKind = getMaskedReduceCombiningKind(*kind);
    if (!vectorKind) {
      return failure();
    }

    Location loc = op.getLoc();
    Type resultType = extract.getType();
    Value accInit = createMaskedReduceInit(loc, *kind, resultType, rewriter);
    Value validLen =
        ofrToIndexValue(load.getMixedMaskDims()[0], loc, rewriter);
    Value other = load.getOther();
    int64_t fullSize = loadType.getShape()[0];

    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    auto cFull = rewriter.create<arith::ConstantIndexOp>(loc, fullSize);

    Value vecIters = rewriter.create<arith::DivUIOp>(loc, validLen, c16);
    Value vecUpper = rewriter.create<arith::MulIOp>(loc, vecIters, c16);
    auto vecType = VectorType::get({16}, loadType.getElementType());

    auto vecLoop =
        rewriter.create<scf::ForOp>(loc, c0, vecUpper, c16, ValueRange{accInit});
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value accVec = vecLoop.getRegionIterArgs().front();
    Value vec = rewriter.create<vector::LoadOp>(loc, vecType, ptr,
                                                ValueRange{ivVec});
    Value vecReduced = rewriter.create<vector::ReductionOp>(
        loc, *vectorKind, vec, accVec);
    rewriter.create<scf::YieldOp>(loc, vecReduced);

    rewriter.setInsertionPointAfter(vecLoop);
    auto scalarLoop = rewriter.create<scf::ForOp>(
        loc, vecUpper, validLen, c1, ValueRange{vecLoop.getResult(0)});
    rewriter.setInsertionPointToStart(scalarLoop.getBody());
    Value iv = scalarLoop.getInductionVar();
    Value accScalar = scalarLoop.getRegionIterArgs().front();
    Value elem =
        rewriter.create<memref::LoadOp>(loc, ptr, ValueRange{iv});
    Value next = combineMaskedReduceValue(loc, *kind, elem, accScalar,
                                          resultType, rewriter);
    rewriter.create<scf::YieldOp>(loc, next);

    rewriter.setInsertionPointAfter(scalarLoop);
    auto paddingLoop = rewriter.create<scf::ForOp>(
        loc, validLen, cFull, c1, ValueRange{scalarLoop.getResult(0)});
    rewriter.setInsertionPointToStart(paddingLoop.getBody());
    Value accPadding = paddingLoop.getRegionIterArgs().front();
    Value nextPadding = combineMaskedReduceValue(loc, *kind, other, accPadding,
                                                 resultType, rewriter);
    rewriter.create<scf::YieldOp>(loc, nextPadding);

    rewriter.replaceOp(extract, paddingLoop.getResult(0));
    rewriter.eraseOp(op);
    if (load->use_empty()) {
      rewriter.eraseOp(load);
    }
    if (makeTPtr && makeTPtr->use_empty()) {
      rewriter.eraseOp(makeTPtr);
    }
    return success();
  }
};

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{

            // Row stays the same, but mlir doesn't allow this anymore. Put
            // dynamic.
            ShapedType::kDynamic,

            // Column is dynamic, in most cases, this
            // should be the same as the original column.
            // The last chunk may be smaller due to
            // wrapping around.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);
    SmallVector<Value> sizes1{rowSize, d1};

    auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), targetOffset, sizes1, strideVals);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);
    SmallVector<Value> sizes2{rowSize, d2};

    auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), y, sizes2, strideVals);

    return {cast1, cast2};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*---------------------
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |             |            |       |
    //  rows|    rowSize  |            |       |
    //      |             |            |       |
    //      |             |            |       |
    //      |             *------------|       |
    //      |          nextOff                 |
    //      |                                  |
    //      |          clampedOff              |
    //      --------------*---------------------
    //
    //    For the case that clampedOff is not overflown
    //    d1 = min(d1, rowSize)
    //

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{
            // Row is dynamic, in most cases, this should
            // be the same as the original row. The last
            // chunk may be smaller due to wrapping
            // around.
            ShapedType::kDynamic,

            // Col stays the same, which is resultShape[1], but mlir doesn't
            // allow this anymore. So we put dynamic instead.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    // First chunk
    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);
    d1 = rewriter.create<arith::MinSIOp>(loc, d1, rowSize);

    SmallVector<Value> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), targetOffset, sizes1,
            ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
    SmallVector<Value> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), wrappedAroundOff, sizes2,
            ValueRange{strideRow, strideCol});

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto parentShape = op.getStaticShape();
    SmallVector<Value> casts;
    StringRef wrapType;

    // For split pointers, a split dimension is either a dynamic or a non-zero
    // value. The other dimension must be zero.
    auto isSplitDimension = [](int64_t dim) {
      return dim == ShapedType::kDynamic || dim != 0;
    };

    if (parentShape.size() == 1) {
      if (!isSplitDimension(parentShape[0])) {
        return rewriteStructuredPtr(op, adaptor, rewriter);
      }

      auto loc = op->getLoc();
      auto resultShape = cast<RankedTensorType>(op.getType()).getShape();
      auto resultType = getResultMemrefType(
          op, /*offset=*/ShapedType::kDynamic,
          SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
          SmallVector<int64_t>{ShapedType::kDynamic});
      Value targetOffset = ofrToIndexValue(
          accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
          loc, rewriter);
      Value size = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(op.getSizes()[0]));
      Value stride = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
      Value modulo = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);
      Value start = rewriter.create<arith::RemSIOp>(loc, targetOffset, modulo);
      auto splitSizes =
          computeRank1SplitSizes(loc, start, size, modulo, stride, rewriter);
      Value d1 = splitSizes[0];
      Value d2 = splitSizes[1];
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, adaptor.getBase(), start, ValueRange{d1},
          ValueRange{stride});
      auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, adaptor.getBase(), zero, ValueRange{d2},
          ValueRange{stride});
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_LINEAR;
    } else if (parentShape.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "split pointer lowering supports only rank-1 and rank-2 shapes");
    } else if (isSplitDimension(parentShape[0])) {
      // Stacked case
      assert(parentShape[1] == 0);
      auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_STACKED;
    } else if (isSplitDimension(parentShape[1])) {
      assert(parentShape[0] == 0);
      auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_SIDE_BY_SIDE;
    } else {
      llvm_unreachable("Unexpected split pointer shape");
    }

    auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast);

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset =
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, adaptor.getBase(), targetOffset,
        op.getMixedSizes(), mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }

    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

struct MakeGatherScatterTensorPtrConverter
    : public OpConversionPattern<tts::MakeGatherScatterTensorPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::MakeGatherScatterTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The gatherScatterPtr is rewritten as separate rows during load/store
    // operations. Therefore, no action is needed here except saving
    // adaptor.getBase(). DialectConversion will ignore pure type conversion if
    // we were to simply replace the op with adaptor.getBase(). To circumvent
    // this we create an identity cast.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, adaptor.getBase().getType(), adaptor.getBase());
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  bool enableTensorFirstVectorCpu;

  bool isTensorFirstFastPathCandidate(tts::LoadOp op, Value ptr) const {
    if (!enableTensorFirstVectorCpu) {
      return false;
    }

    auto ptrDefiningOp = ptr.getDefiningOp();
    if (ptrDefiningOp &&
        (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
         ptrDefiningOp->hasAttr(WRAP_STACKED) ||
         ptrDefiningOp->hasAttr(WRAP_LINEAR) ||
         isa<tts::MakeGatherScatterTensorPtrOp>(ptrDefiningOp))) {
      return false;
    }

    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    auto memrefType = dyn_cast<MemRefType>(ptr.getType());
    if (!tensorType || !memrefType) {
      return false;
    }
    if (tensorType.getElementType() != memrefType.getElementType()) {
      return false;
    }
    return staticShapeCompatible(tensorType, memrefType);
  }

  Value createTensorFromMemref(tts::LoadOp op, Value source,
                               RankedTensorType targetTensorType,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto sourceType = cast<MemRefType>(source.getType());
    auto dynamicTensorType =
        RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, dynamicTensorType, source, true /*restrict*/,
        false /*writable*/);
    if (tensor.getType() != targetTensorType) {
      tensor = rewriter.create<tensor::CastOp>(loc, targetTensorType, tensor);
    }
    return tensor;
  }

  bool needsTensorFirstLoadSnapshot(tts::LoadOp op) const {
    Value loadResult = op.getResult();
    Operation *loadOperation = op.getOperation();
    Operation *lastUseInBlock = nullptr;

    for (Operation *it = loadOperation->getNextNode(); it;
         it = it->getNextNode()) {
      for (Value operand : it->getOperands()) {
        if (operand == loadResult) {
          lastUseInBlock = it;
          break;
        }
      }
    }

    if (!lastUseInBlock) {
      return false;
    }

    Value loadPtr = op.getPtr();
    for (Operation *it = loadOperation->getNextNode();
         it && it != lastUseInBlock; it = it->getNextNode()) {
      if (auto store = dyn_cast<tts::StoreOp>(it)) {
        if (store.getPtr() == loadPtr) {
          return true;
        }
      }
    }

    return false;
  }

  LogicalResult rewriteTensorFirstUnmaskedLoad(
      tts::LoadOp op, Value ptr, ConversionPatternRewriter &rewriter) const {
    auto tensorType = cast<RankedTensorType>(op.getType());

    if (!needsTensorFirstLoadSnapshot(op)) {
      Value tensor = createTensorFromMemref(op, ptr, tensorType, rewriter);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    auto loc = op->getLoc();
    auto ptrType = cast<MemRefType>(ptr.getType());
    auto allocType = MemRefType::get(ptrType.getShape(), ptrType.getElementType());
    SmallVector<Value> dynamicDims;
    dynamicDims.reserve(ptrType.getNumDynamicDims());
    for (int64_t i = 0, e = ptrType.getRank(); i < e; ++i) {
      if (ptrType.isDynamicDim(i)) {
        dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, ptr, i));
      }
    }
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType, dynamicDims);
    rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    Value tensor = createTensorFromMemref(op, alloc, tensorType, rewriter);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  LogicalResult rewriteTensorFirstMaskedLoad(
      tts::LoadOp op, Value ptr, ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    int64_t rank = tensorType.getRank();
    if (rank == 1) {
      auto alloc = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(tensorType.getShape(), tensorType.getElementType()));

      if (Value other = op.getOther()) {
        fillWithValue(loc, alloc, other, tensorType.getShape(),
                      op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
      }

      SmallVector<OpFoldResult> copyDims = op.getMixedMaskDims();
      auto srcSubview = getSubview(rank, copyDims, ptr, loc, rewriter);
      auto dstSubview = getSubview(rank, copyDims, alloc, loc, rewriter);
      Value copyLen = ofrToIndexValue(copyDims[0], loc, rewriter);
      emit1DMemrefToMemrefCopyLoop(loc, srcSubview, dstSubview, copyLen,
                                   rewriter);

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /*restrict*/, true /*writable*/);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto srcSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);
    auto sliceType = cast<RankedTensorType>(tensor::ExtractSliceOp::inferResultType(
        tensorType, offsets, mixedDims, strides));
    Value loadedSlice = createTensorFromMemref(op, srcSubview, sliceType, rewriter);

    auto dynamicResultDims = getDynamicTensorDims(loc, tensorType, ptr, rewriter);
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType(), dynamicResultDims);
    if (Value other = op.getOther()) {
      init = rewriter
                 .create<linalg::FillOp>(loc, ValueRange{other}, ValueRange{init})
                 .getResult(0);
    }

    Value tensor = rewriter.create<tensor::InsertSliceOp>(
        loc, loadedSlice, init, offsets, mixedDims, strides);
    rewriter.replaceOp(op, tensor);
    return success();
  }

private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createLinearCopies(Value block1, Value block2, Value dst, Location loc,
                          ConversionPatternRewriter &rewriter) const {
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value block1Size = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block2Size = rewriter.create<memref::DimOp>(loc, block2, 0);

    auto block1Dst = rewriter.create<memref::SubViewOp>(
        loc, dst, ValueRange{zero}, ValueRange{block1Size}, ValueRange{one});
    auto block2Dst = rewriter.create<memref::SubViewOp>(
        loc, dst, ValueRange{block1Size}, ValueRange{block2Size},
        ValueRange{one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                              src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult subviewCol1 =
        rewriter.create<memref::DimOp>(loc, block1, 1).getResult();
    OpFoldResult subviewCol2 =
        rewriter.create<memref::DimOp>(loc, block2, 1).getResult();

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     const Location loc,
                     ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult subviewRow1 =
        rewriter.create<memref::DimOp>(loc, block1, 0).getResult();
    OpFoldResult subviewRow2 =
        rewriter.create<memref::DimOp>(loc, block2, 0).getResult();

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getLinearSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                    Location loc, ConversionPatternRewriter &rewriter) const {
    Value requested = ofrToIndexValue(dims[0], loc, rewriter);
    Value block1Size = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block2Size = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value size1 = rewriter.create<arith::MinSIOp>(loc, requested, block1Size);
    Value remaining = rewriter.create<arith::SubIOp>(loc, requested, size1);
    Value size2 = rewriter.create<arith::MinSIOp>(loc, remaining, block2Size);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<OpFoldResult> zeroOffset{zero};
    SmallVector<OpFoldResult> size1Ofr{size1};
    SmallVector<OpFoldResult> size2Ofr{size2};
    SmallVector<OpFoldResult> oneStride{one};
    auto sv1 =
        createSubview(block1, zeroOffset, size1Ofr, oneStride, loc, rewriter);
    auto sv2 =
        createSubview(block2, zeroOffset, size2Ofr, oneStride, loc, rewriter);
    return {sv1, sv2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getSideBySideSlices(Value source, Value block1, Value block2, Location loc,
                      ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rowSize = rewriter.create<tensor::DimOp>(loc, source, 0);
    Value colSize1 = rewriter.create<memref::DimOp>(loc, block1, 1);
    Value colSize2 = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, c0}, ValueRange{rowSize, colSize1},
        ValueRange{c1, c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, colSize1}, ValueRange{rowSize, colSize2},
        ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getStackedSlices(Value source, Value block1, Value block2, Location loc,
                   ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rowSize1 = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value rowSize2 = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value colSize = rewriter.create<tensor::DimOp>(loc, source, 1);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, c0}, ValueRange{rowSize1, colSize},
        ValueRange{c1, c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{rowSize1, c0}, ValueRange{rowSize2, colSize},
        ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getLinearSlices(Value source, Value block1, Value block2, Location loc,
                  ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value size1 = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value size2 = rewriter.create<memref::DimOp>(loc, block2, 0);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0}, ValueRange{size1}, ValueRange{c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{size1}, ValueRange{size2}, ValueRange{c1});
    return {slice1, slice2};
  }

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, Value ptr,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();
    if (auto unrealizedCast = getWraparoundCast(ptr)) {
      auto alloc = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(tensorType.getShape(), elemType));
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      assert(!other && "other value used in non-masked load");
      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        createSideBySideCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        createStackedCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        createLinearCopies(block1, block2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    auto rankedPtr =
        ensureRankedMemRef(ptr, tensorType.getRank(), elemType, loc, rewriter);
    if (failed(rankedPtr)) {
      return rewriter.notifyMatchFailure(
          op, "expected pointer to lower to ranked/unranked memref");
    }
    ptr = *rankedPtr;

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    // No mask
    assert(!other && "other value used in non-masked load");

    rewriter.create<memref::CopyOp>(loc, ptr, alloc);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, Value ptr,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();
    if (auto unrealizedCast = getWraparoundCast(ptr)) {
      auto mixedDims = op.getMixedMaskDims();
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];
      SmallVector<Value> dynamicResultDims;
      dynamicResultDims.reserve(tensorType.getNumDynamicDims());
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), elemType, dynamicResultDims);
      if (Value other = op.getOther()) {
        init = rewriter
                   .create<linalg::FillOp>(loc, ValueRange{other}, ValueRange{init})
                   .getResult(0);
      }
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, block1, block2, loc, rewriter);
        auto subview1Type = cast<MemRefType>(subview1.getType());
        auto subview2Type = cast<MemRefType>(subview2.getType());
        auto slice1Type =
            RankedTensorType::get(subview1Type.getShape(), elemType);
        auto slice2Type =
            RankedTensorType::get(subview2Type.getShape(), elemType);
        auto genericSliceType =
            RankedTensorType::get(SmallVector<int64_t>{ShapedType::kDynamic,
                                                       ShapedType::kDynamic},
                                  elemType);
        Value tensor1 = createTensorFromMemref(op, subview1, slice1Type, rewriter);
        Value tensor2 = createTensorFromMemref(op, subview2, slice2Type, rewriter);
        tensor1 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor1);
        tensor2 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor2);
        Value rowSize = rewriter.create<memref::DimOp>(loc, subview1, 0);
        Value colSize1 = rewriter.create<memref::DimOp>(loc, subview1, 1);
        Value colSize2 = rewriter.create<memref::DimOp>(loc, subview2, 1);
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor1, init, ValueRange{c0, c0},
            ValueRange{rowSize, colSize1}, ValueRange{c1, c1});
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor2, init, ValueRange{c0, colSize1},
            ValueRange{rowSize, colSize2}, ValueRange{c1, c1});
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, block1, block2, loc, rewriter);
        auto subview1Type = cast<MemRefType>(subview1.getType());
        auto subview2Type = cast<MemRefType>(subview2.getType());
        auto slice1Type =
            RankedTensorType::get(subview1Type.getShape(), elemType);
        auto slice2Type =
            RankedTensorType::get(subview2Type.getShape(), elemType);
        auto genericSliceType =
            RankedTensorType::get(SmallVector<int64_t>{ShapedType::kDynamic,
                                                       ShapedType::kDynamic},
                                  elemType);
        Value tensor1 = createTensorFromMemref(op, subview1, slice1Type, rewriter);
        Value tensor2 = createTensorFromMemref(op, subview2, slice2Type, rewriter);
        tensor1 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor1);
        tensor2 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor2);
        Value rowSize1 = rewriter.create<memref::DimOp>(loc, subview1, 0);
        Value rowSize2 = rewriter.create<memref::DimOp>(loc, subview2, 0);
        Value colSize = rewriter.create<memref::DimOp>(loc, subview1, 1);
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor1, init, ValueRange{c0, c0},
            ValueRange{rowSize1, colSize}, ValueRange{c1, c1});
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor2, init, ValueRange{rowSize1, c0},
            ValueRange{rowSize2, colSize}, ValueRange{c1, c1});
      } else if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        auto [subview1, subview2] =
            getLinearSubviews(mixedDims, block1, block2, loc, rewriter);
        auto subview1Type = cast<MemRefType>(subview1.getType());
        auto subview2Type = cast<MemRefType>(subview2.getType());
        auto slice1Type =
            RankedTensorType::get(subview1Type.getShape(), elemType);
        auto slice2Type =
            RankedTensorType::get(subview2Type.getShape(), elemType);
        auto genericSliceType = RankedTensorType::get(
            SmallVector<int64_t>{ShapedType::kDynamic}, elemType);
        Value tensor1 =
            createTensorFromMemref(op, subview1, slice1Type, rewriter);
        Value tensor2 =
            createTensorFromMemref(op, subview2, slice2Type, rewriter);
        tensor1 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor1);
        tensor2 =
            rewriter.create<tensor::CastOp>(loc, genericSliceType, tensor2);
        Value size1 = rewriter.create<memref::DimOp>(loc, subview1, 0);
        Value size2 = rewriter.create<memref::DimOp>(loc, subview2, 0);
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor1, init, ValueRange{c0}, ValueRange{size1},
            ValueRange{c1});
        init = rewriter.create<tensor::InsertSliceOp>(
            loc, tensor2, init, ValueRange{size1}, ValueRange{size2},
            ValueRange{c1});
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      rewriter.replaceOp(op, init);
      return success();
    }

    auto rankedPtr =
        ensureRankedMemRef(ptr, tensorType.getRank(), elemType, loc, rewriter);
    if (failed(rankedPtr)) {
      return rewriter.notifyMatchFailure(
          op, "expected pointer to lower to ranked/unranked memref");
    }
    ptr = *rankedPtr;

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    // Fill load destination with other value
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, tensorType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    memref::SubViewOp srcSubview =
        getSubview(tensorType.getRank(), mixedDims, ptr, loc, rewriter);
    memref::SubViewOp dstSubview =
        getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteGather(tts::MakeGatherScatterTensorPtrOp ptr,
                              tts::LoadOp op, Value memRefPtr,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        rewriter.create<arith::IndexCastOp>(loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        if (gatherMaskDimValue == 0 && ptr.getGatherScatterMask()) {
          // For unstructured mask case, loop over all elements and use the
          // unstructured mask to guard the store.
          gatherMaskDimValue = offsetSize;
        }
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = rewriter.create<arith::ConstantIndexOp>(loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, offsetSize);
        upperBound =
            rewriter
                .create<arith::MinSIOp>(loc, gatherMaskDimVal, offsetSizeVal)
                .getResult();
      }
    }
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    if (resultType.getRank() == 1) {
      SmallVector<Value> dynamicDims;
      if (resultType.isDynamicDim(0)) {
        dynamicDims.push_back(
            rewriter.create<tensor::DimOp>(loc, gatherOffset, 0));
      }
      Value result = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType(), dynamicDims);
      if (Value other = op.getOther()) {
        result = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{other},
                                             ValueRange{result})
                     .getResult(0);
      }

      auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                              ValueRange{result});
      rewriter.setInsertionPointToStart(loop.getBody());
      Value inductionVar = loop.getInductionVar();
      Value tensorAcc = loop.getRegionIterArgs()[0];
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      auto buildInsert = [&](OpBuilder &builder, Value acc) -> Value {
        auto gatherOffsetElt = builder.create<tensor::ExtractOp>(
            loc, gatherOffset, ValueRange{inductionVar});
        Value srcPtr = rewriteGatherScatterPtrElement(
            staticSizes, ptr, memRefPtr, gatherOffsetElt.getResult(), gatherDim,
            rewriter);
        Value loaded =
            builder.create<memref::LoadOp>(loc, srcPtr, ValueRange{zero});
        return builder
            .create<tensor::InsertOp>(loc, loaded, acc,
                                      ValueRange{inductionVar})
            .getResult();
      };

      Value nextTensor;
      if (Value unstructuredMask = ptr.getGatherScatterMask()) {
        auto maskValue = rewriter.create<tensor::ExtractOp>(
            loc, unstructuredMask, ValueRange{inductionVar});
        auto ifOp =
            rewriter.create<scf::IfOp>(loc, tensorAcc.getType(), maskValue,
                                       /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          rewriter.create<scf::YieldOp>(loc, buildInsert(rewriter, tensorAcc));
        }
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          rewriter.create<scf::YieldOp>(loc, tensorAcc);
        }
        nextTensor = ifOp.getResult(0);
      } else {
        nextTensor = buildInsert(rewriter, tensorAcc);
      }
      rewriter.create<scf::YieldOp>(loc, nextTensor);
      rewriter.replaceOp(op, loop.getResult(0));

      return success();
    }

    // Create alloc to save the result.
    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType);

    // Fill load destination with other value.
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, resultType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // Create tensor from alloc and use it as the result to replace op.
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, op.getType(), alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // load.
      auto maskValue = rewriter.create<tensor::ExtractOp>(
          loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = rewriter.create<tensor::ExtractOp>(
        loc, gatherOffset, ValueRange{inductionVar});

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value srcPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    unsigned rank = ptr.getSizes().size();
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    // subview from srcPtr for mask.
    // With offsets[gatherDim] set to 0 since the offset already in
    // reinterpret_cast. With sizes[gatherDim] set to 1 since we are load one
    // row each time.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      // Use oneStrides for subview.
      auto dstSubViewType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(srcPtr.getType()), maskOffsets, sizes, oneStrides);
      srcPtr =
          rewriter
              .create<memref::SubViewOp>(loc, cast<MemRefType>(dstSubViewType),
                                         srcPtr, maskOffsets, sizes, oneStrides)
              .getResult();
    }

    // alloc[inductionVar]
    SmallVector<OpFoldResult> allocOffsets(rank, OpFoldResult(lowerBound));
    allocOffsets[gatherDim] = inductionVar;
    auto dstAllocType = memref::SubViewOp::inferResultType(
        allocType, allocOffsets, sizes, oneStrides);
    auto dstSubview = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(dstAllocType), alloc, allocOffsets, sizes,
        oneStrides);
    // Copy srcPtr to alloc[inductionVar].
    rewriter.create<memref::CopyOp>(loc, srcPtr, dstSubview);

    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter,
                bool enableTensorFirstVectorCpu, MLIRContext *context)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context),
        enableTensorFirstVectorCpu(enableTensorFirstVectorCpu) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto originalPtr = op.getPtr();
    if (auto gatherScatterPtr =
            originalPtr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      return rewriteGather(gatherScatterPtr, op, adaptor.getPtr(), rewriter);
    }

    auto ptr = adaptor.getPtr();
    auto makeTPtr = originalPtr.getDefiningOp<tts::MakeTensorPtrOp>();
    if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType()) && makeTPtr) {
      FailureOr<Value> materialized = failure();
      if (makeTPtr.isStructuredPtr()) {
        materialized = materializeStructuredTPtrMemRef(makeTPtr, op.getLoc(),
                                                       rewriter);
      } else if (makeTPtr.isSplitPtr()) {
        materialized =
            materializeSplitTPtrMemRef(makeTPtr, op.getLoc(), rewriter);
      }
      if (failed(materialized)) {
        return rewriter.notifyMatchFailure(
            op, "expected pointer operand to lower from tts.make_tptr");
      }
      ptr = *materialized;
    }

    LogicalResult result = failure();
    if (isTensorFirstFastPathCandidate(op, ptr)) {
      result = op.hasMask() ? rewriteTensorFirstMaskedLoad(op, ptr, rewriter)
                            : rewriteTensorFirstUnmaskedLoad(op, ptr, rewriter);
    } else if (op.hasMask()) {
      result = rewriteMaskedLoad(op, ptr, rewriter);
    } else {
      result = rewriteStructuredLoad(op, ptr, rewriter);
    }

    if (succeeded(result) && makeTPtr && makeTPtr->use_empty()) {
      rewriter.eraseOp(makeTPtr);
    }
    return result;
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  bool enableTensorFirstVectorCpu;

  bool isTensorFirstFastPathCandidate(tts::StoreOp op, Value ptr) const {
    if (!enableTensorFirstVectorCpu) {
      return false;
    }

    auto ptrDefiningOp = ptr.getDefiningOp();
    if (ptrDefiningOp &&
        (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
         ptrDefiningOp->hasAttr(WRAP_STACKED) ||
         ptrDefiningOp->hasAttr(WRAP_LINEAR) ||
         isa<tts::MakeGatherScatterTensorPtrOp>(ptrDefiningOp))) {
      return false;
    }

    auto stValType = dyn_cast<RankedTensorType>(op.getValue().getType());
    auto memrefType = dyn_cast<MemRefType>(ptr.getType());
    if (!stValType || !memrefType) {
      return false;
    }
    if (stValType.getElementType() != memrefType.getElementType()) {
      return false;
    }
    return hasUnitStride1DLayout(memrefType) &&
           staticSizeCompatible1D(stValType, memrefType);
  }

  LogicalResult rewriteTensorFirstStore(tts::StoreOp op, Value ptr, Value stVal,
                                        ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto ptrSubview = createFullSubview1D(loc, ptr, rewriter);
    auto srcTensorType = cast<RankedTensorType>(stVal.getType());
    auto dynamicTensorType =
        RankedTensorType::get({ShapedType::kDynamic}, srcTensorType.getElementType());
    if (srcTensorType != dynamicTensorType) {
      stVal = rewriter.create<tensor::CastOp>(loc, dynamicTensorType, stVal);
    }
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, stVal, ptrSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult rewriteTensorFirstMaskedStore(
      tts::StoreOp op, Value ptr, Value stVal,
      ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());
    auto loc = op->getLoc();
    auto storeTensorType = dyn_cast<RankedTensorType>(stVal.getType());
    if (!storeTensorType) {
      return failure();
    }
    auto rank = storeTensorType.getRank();
    if (rank == 1) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);
      Value copyLen = ofrToIndexValue(mixedDims[0], loc, rewriter);
      emit1DTensorToMemrefStoreLoop(loc, stVal, dstSubview, copyLen, rewriter);
      rewriter.eraseOp(op);
      return success();
    }

    auto mixedDims = op.getMixedMaskDims();
    auto srcSlice = getExtractSlice(rank, mixedDims, stVal, loc, rewriter);
    auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, srcSlice, dstSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }

  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getSideBySideSlices(Value source, Value block1, Value block2, Location loc,
                      ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rowSize = rewriter.create<tensor::DimOp>(loc, source, 0);
    Value colSize1 = rewriter.create<memref::DimOp>(loc, block1, 1);
    Value colSize2 = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, c0}, ValueRange{rowSize, colSize1},
        ValueRange{c1, c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, colSize1}, ValueRange{rowSize, colSize2},
        ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getStackedSlices(Value source, Value block1, Value block2, Location loc,
                   ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rowSize1 = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value rowSize2 = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value colSize = rewriter.create<tensor::DimOp>(loc, source, 1);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0, c0}, ValueRange{rowSize1, colSize},
        ValueRange{c1, c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{rowSize1, c0}, ValueRange{rowSize2, colSize},
        ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getLinearSlices(Value source, Value block1, Value block2, Location loc,
                  ConversionPatternRewriter &rewriter) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value size1 = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value size2 = rewriter.create<memref::DimOp>(loc, block2, 0);

    auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{c0}, ValueRange{size1}, ValueRange{c1});
    auto slice2 = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, ValueRange{size1}, ValueRange{size2}, ValueRange{c1});
    return {slice1, slice2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getLinearMaskedStoreSubviews(tts::StoreOp op, Value block1, Value block2,
                               Location loc,
                               ConversionPatternRewriter &rewriter) const {
    Value requested = ofrToIndexValue(op.getMixedMaskDims()[0], loc, rewriter);
    Value block1Size = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block2Size = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value size1 = rewriter.create<arith::MinSIOp>(loc, requested, block1Size);
    Value remaining = rewriter.create<arith::SubIOp>(loc, requested, size1);
    Value size2 = rewriter.create<arith::MinSIOp>(loc, remaining, block2Size);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto block1Subview = rewriter.create<memref::SubViewOp>(
        loc, block1, ValueRange{c0}, ValueRange{size1}, ValueRange{c1});
    auto block2Subview = rewriter.create<memref::SubViewOp>(
        loc, block2, ValueRange{c0}, ValueRange{size2}, ValueRange{c1});
    return {block1Subview, block2Subview};
  }

  LogicalResult rewriteScatter(tts::MakeGatherScatterTensorPtrOp ptr,
                               tts::StoreOp op, Value memRefPtr, Value stVal,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index.
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        rewriter.create<arith::IndexCastOp>(loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        if (gatherMaskDimValue == 0 && ptr.getGatherScatterMask()) {
          // For unstructured mask case, loop over all elements and use the
          // unstructured mask to guard the store.
          gatherMaskDimValue = offsetSize;
        }
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = rewriter.create<arith::ConstantIndexOp>(loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            rewriter.create<arith::ConstantIndexOp>(loc, offsetSize);
        upperBound =
            rewriter
                .create<arith::MinSIOp>(loc, gatherMaskDimVal, offsetSizeVal)
                .getResult();
      }
    }
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // store.
      auto maskValue = rewriter.create<tensor::ExtractOp>(
          loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = rewriter.create<tensor::ExtractOp>(
        loc, gatherOffset, ValueRange{inductionVar});

    // Create extract_slice stVal[inductionVar].
    unsigned rank = ptr.getSizes().size();
    SmallVector<OpFoldResult> stValOffsets(rank, OpFoldResult(lowerBound));
    stValOffsets[gatherDim] = inductionVar;

    // Use mixed mask dims as sizes with mixedDims[gatherDim] set to 1 when
    // hasMask.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
    }
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, stVal, stValOffsets, sizes, oneStrides);

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value dstPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    // subview from dstPtr for mask.
    // Set offsets[] to 0 since it gatherOffsetElt already in reinterpret_cast.
    if (op.hasMask()) {
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      auto dstType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(dstPtr.getType()), maskOffsets, sizes, oneStrides);

      dstPtr =
          rewriter
              .create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), dstPtr,
                                         maskOffsets, sizes, oneStrides)
              .getResult();
    }
    // store slice to dstPtr.
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, slice, dstPtr);
    storeOp.setWritable(true);

    rewriter.eraseOp(op);

    return success();
  }

public:
  StoreConverter(const TypeConverter &typeConverter,
                 bool enableTensorFirstVectorCpu, MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context),
        enableTensorFirstVectorCpu(enableTensorFirstVectorCpu) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto originalPtr = op.getPtr();
    if (auto gatherScatterPtr =
            originalPtr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      return rewriteScatter(gatherScatterPtr, op, adaptor.getPtr(),
                            adaptor.getValue(), rewriter);
    }

    auto ptr = adaptor.getPtr();
    auto makeTPtr = originalPtr.getDefiningOp<tts::MakeTensorPtrOp>();
    if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType()) && makeTPtr) {
      FailureOr<Value> materialized = failure();
      if (makeTPtr.isStructuredPtr()) {
        materialized = materializeStructuredTPtrMemRef(makeTPtr, op.getLoc(),
                                                       rewriter);
      } else if (makeTPtr.isSplitPtr()) {
        materialized =
            materializeSplitTPtrMemRef(makeTPtr, op.getLoc(), rewriter);
      }
      if (failed(materialized)) {
        return rewriter.notifyMatchFailure(
            op, "expected pointer operand to lower from tts.make_tptr");
      }
      ptr = *materialized;
    }

    if (isTensorFirstFastPathCandidate(op, ptr)) {
      auto res = op.hasMask()
                     ? rewriteTensorFirstMaskedStore(op, ptr, adaptor.getValue(),
                                                     rewriter)
                     : rewriteTensorFirstStore(op, ptr, adaptor.getValue(),
                                               rewriter);
      if (succeeded(res) && makeTPtr && makeTPtr->use_empty()) {
        rewriter.eraseOp(makeTPtr);
      }
      return res;
    }

    if (auto unrealizedCast = getWraparoundCast(ptr)) {
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (op.hasMask() && !unrealizedCast->hasAttr(WRAP_LINEAR)) {
        return rewriter.notifyMatchFailure(
            op, "masked split-pointer store is not implemented");
      }

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [slice1, slice2] =
            getSideBySideSlices(adaptor.getValue(), block1, block2, loc, rewriter);
        auto store1 = rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, slice1, block1);
        store1.setWritable(true);
        auto store2 = rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, slice2, block2);
        store2.setWritable(true);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [slice1, slice2] =
            getStackedSlices(adaptor.getValue(), block1, block2, loc, rewriter);
        auto store1 = rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, slice1, block1);
        store1.setWritable(true);
        auto store2 = rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, slice2, block2);
        store2.setWritable(true);
      } else if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        if (op.hasMask()) {
          auto [block1Subview, block2Subview] =
              getLinearMaskedStoreSubviews(op, block1, block2, loc, rewriter);
          auto [slice1, slice2] = getLinearSlices(
              adaptor.getValue(), block1Subview, block2Subview, loc, rewriter);
          auto store1 =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  loc, slice1, block1Subview);
          store1.setWritable(true);
          auto store2 =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  loc, slice2, block2Subview);
          store2.setWritable(true);
        } else {
          auto [slice1, slice2] = getLinearSlices(adaptor.getValue(), block1,
                                                  block2, loc, rewriter);
          auto store1 =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  loc, slice1, block1);
          store1.setWritable(true);
          auto store2 =
              rewriter.create<bufferization::MaterializeInDestinationOp>(
                  loc, slice2, block2);
          store2.setWritable(true);
        }
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      rewriter.eraseOp(op);
      if (makeTPtr && makeTPtr->use_empty()) {
        rewriter.eraseOp(makeTPtr);
      }
      return success();
    }

    auto storeValue = op.getValue();
    auto storeTensorType = cast<RankedTensorType>(storeValue.getType());
    auto rank = storeTensorType.getRank();
    auto rankedPtr = ensureRankedMemRef(ptr, rank, storeTensorType.getElementType(),
                                        loc, rewriter);
    if (failed(rankedPtr)) {
      return rewriter.notifyMatchFailure(
          op, "expected pointer to lower to ranked/unranked memref");
    }
    ptr = *rankedPtr;

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    if (makeTPtr && makeTPtr->use_empty()) {
      rewriter.eraseOp(makeTPtr);
    }
    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefPreConversionPatterns(
    RewritePatternSet &patterns, bool enableTensorFirstVectorCpu) {
  if (!enableTensorFirstVectorCpu) {
    return;
  }
  patterns.add<MaskedReduceFusionPattern>(patterns.getContext(),
                                          PatternBenefit(10));
}

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    bool enableTensorFirstVectorCpu) {
  patterns.add<MakeTensorPtrConverter, MakeGatherScatterTensorPtrConverter>(
      typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(typeConverter,
                                              enableTensorFirstVectorCpu,
                                              patterns.getContext());
}
