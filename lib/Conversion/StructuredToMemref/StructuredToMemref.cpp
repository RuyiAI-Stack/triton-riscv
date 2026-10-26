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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
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
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using mlir::triton::TritonLoweringConfig;

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

static bool hasStaticallyUnitStride1D(Value ptr) {
  if (auto memrefType = dyn_cast<MemRefType>(ptr.getType())) {
    return hasUnitStride1DLayout(memrefType);
  }

  auto makeTPtr = ptr.getDefiningOp<tts::MakeTensorPtrOp>();
  if (!makeTPtr || makeTPtr.getSizes().size() != 1 ||
      makeTPtr.getMixedStrides().size() != 1) {
    return false;
  }
  auto stride = getIntAttr(makeTPtr.getMixedStrides().front());
  return stride && *stride == 1;
}

static bool staticSizeCompatible1D(RankedTensorType tensorType,
                                   MemRefType memrefType) {
  if (tensorType.getRank() != 1 || memrefType.getRank() != 1) {
    return false;
  }
  int64_t tensorSize = tensorType.getShape()[0];
  int64_t memrefSize = memrefType.getShape()[0];
  return tensorSize == ShapedType::kDynamic ||
         memrefSize == ShapedType::kDynamic || tensorSize == memrefSize;
}

static bool staticShapeCompatible(RankedTensorType tensorType,
                                  MemRefType memrefType) {
  if (tensorType.getRank() != memrefType.getRank()) {
    return false;
  }

  for (auto [tensorDim, memrefDim] :
       llvm::zip(tensorType.getShape(), memrefType.getShape())) {
    if (!ShapedType::isDynamic(tensorDim) &&
        !ShapedType::isDynamic(memrefDim) && tensorDim != memrefDim) {
      return false;
    }
  }
  return true;
}

static memref::SubViewOp createFullSubview1D(Location loc, Value source,
                                             OpBuilder &b) {
  auto c0 = arith::ConstantIndexOp::create(b, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto size = memref::DimOp::create(b, loc, source, 0).getResult();
  SmallVector<OpFoldResult> offsets = {c0.getResult()};
  SmallVector<OpFoldResult> sizes = {size};
  SmallVector<OpFoldResult> strides = {c1.getResult()};
  auto srcType = cast<MemRefType>(source.getType());
  auto subviewType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(subviewType),
                                   source, offsets, sizes, strides);
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
  auto dynamicLayout = StridedLayoutAttr::get(
      rewriter.getContext(), ShapedType::kDynamic, dynamicStrides);
  auto rankedType = MemRefType::get(dynamicShape, elementType, dynamicLayout,
                                    unrankedType.getMemorySpace());
  return memref::CastOp::create(rewriter, loc, rankedType, source).getResult();
}

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return memref::SubViewOp::create(b, loc, cast<MemRefType>(dstType), source,
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
    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto cVec = arith::ConstantIndexOp::create(rewriter, loc, 16);

    // Vectorized main body: [0, floor(upperBound / VL) * VL) with step VL.
    Value vecIters = arith::DivUIOp::create(rewriter, loc, upperBound, cVec);
    Value vecUpper = arith::MulIOp::create(rewriter, loc, vecIters, cVec);
    auto vecType = VectorType::get({16}, srcType.getElementType());
    auto vecLoop = scf::ForOp::create(rewriter, loc, c0, vecUpper, cVec);
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value vec = vector::LoadOp::create(rewriter, loc, vecType, srcSubview,
                                       ValueRange{ivVec});
    vector::StoreOp::create(rewriter, loc, vec, dstSubview, ValueRange{ivVec});

    // Scalar tail: [vecUpper, upperBound).
    rewriter.setInsertionPointAfter(vecLoop);
    auto tailLoop = scf::ForOp::create(rewriter, loc, vecUpper, upperBound, c1);
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    Value iv = tailLoop.getInductionVar();
    Value v = memref::LoadOp::create(rewriter, loc, srcSubview, ValueRange{iv});
    memref::StoreOp::create(rewriter, loc, v, dstSubview, ValueRange{iv});
    return;
  }

  // Conservative fallback when we cannot build a fixed-size vector load/store.
  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto loop = scf::ForOp::create(rewriter, loc, c0, upperBound, c1);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();
  Value v = memref::LoadOp::create(rewriter, loc, srcSubview, ValueRange{iv});
  memref::StoreOp::create(rewriter, loc, v, dstSubview, ValueRange{iv});
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
    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto cVec = arith::ConstantIndexOp::create(rewriter, loc, 16);

    // Vectorized main body: [0, floor(upperBound / VL) * VL) with step VL.
    Value vecIters = arith::DivUIOp::create(rewriter, loc, upperBound, cVec);
    Value vecUpper = arith::MulIOp::create(rewriter, loc, vecIters, cVec);
    auto elemType = srcType.getElementType();
    auto vecType = VectorType::get({16}, elemType);
    auto vecLoop = scf::ForOp::create(rewriter, loc, c0, vecUpper, cVec);
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value padding = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getZeroAttr(elemType));
    auto identityMap = AffineMap::getMinorIdentityMap(
        /*numDims=*/1, /*numResults=*/1, rewriter.getContext());
    const bool inBoundsArr[] = {true};
    Value vec = vector::TransferReadOp::create(
        rewriter, loc, vecType, srcTensor, ValueRange{ivVec},
        std::optional<Value>(padding), identityMap,
        std::optional<ArrayRef<bool>>(ArrayRef<bool>(inBoundsArr)));
    vector::StoreOp::create(rewriter, loc, vec, dstSubview, ValueRange{ivVec});

    // Scalar tail: [vecUpper, upperBound).
    rewriter.setInsertionPointAfter(vecLoop);
    auto tailLoop = scf::ForOp::create(rewriter, loc, vecUpper, upperBound, c1);
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    Value iv = tailLoop.getInductionVar();
    Value v =
        tensor::ExtractOp::create(rewriter, loc, srcTensor, ValueRange{iv});
    memref::StoreOp::create(rewriter, loc, v, dstSubview, ValueRange{iv});
    return;
  }

  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto loop = scf::ForOp::create(rewriter, loc, c0, upperBound, c1);
  rewriter.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar();
  Value v = tensor::ExtractOp::create(rewriter, loc, srcTensor, ValueRange{iv});
  memref::StoreOp::create(rewriter, loc, v, dstSubview, ValueRange{iv});
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
    dynamicDims.push_back(memref::DimOp::create(b, loc, sourceMemRef, i));
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
  // tensor<128x64x!tt.ptr<bf16>>
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return ptrType.getPointeeType();
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

  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return MemRefType::get(resultShape, ptrType.getPointeeType(), layout);
}

static Type
getElementTypeGatherScatterPtr(tts::MakeGatherScatterTensorPtrOp op) {
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return ptrType.getPointeeType();
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

static UnrealizedConversionCastOp
createRank1WraparoundDescriptor(Location loc, Type resultType, Value base,
                                Value targetOffset, Value modulo, Value stride,
                                OpBuilder &b) {
  Value start = arith::RemSIOp::create(b, loc, targetOffset, modulo);
  auto descriptor = UnrealizedConversionCastOp::create(
      b, loc, resultType, ValueRange{base, start, modulo, stride});
  descriptor->setAttr(WRAP_LINEAR, b.getUnitAttr());
  return descriptor;
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
    base = UnrealizedConversionCastOp::create(rewriter, loc, unrankedType, base)
               .getResult(0);
  }

  auto mixedStrides = getMixedStridesForMemref(op, rewriter);
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  auto targetOffset =
      accumulateTargetOffset(loc, op.getMixedOffsets(), rewriter);
  auto staticTargetOffset = getIntAttr(targetOffset);
  ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
  auto resultType =
      getResultMemrefType(op, staticTargetOffset.value_or(ShapedType::kDynamic),
                          staticStrides, resultShape);

  return memref::ReinterpretCastOp::create(rewriter, loc, resultType, base,
                                           targetOffset, op.getMixedSizes(),
                                           mixedStrides)
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
    base = UnrealizedConversionCastOp::create(rewriter, loc, unrankedType, base)
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

    Value modulo = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);
    return createRank1WraparoundDescriptor(loc, op.getType(), base,
                                           targetOffset, modulo, strideVals[0],
                                           rewriter)
        .getResult(0);
  }

  if (resultShape.size() != 2 || parentShape.size() != 2) {
    return failure();
  }

  auto resultType = getResultMemrefType(
      op, /*offset=*/ShapedType::kDynamic,
      SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
      SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic});
  Value rowSize = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIndexAttr(op.getSizes()[0]));
  Value colSize = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIndexAttr(op.getSizes()[1]));

  SmallVector<Value> casts;
  StringRef wrapType;
  if (isSplitDimension(parentShape[0])) {
    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);
    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    Value wrappedAroundOff =
        arith::RemSIOp::create(rewriter, loc, targetOffset, strideRow);
    Value clampedOff =
        arith::AddIOp::create(rewriter, loc, modRow, wrappedAroundOff);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedOff, targetOffset);
    d1 = arith::DivSIOp::create(rewriter, loc, d1, strideRow);
    d1 = arith::MinSIOp::create(rewriter, loc, d1, rowSize);
    Value d2 = arith::SubIOp::create(rewriter, loc, rowSize, d1);

    casts.push_back(
        memref::ReinterpretCastOp::create(rewriter, loc, resultType, base,
                                          targetOffset, ValueRange{d1, colSize},
                                          ValueRange{strideRow, strideCol})
            .getResult());
    casts.push_back(memref::ReinterpretCastOp::create(
                        rewriter, loc, resultType, base, wrappedAroundOff,
                        ValueRange{d2, colSize},
                        ValueRange{strideRow, strideCol})
                        .getResult());
    wrapType = WRAP_STACKED;
  } else if (isSplitDimension(parentShape[1])) {
    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);
    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);
    Value zero;
    Value one;
    auto getZero = [&]() {
      if (!zero) {
        zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      }
      return zero;
    };
    auto getOne = [&]() {
      if (!one) {
        one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      }
      return one;
    };
    auto strideRowConst = getConstantIntValue(strideRow);
    auto strideColConst = getConstantIntValue(strideCol);
    if (!strideRowConst || *strideRowConst == 0) {
      Value rowSpan = arith::MulIOp::create(rewriter, loc, modN, strideCol);
      if (!strideRowConst) {
        Value isZero = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::eq, strideRow, getZero());
        strideRow =
            arith::SelectOp::create(rewriter, loc, isZero, rowSpan, strideRow);
      } else {
        strideRow = rowSpan;
      }
    }
    if ((!strideRowConst || *strideRowConst == 0) &&
        (!strideColConst || *strideColConst == 0)) {
      Value isZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, strideRow, getZero());
      strideRow =
          arith::SelectOp::create(rewriter, loc, isZero, getOne(), strideRow);
    }
    Value intraRowOffset =
        arith::RemSIOp::create(rewriter, loc, targetOffset, strideRow);
    Value rowBase =
        arith::SubIOp::create(rewriter, loc, targetOffset, intraRowOffset);
    Value colIndex;
    if (strideColConst && *strideColConst == 0) {
      colIndex = getZero();
    } else if (!strideColConst) {
      Value isZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, strideCol, getZero());
      Value safeStrideCol =
          arith::SelectOp::create(rewriter, loc, isZero, getOne(), strideCol);
      Value dividedColIndex =
          arith::DivSIOp::create(rewriter, loc, intraRowOffset, safeStrideCol);
      colIndex = arith::SelectOp::create(rewriter, loc, isZero, getZero(),
                                         dividedColIndex);
    } else {
      colIndex =
          arith::DivSIOp::create(rewriter, loc, intraRowOffset, strideCol);
    }
    Value nextCol = arith::AddIOp::create(rewriter, loc, colIndex, colSize);
    Value clampedCol = arith::MinSIOp::create(rewriter, loc, nextCol, modN);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedCol, colIndex);
    Value d2 = arith::SubIOp::create(rewriter, loc, colSize, d1);

    casts.push_back(memref::ReinterpretCastOp::create(
                        rewriter, loc, resultType, base, targetOffset,
                        ValueRange{rowSize, d1}, strideVals)
                        .getResult());
    casts.push_back(memref::ReinterpretCastOp::create(
                        rewriter, loc, resultType, base, rowBase,
                        ValueRange{rowSize, d2}, strideVals)
                        .getResult());
    wrapType = WRAP_SIDE_BY_SIDE;
  } else {
    return failure();
  }

  auto combinedCast =
      UnrealizedConversionCastOp::create(rewriter, loc, op.getType(), casts);
  combinedCast->setAttr(wrapType, rewriter.getUnitAttr());
  return combinedCast.getResult(0);
}

static FailureOr<Value>
materializeGatherScatterBaseMemRef(tts::MakeGatherScatterTensorPtrOp op,
                                   Location loc, OpBuilder &rewriter) {
  Value base = op.getBase();
  Type elementType = getElementTypeGatherScatterPtr(op);

  if (auto memrefType = dyn_cast<MemRefType>(base.getType())) {
    if (memrefType.getElementType() != elementType) {
      return failure();
    }
    return base;
  }

  if (auto unrankedType = dyn_cast<UnrankedMemRefType>(base.getType())) {
    if (unrankedType.getElementType() != elementType) {
      return failure();
    }
    return base;
  }

  if (!isa<triton::PointerType>(base.getType())) {
    return failure();
  }

  auto unrankedType = UnrankedMemRefType::get(elementType, 0);
  return UnrealizedConversionCastOp::create(rewriter, loc, unrankedType, base)
      .getResult(0);
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

struct Rank1WraparoundDescriptor {
  Value base;
  Value start;
  Value modulo;
  Value stride;
};

static std::optional<Rank1WraparoundDescriptor>
getRank1WraparoundDescriptor(UnrealizedConversionCastOp castOp) {
  if (!castOp->hasAttr(WRAP_LINEAR) || castOp.getOperands().size() != 4) {
    return std::nullopt;
  }
  return Rank1WraparoundDescriptor{
      castOp.getOperands()[0], castOp.getOperands()[1], castOp.getOperands()[2],
      castOp.getOperands()[3]};
}

static Value createNormalizedRemainder(Location loc, Value dividend,
                                       Value positiveDivisor, OpBuilder &b) {
  Value remainder = arith::RemSIOp::create(b, loc, dividend, positiveDivisor);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value isNegative =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, remainder, zero);
  Value adjusted = arith::AddIOp::create(b, loc, remainder, positiveDivisor);
  return arith::SelectOp::create(b, loc, isNegative, adjusted, remainder);
}

static Value createRank1WraparoundElementMemRef(
    Location loc, const Rank1WraparoundDescriptor &descriptor,
    Value logicalIndex, Type elementType, OpBuilder &b) {
  Value distance =
      arith::MulIOp::create(b, loc, logicalIndex, descriptor.stride);
  Value linearOffset =
      arith::AddIOp::create(b, loc, descriptor.start, distance);
  Value wrappedOffset =
      createNormalizedRemainder(loc, linearOffset, descriptor.modulo, b);
  auto baseType = cast<BaseMemRefType>(descriptor.base.getType());
  auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                       ArrayRef<int64_t>{1});
  auto elementMemRefType =
      MemRefType::get({1}, elementType, layout, baseType.getMemorySpace());
  SmallVector<OpFoldResult> one{b.getIndexAttr(1)};
  return memref::ReinterpretCastOp::create(b, loc, elementMemRefType,
                                           descriptor.base, wrappedOffset, one,
                                           one)
      .getResult();
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

  auto castOp = memref::ReinterpretCastOp::create(
      rewriter, op.getLoc(), resultType, basePtr, targetOffset, sizes,
      mixedStrides);

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
      arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false))
          .getResult();
  for (size_t i = 0; i < shape.size(); i++) {
    auto shapei = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getIndexAttr(shape[i]));

    Value dimi = dyn_cast<Value>(mixedDims[i]);
    if (!dimi) {
      dimi = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(staticMaskDims[i]));
    }

    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                      dimi, shapei);
    accBase = arith::OrIOp::create(rewriter, loc, accBase, cmp);
  }

  // condition the memset on the or-accumulation
  // initialize with padding prior to CopyOp
  scf::IfOp::create(rewriter, loc, accBase, [&](OpBuilder &b, Location loc) {
    linalg::FillOp::create(b, loc, ValueRange{other}, ValueRange{alloc});
    scf::YieldOp::create(b, loc);
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
    return arith::ConstantOp::create(rewriter, loc, resultType,
                                     rewriter.getFloatAttr(resultType, 0.0))
        .getResult();
  case MaskedReduceKind::AddI:
  case MaskedReduceKind::MaxUI:
    return arith::ConstantOp::create(rewriter, loc, resultType,
                                     rewriter.getIntegerAttr(resultType, 0))
        .getResult();
  case MaskedReduceKind::MaximumF:
  case MaskedReduceKind::MaxNumF:
    return arith::ConstantOp::create(
               rewriter, loc, resultType,
               rewriter.getFloatAttr(resultType,
                                     -std::numeric_limits<float>::infinity()))
        .getResult();
  case MaskedReduceKind::MaxSI:
    return arith::ConstantOp::create(
               rewriter, loc, resultType,
               rewriter.getIntegerAttr(
                   resultType,
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
    return arith::ExtFOp::create(rewriter, loc, resultType, value,
                                 arith::FastMathFlagsAttr());
  }
  llvm_unreachable("unexpected masked reduce type mismatch");
}

static Value combineMaskedReduceValue(Location loc, MaskedReduceKind kind,
                                      Value input, Value acc, Type resultType,
                                      PatternRewriter &rewriter) {
  input = castMaskedReduceInput(loc, input, resultType, kind, rewriter);
  switch (kind) {
  case MaskedReduceKind::AddF:
    return arith::AddFOp::create(rewriter, loc, input, acc).getResult();
  case MaskedReduceKind::AddI:
    return arith::AddIOp::create(rewriter, loc, input, acc).getResult();
  case MaskedReduceKind::MaximumF:
    return arith::MaximumFOp::create(rewriter, loc, input, acc).getResult();
  case MaskedReduceKind::MaxNumF:
    return arith::MaxNumFOp::create(rewriter, loc, input, acc).getResult();
  case MaskedReduceKind::MaxSI:
    return arith::MaxSIOp::create(rewriter, loc, input, acc).getResult();
  case MaskedReduceKind::MaxUI:
    return arith::MaxUIOp::create(rewriter, loc, input, acc).getResult();
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
    auto extract = dyn_cast<tensor::ExtractOp>(*op->getResult(0).user_begin());
    if (!extract || !extract.getIndices().empty()) {
      return failure();
    }

    auto ptr = load.getPtr();
    if (!hasStaticallyUnitStride1D(ptr)) {
      return failure();
    }
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

    auto rankedPtr = ensureRankedMemRef(
        ptr, /*rank=*/1, loadType.getElementType(), load.getLoc(), rewriter);
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
    Value validLen = ofrToIndexValue(load.getMixedMaskDims()[0], loc, rewriter);
    Value other = load.getOther();
    int64_t fullSize = loadType.getShape()[0];

    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    auto cFull = arith::ConstantIndexOp::create(rewriter, loc, fullSize);

    Value vecIters = arith::DivUIOp::create(rewriter, loc, validLen, c16);
    Value vecUpper = arith::MulIOp::create(rewriter, loc, vecIters, c16);
    auto vecType = VectorType::get({16}, loadType.getElementType());

    auto vecLoop = scf::ForOp::create(rewriter, loc, c0, vecUpper, c16,
                                      ValueRange{accInit});
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    Value accVec = vecLoop.getRegionIterArgs().front();
    Value vec =
        vector::LoadOp::create(rewriter, loc, vecType, ptr, ValueRange{ivVec});
    Value vecReduced =
        vector::ReductionOp::create(rewriter, loc, *vectorKind, vec, accVec);
    scf::YieldOp::create(rewriter, loc, vecReduced);

    rewriter.setInsertionPointAfter(vecLoop);
    auto scalarLoop = scf::ForOp::create(rewriter, loc, vecUpper, validLen, c1,
                                         ValueRange{vecLoop.getResult(0)});
    rewriter.setInsertionPointToStart(scalarLoop.getBody());
    Value iv = scalarLoop.getInductionVar();
    Value accScalar = scalarLoop.getRegionIterArgs().front();
    Value elem = memref::LoadOp::create(rewriter, loc, ptr, ValueRange{iv});
    Value next = combineMaskedReduceValue(loc, *kind, elem, accScalar,
                                          resultType, rewriter);
    scf::YieldOp::create(rewriter, loc, next);

    rewriter.setInsertionPointAfter(scalarLoop);
    auto paddingLoop = scf::ForOp::create(rewriter, loc, validLen, cFull, c1,
                                          ValueRange{scalarLoop.getResult(0)});
    rewriter.setInsertionPointToStart(paddingLoop.getBody());
    Value accPadding = paddingLoop.getRegionIterArgs().front();
    Value nextPadding = combineMaskedReduceValue(loc, *kind, other, accPadding,
                                                 resultType, rewriter);
    scf::YieldOp::create(rewriter, loc, nextPadding);

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

// Fuse a rank-1 elementwise DAG rooted at a masked store before masked loads
// are materialized into temporary buffers. Only valid lanes are accessed, so
// masked-load bounds semantics are preserved without padded heap allocations.
struct MaskedElementwiseStoreFusionPattern
    : public OpRewritePattern<tts::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isIdentityElementwise(linalg::GenericOp generic) {
    if (generic->getNumResults() != 1 || generic.getNumLoops() != 1 ||
        generic.getNumParallelLoops() != 1)
      return false;
    return llvm::all_of(generic.getIndexingMapsArray(),
                        [](AffineMap map) { return map.isIdentity(); });
  }

  static bool isCloneableElementwiseOp(Operation &op) {
    return isa<arith::ConstantOp>(op) ||
           (op.getNumRegions() == 0 && op.hasTrait<OpTrait::Elementwise>() &&
            isMemoryEffectFree(&op));
  }

  LogicalResult
  collectExpression(Value value, tts::StoreOp store,
                    SmallVectorImpl<tts::LoadOp> &loads,
                    SmallVectorImpl<linalg::GenericOp> &generics) const {
    if (auto load = value.getDefiningOp<tts::LoadOp>()) {
      auto type = dyn_cast<RankedTensorType>(load.getType());
      if (!load.hasMask() || !type || type.getRank() != 1 ||
          !type.hasStaticShape() ||
          load.getMixedMaskDims() != store.getMixedMaskDims())
        return failure();
      if (!llvm::is_contained(loads, load))
        loads.push_back(load);
      return success();
    }

    auto generic = value.getDefiningOp<linalg::GenericOp>();
    if (!generic || !isIdentityElementwise(generic) ||
        !llvm::is_contained(generic->getResults(), value))
      return failure();
    if (llvm::is_contained(generics, generic))
      return success();

    Block &body = generic.getRegion().front();
    unsigned inputCount = generic.getInputs().size();
    if (body.getNumArguments() != inputCount + generic.getDpsInits().size())
      return failure();
    for (BlockArgument outputArg : body.getArguments().drop_front(inputCount))
      if (!outputArg.use_empty())
        return failure();
    for (Operation &bodyOp : body.without_terminator())
      if (!isCloneableElementwiseOp(bodyOp))
        return failure();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();

    generics.push_back(generic);
    for (Value input : generic.getInputs())
      if (failed(collectExpression(input, store, loads, generics)))
        return failure();
    return success();
  }

  FailureOr<Value>
  emitExpression(Value value, Value index, bool vectorMode,
                 const llvm::DenseMap<Operation *, Value> &loadPtrs,
                 llvm::DenseMap<Value, Value> &cache,
                 PatternRewriter &rewriter) const {
    if (auto it = cache.find(value); it != cache.end())
      return it->second;

    if (auto load = value.getDefiningOp<tts::LoadOp>()) {
      auto ptrIt = loadPtrs.find(load.getOperation());
      if (ptrIt == loadPtrs.end())
        return failure();
      Type elementType =
          cast<RankedTensorType>(load.getType()).getElementType();
      Value result =
          vectorMode
              ? vector::LoadOp::create(rewriter, load.getLoc(),
                                       VectorType::get({16}, elementType),
                                       ptrIt->second, ValueRange{index})
                    .getResult()
              : memref::LoadOp::create(rewriter, load.getLoc(), ptrIt->second,
                                       ValueRange{index})
                    .getResult();
      cache[value] = result;
      return result;
    }

    auto generic = value.getDefiningOp<linalg::GenericOp>();
    if (!generic)
      return failure();
    Block &body = generic.getRegion().front();
    IRMapping mapping;
    for (auto [arg, input] :
         llvm::zip(body.getArguments().take_front(generic.getInputs().size()),
                   generic.getInputs())) {
      auto emitted =
          emitExpression(input, index, vectorMode, loadPtrs, cache, rewriter);
      if (failed(emitted))
        return failure();
      mapping.map(arg, *emitted);
    }

    for (Operation &bodyOp : body.without_terminator()) {
      if (auto constant = dyn_cast<arith::ConstantOp>(bodyOp)) {
        Type resultType = constant.getType();
        if (vectorMode) {
          auto typed = dyn_cast<TypedAttr>(constant.getValue());
          if (!typed)
            return failure();
          auto vectorType = VectorType::get({16}, resultType);
          auto splat = DenseElementsAttr::get(vectorType, typed);
          Value cloned = arith::ConstantOp::create(rewriter, constant.getLoc(),
                                                   vectorType, splat);
          mapping.map(constant.getResult(), cloned);
        } else {
          Value cloned = arith::ConstantOp::create(
              rewriter, constant.getLoc(), resultType, constant.getValue());
          mapping.map(constant.getResult(), cloned);
        }
        continue;
      }

      OperationState state(bodyOp.getLoc(), bodyOp.getName());
      for (Value operand : bodyOp.getOperands()) {
        Value mapped = mapping.lookupOrNull(operand);
        if (!mapped)
          return failure();
        state.addOperands(mapped);
      }
      for (Type type : bodyOp.getResultTypes())
        state.addTypes(vectorMode ? Type(VectorType::get({16}, type)) : type);
      state.addAttributes(bodyOp.getAttrs());
      Operation *cloned = rewriter.create(state);
      for (auto [oldResult, newResult] :
           llvm::zip(bodyOp.getResults(), cloned->getResults()))
        mapping.map(oldResult, newResult);
    }

    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    Value result = mapping.lookupOrNull(yield.getValues().front());
    if (!result)
      return failure();
    cache[value] = result;
    return result;
  }

  LogicalResult matchAndRewrite(tts::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!store.hasMask())
      return failure();

    auto storeType = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!storeType || storeType.getRank() != 1 || !storeType.hasStaticShape())
      return failure();

    SmallVector<tts::LoadOp> loads;
    SmallVector<linalg::GenericOp> generics;
    if (failed(collectExpression(store.getValue(), store, loads, generics)) ||
        loads.empty() || generics.empty())
      return failure();

    for (linalg::GenericOp generic : generics)
      for (OpOperand &use : generic->getUses())
        if (use.getOwner() != store.getOperation() &&
            !llvm::is_contained(generics, use.getOwner()))
          return failure();
    for (tts::LoadOp load : loads)
      for (Operation *user : load->getUsers())
        if (!llvm::is_contained(generics, user))
          return failure();

    if (!hasStaticallyUnitStride1D(store.getPtr()) ||
        llvm::any_of(loads, [](tts::LoadOp load) {
          return !hasStaticallyUnitStride1D(load.getPtr());
        })) {
      return failure();
    }

    auto materializePtr = [&](Value ptr, Type elementType,
                              Location loc) -> FailureOr<Value> {
      if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType())) {
        auto makeTPtr = ptr.getDefiningOp<tts::MakeTensorPtrOp>();
        if (!makeTPtr)
          return failure();
        auto materialized =
            materializeStructuredTPtrMemRef(makeTPtr, loc, rewriter);
        if (failed(materialized))
          return failure();
        ptr = *materialized;
      }
      return ensureRankedMemRef(ptr, /*rank=*/1, elementType, loc, rewriter);
    };

    Location loc = store.getLoc();
    llvm::DenseMap<Operation *, Value> loadPtrs;
    for (tts::LoadOp load : loads) {
      Type elementType =
          cast<RankedTensorType>(load.getType()).getElementType();
      auto ptr = materializePtr(load.getPtr(), elementType, loc);
      if (failed(ptr))
        return failure();
      auto type = dyn_cast<MemRefType>(ptr->getType());
      if (!type || !hasUnitStride1DLayout(type))
        return failure();
      loadPtrs[load.getOperation()] = *ptr;
    }
    auto dstPtr =
        materializePtr(store.getPtr(), storeType.getElementType(), loc);
    if (failed(dstPtr))
      return failure();
    auto dstType = dyn_cast<MemRefType>(dstPtr->getType());
    if (!dstType || !hasUnitStride1DLayout(dstType))
      return failure();

    Value validLen =
        ofrToIndexValue(store.getMixedMaskDims()[0], loc, rewriter);
    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    Value vecIters = arith::DivUIOp::create(rewriter, loc, validLen, c16);
    Value vecUpper = arith::MulIOp::create(rewriter, loc, vecIters, c16);
    auto vecLoop = scf::ForOp::create(rewriter, loc, c0, vecUpper, c16);
    rewriter.setInsertionPointToStart(vecLoop.getBody());
    Value ivVec = vecLoop.getInductionVar();
    llvm::DenseMap<Value, Value> vectorCache;
    auto vectorResult =
        emitExpression(store.getValue(), ivVec,
                       /*vectorMode=*/true, loadPtrs, vectorCache, rewriter);
    if (failed(vectorResult))
      return failure();
    vector::StoreOp::create(rewriter, loc, *vectorResult, *dstPtr,
                            ValueRange{ivVec});

    rewriter.setInsertionPointAfter(vecLoop);
    auto tailLoop = scf::ForOp::create(rewriter, loc, vecUpper, validLen, c1);
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    Value iv = tailLoop.getInductionVar();
    llvm::DenseMap<Value, Value> scalarCache;
    auto scalarResult =
        emitExpression(store.getValue(), iv,
                       /*vectorMode=*/false, loadPtrs, scalarCache, rewriter);
    if (failed(scalarResult))
      return failure();
    memref::StoreOp::create(rewriter, loc, *scalarResult, *dstPtr,
                            ValueRange{iv});

    rewriter.eraseOp(store);
    for (linalg::GenericOp generic : generics)
      if (generic->use_empty())
        rewriter.eraseOp(generic);
    for (tts::LoadOp load : loads)
      if (load->use_empty())
        rewriter.eraseOp(load);
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
    // tensor<128x64x!tt.ptr<bf16>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
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

    Value rowSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = arith::RemSIOp::create(rewriter, loc, targetOffset, modN);
    Value y = arith::SubIOp::create(rewriter, loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = arith::AddIOp::create(rewriter, loc, x, colSize);
    Value clampedOffset =
        arith::MinSIOp::create(rewriter, loc, nextOffset, modN);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedOffset, x);
    SmallVector<Value> sizes1{rowSize, d1};

    auto cast1 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), targetOffset, sizes1,
        strideVals);

    // Second chunk
    Value d2 = arith::SubIOp::create(rewriter, loc, colSize, d1);
    SmallVector<Value> sizes2{rowSize, d2};

    auto cast2 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), y, sizes2, strideVals);

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

    Value rowSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    // First chunk
    Value wrappedAroundOff =
        arith::RemSIOp::create(rewriter, loc, targetOffset, strideRow);
    Value clampedOff =
        arith::AddIOp::create(rewriter, loc, modRow, wrappedAroundOff);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedOff, targetOffset);
    d1 = arith::DivSIOp::create(rewriter, loc, d1, strideRow);
    d1 = arith::MinSIOp::create(rewriter, loc, d1, rowSize);

    SmallVector<Value> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), targetOffset, sizes1,
        ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = arith::SubIOp::create(rewriter, loc, rowSize, d1);
    SmallVector<Value> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), wrappedAroundOff, sizes2,
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
      Value targetOffset = ofrToIndexValue(
          accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
          loc, rewriter);
      Value stride = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
      Value modulo = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);
      auto descriptor = createRank1WraparoundDescriptor(
          loc, op.getType(), adaptor.getBase(), targetOffset, modulo, stride,
          rewriter);
      rewriter.replaceOp(op, descriptor);
      return success();
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

    auto combinedCast = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getType(), casts);

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

    auto castOp = memref::ReinterpretCastOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getBase(), targetOffset,
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
    // order carries block-pointer layout semantics.
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
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

// Follow the view-like operations that can be introduced between a Triton
// pointer and the eventual memref.copy destination.  This lets provenance
// distinguish a function argument (external DDR) from a compiler-owned
// temporary even after ensureRankedMemRef has inserted a memref.cast.
static Value getMemRefRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized.getNumOperands() != 1)
        break;
      value = unrealized.getOperand(0);
      continue;
    }
    // The pointer bridge can survive until this pass for scalar/unstructured
    // forms.  It is a one-operand view whose source still identifies the DDR
    // function argument or compiler-owned allocation.
    if (def->getName().getStringRef() == "tptr.to_memref" &&
        def->getNumOperands() == 1) {
      value = def->getOperand(0);
      continue;
    }
    if (def->getName().getStringRef() == "tptr.from_memref" &&
        def->getNumOperands() == 1) {
      value = def->getOperand(0);
      continue;
    }
    // Pointer arithmetic preserves the allocation identity in operand zero;
    // follow it so an address derived from a compiler temporary is not
    // misclassified as an externally supplied DDR pointer.
    StringRef name = def->getName().getStringRef();
    if ((name == "tptr.ptradd" || name == "ptr.ptradd" ||
         name == "tt.addptr") &&
        def->getNumOperands() >= 1) {
      value = def->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

static StringRef classifyStoreDestination(Value value) {
  Value root = getMemRefRoot(value);
  if (isa<BlockArgument>(root))
    return "external_ddr";
  if (Operation *def = root.getDefiningOp()) {
    if (isa<memref::AllocOp, memref::AllocaOp, bufferization::ToBufferOp>(def))
      return "internal_temp";
  }
  // A pointer whose root cannot be proven to be an allocation is conservatively
  // treated as an externally supplied address, matching Triton function
  // pointer semantics and preserving direct-DDR lowering.
  return "external_ddr";
}

// Keep the memory event on an ordinary copy. Its operands retain the real DDR
// root, descriptor offset/strides and dynamic mask bounds through
// bufferization.  The store destination is recorded independently of the
// source tensor's eventual staging location so the backend can select a
// direct-DDR path without guessing from a memref type after bufferization.
static DictionaryAttr memoryProvenance(OpBuilder &builder, StringRef kind,
                                       RankedTensorType logicalType,
                                       ArrayRef<OpFoldResult> mask,
                                       Value destination = {}, Value other = {},
                                       Operation *sourceEvent = nullptr) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(kind));
  if (sourceEvent) {
    if (Attribute id = sourceEvent->getAttr("contract.origin_id"))
      attrs.set("origin_id", id);
  }
  attrs.set("boundary",
            builder.getStringAttr(kind == "store" && destination
                                      ? classifyStoreDestination(destination)
                                      : "external_ddr"));
  attrs.set("logical_shape",
            builder.getDenseI64ArrayAttr(logicalType.getShape()));
  NamedAttrList maskAttrs;
  maskAttrs.set("kind",
                builder.getStringAttr(mask.empty() ? "none" : "rectangular"));
  if (!mask.empty()) {
    SmallVector<int64_t> validShape(logicalType.getShape());
    for (auto [i, extent] : llvm::enumerate(mask))
      validShape[i] =
          getConstantIntValue(extent).value_or(ShapedType::kDynamic);
    maskAttrs.set("valid_shape", builder.getDenseI64ArrayAttr(validShape));
  }
  attrs.set("mask", maskAttrs.getDictionary(builder.getContext()));
  if (other)
    if (auto constant = other.getDefiningOp<arith::ConstantOp>())
      attrs.set("other", constant.getValue());
  return attrs.getDictionary(builder.getContext());
}

static IntegerAttr getSnapshotAlignment(Operation *op, Builder &builder) {
  auto module = op->getParentOfType<ModuleOp>();
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  if (abi)
    if (auto alignment = abi.getAs<IntegerAttr>("args_aligned_bytes"))
      return builder.getI64IntegerAttr(alignment.getInt());
  return {};
}

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
  bool emitContract;

private:
  bool enableTensorFirstVectorCpu;

  LogicalResult
  rewriteContractLogicalLoad(tts::LoadOp op, Value ptr,
                             ConversionPatternRewriter &rewriter) const {
    auto tensorType = cast<RankedTensorType>(op.getType());
    if (op.hasMask()) {
      auto other = op.getOther()
                       ? op.getOther().getDefiningOp<arith::ConstantOp>()
                       : arith::ConstantOp();
      if (!other || other.getType() != tensorType.getElementType())
        return op.emitError("neutral contract masked load requires a scalar "
                            "constant other in contract v1");
    }
    if (!tensorType.hasStaticShape() || getWraparoundCast(ptr))
      return op.emitError(
          "neutral contract load requires a static rectangular DDR view; "
          "split/wrapping pointers are unsupported");
    auto ranked =
        ensureRankedMemRef(ptr, tensorType.getRank(),
                           tensorType.getElementType(), op.getLoc(), rewriter);
    if (failed(ranked))
      return op.emitError("neutral contract load requires a ranked DDR view");
    ptr = *ranked;
    SmallVector<OpFoldResult> mask =
        op.hasMask() ? op.getMixedMaskDims() : SmallVector<OpFoldResult>{};
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(cast<MemRefType>(ptr.getType())
                   .getStridesAndOffset(strides, offset)) ||
        llvm::any_of(strides, [](int64_t stride) {
          return ShapedType::isDynamic(stride) || stride <= 0;
        }))
      strides.clear();

    // Preserve the physical contiguous axis, not the DDR pitch. A snapshot
    // owns exactly one logical tile, including padding for a masked load.
    // Source and destination subviews below retain their independent layouts.
    // Prove separation using the actual copied rectangle. In particular a
    // short K may have a dense DDR pitch smaller than the padded tile width;
    // excluded lanes belong only to the private snapshot, never to DDR.
    auto accessedExtent = [&](unsigned axis) {
      int64_t logical = tensorType.getDimSize(axis);
      if (!mask.empty())
        if (auto valid = getConstantIntValue(mask[axis]);
            valid && *valid >= 0 && *valid <= logical)
          return *valid;
      return logical;
    };
    bool rowMajor = false;
    if (tensorType.getRank() == 1 && strides.size() == 1 && strides[0] == 1)
      rowMajor = true;
    else if (tensorType.getRank() == 2 && strides.size() == 2) {
      if (strides[1] == 1 && strides[0] >= accessedExtent(1))
        rowMajor = true;
      else if (strides[0] != 1 || strides[1] < accessedExtent(0))
        return op.emitError(
            "neutral contract load requires a non-overlapping rectangular "
            "layout with a static unit-stride axis");
    } else {
      return op.emitError(
          "neutral contract load snapshot requires a rank-one or rank-two "
          "rectangular layout with a static unit-stride axis");
    }
    int64_t elements = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim <= 0 || llvm::MulOverflow(elements, dim, elements))
        return op.emitError("neutral contract load snapshot size overflows");
    }
    Value snapshot;
    memref::AllocOp alloc;
    if (rowMajor) {
      alloc = rewriter.create<memref::AllocOp>(
          op.getLoc(),
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()),
          getSnapshotAlignment(op, rewriter));
      snapshot = alloc;
    } else {
      // LLVM allocations need identity layouts. Represent column-major B as
      // a logical [K,N] view over compact physical [N,K] backing storage.
      SmallVector<int64_t> compactStrides{1, tensorType.getDimSize(0)};
      auto bufferType = MemRefType::get(
          tensorType.getShape(), tensorType.getElementType(),
          StridedLayoutAttr::get(rewriter.getContext(), 0, compactStrides));
      alloc = rewriter.create<memref::AllocOp>(
          op.getLoc(), MemRefType::get({elements}, tensorType.getElementType()),
          getSnapshotAlignment(op, rewriter));
      snapshot = rewriter.create<memref::ReinterpretCastOp>(
          op.getLoc(), bufferType, alloc, 0, tensorType.getShape(),
          compactStrides);
    }
    alloc->setAttr(
        "contract.buffer",
        rewriter.getDictionaryAttr(
            {rewriter.getNamedAttr("role", rewriter.getStringAttr("tile")),
             rewriter.getNamedAttr("kind", rewriter.getStringAttr("temporary")),
             rewriter.getNamedAttr(
                 "logical_shape",
                 rewriter.getDenseI64ArrayAttr(tensorType.getShape()))}));
    if (op.hasMask() && op.getOther()) {
      auto fill = rewriter.create<linalg::FillOp>(
          op.getLoc(), ValueRange{op.getOther()}, ValueRange{snapshot});
      fill->setAttr("contract.fill",
                    rewriter.getDictionaryAttr({rewriter.getNamedAttr(
                        "padding", rewriter.getBoolAttr(true))}));
    }
    Value source = ptr;
    Value destination = snapshot;
    if (op.hasMask()) {
      source =
          getSubview(tensorType.getRank(), mask, ptr, op.getLoc(), rewriter);
      destination = getSubview(tensorType.getRank(), mask, snapshot,
                               op.getLoc(), rewriter);
    }
    auto copy =
        rewriter.create<memref::CopyOp>(op.getLoc(), source, destination);
    copy->setAttr("contract.access",
                  memoryProvenance(rewriter, "load", tensorType, mask,
                                   /*destination=*/{}, op.getOther(), op));
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        op.getLoc(), tensorType, snapshot, true /*restrict*/,
        true /*writable*/);
    rewriter.replaceOp(op, tensor);
    return success();
  }

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
    auto dynamicTensorType = RankedTensorType::get(sourceType.getShape(),
                                                   sourceType.getElementType());
    Value tensor = bufferization::ToTensorOp::create(
        rewriter, loc, dynamicTensorType, source, true /*restrict*/,
        false /*writable*/);
    if (tensor.getType() != targetTensorType) {
      tensor = tensor::CastOp::create(rewriter, loc, targetTensorType, tensor);
    }
    return tensor;
  }

  LogicalResult
  rewriteRank1WraparoundLoad(tts::LoadOp op,
                             UnrealizedConversionCastOp descriptorCast,
                             ConversionPatternRewriter &rewriter) const {
    auto descriptor = getRank1WraparoundDescriptor(descriptorCast);
    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!descriptor || !tensorType || tensorType.getRank() != 1 ||
        !tensorType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "expected a static rank-1 linear wraparound descriptor");
    }

    Location loc = op.getLoc();
    Type elementType = tensorType.getElementType();
    Value alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(tensorType.getShape(), elementType));
    if (op.hasMask() && op.getOther()) {
      fillWithValue(loc, alloc, op.getOther(), tensorType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        op.hasMask() ? ofrToIndexValue(op.getMixedMaskDims()[0], loc, rewriter)
                     : arith::ConstantIndexOp::create(rewriter, loc,
                                                      tensorType.getDimSize(0))
                           .getResult();
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value elementPtr = createRank1WraparoundElementMemRef(
        loc, *descriptor, loop.getInductionVar(), elementType, rewriter);
    Value loaded = memref::LoadOp::create(rewriter, loc, elementPtr,
                                          ValueRange{lowerBound});
    memref::StoreOp::create(rewriter, loc, loaded, alloc,
                            ValueRange{loop.getInductionVar()});
    rewriter.setInsertionPointAfter(loop);

    Value tensor = bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                     alloc, true /* restrict */,
                                                     true /* writable */);
    rewriter.replaceOp(op, tensor);
    return success();
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

  LogicalResult
  rewriteTensorFirstUnmaskedLoad(tts::LoadOp op, Value ptr,
                                 ConversionPatternRewriter &rewriter) const {
    auto tensorType = cast<RankedTensorType>(op.getType());

    if (!needsTensorFirstLoadSnapshot(op)) {
      Value tensor = createTensorFromMemref(op, ptr, tensorType, rewriter);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    auto loc = op->getLoc();
    auto ptrType = cast<MemRefType>(ptr.getType());
    auto allocType =
        MemRefType::get(ptrType.getShape(), ptrType.getElementType());
    SmallVector<Value> dynamicDims;
    dynamicDims.reserve(ptrType.getNumDynamicDims());
    for (int64_t i = 0, e = ptrType.getRank(); i < e; ++i) {
      if (ptrType.isDynamicDim(i)) {
        dynamicDims.push_back(memref::DimOp::create(rewriter, loc, ptr, i));
      }
    }
    auto alloc = memref::AllocOp::create(rewriter, loc, allocType, dynamicDims);
    memref::CopyOp::create(rewriter, loc, ptr, alloc);
    Value tensor = createTensorFromMemref(op, alloc, tensorType, rewriter);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  LogicalResult
  rewriteTensorFirstMaskedLoad(tts::LoadOp op, Value ptr,
                               ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());

    auto loc = op->getLoc();
    auto tensorType = cast<RankedTensorType>(op.getType());
    int64_t rank = tensorType.getRank();
    if (rank == 1) {
      auto alloc = memref::AllocOp::create(
          rewriter, loc,
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()));

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

      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, alloc, true /*restrict*/,
          true /*writable*/);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto srcSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);
    auto sliceType = cast<RankedTensorType>(
        tensor::ExtractSliceOp::inferResultType(tensorType, mixedDims));
    Value loadedSlice =
        createTensorFromMemref(op, srcSubview, sliceType, rewriter);

    auto dynamicResultDims =
        getDynamicTensorDims(loc, tensorType, ptr, rewriter);
    Value init =
        tensor::EmptyOp::create(rewriter, loc, tensorType.getShape(),
                                tensorType.getElementType(), dynamicResultDims);
    if (Value other = op.getOther()) {
      init = linalg::FillOp::create(rewriter, loc, ValueRange{other},
                                    ValueRange{init})
                 .getResult(0);
    }

    Value tensor = tensor::InsertSliceOp::create(
        rewriter, loc, loadedSlice, init, offsets, mixedDims, strides);
    rewriter.replaceOp(op, tensor);
    return success();
  }

private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));

    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{zero, block1Col},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{block1Row, zero},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstType),
                                     src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult subviewCol1 =
        memref::DimOp::create(rewriter, loc, block1, 1).getResult();
    OpFoldResult subviewCol2 =
        memref::DimOp::create(rewriter, loc, block2, 1).getResult();

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
        memref::DimOp::create(rewriter, loc, block1, 0).getResult();
    OpFoldResult subviewRow2 =
        memref::DimOp::create(rewriter, loc, block2, 0).getResult();

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getSideBySideSlices(Value source, Value block1, Value block2, Location loc,
                      ConversionPatternRewriter &rewriter) const {
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value rowSize = tensor::DimOp::create(rewriter, loc, source, 0);
    Value colSize1 = memref::DimOp::create(rewriter, loc, block1, 1);
    Value colSize2 = memref::DimOp::create(rewriter, loc, block2, 1);

    auto slice1 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, c0},
        ValueRange{rowSize, colSize1}, ValueRange{c1, c1});
    auto slice2 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, colSize1},
        ValueRange{rowSize, colSize2}, ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getStackedSlices(Value source, Value block1, Value block2, Location loc,
                   ConversionPatternRewriter &rewriter) const {
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value rowSize1 = memref::DimOp::create(rewriter, loc, block1, 0);
    Value rowSize2 = memref::DimOp::create(rewriter, loc, block2, 0);
    Value colSize = tensor::DimOp::create(rewriter, loc, source, 1);

    auto slice1 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, c0},
        ValueRange{rowSize1, colSize}, ValueRange{c1, c1});
    auto slice2 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{rowSize1, c0},
        ValueRange{rowSize2, colSize}, ValueRange{c1, c1});
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
      if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        return rewriteRank1WraparoundLoad(op, unrealizedCast, rewriter);
      }
      auto alloc = memref::AllocOp::create(
          rewriter, loc, MemRefType::get(tensorType.getShape(), elemType));
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      assert(!other && "other value used in non-masked load");
      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        createSideBySideCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        createStackedCopies(block1, block2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, alloc, true /* restrict */,
          true /* writable */);
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

    auto alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(tensorType.getShape(), elemType));

    // No mask
    assert(!other && "other value used in non-masked load");

    memref::CopyOp::create(rewriter, loc, ptr, alloc);

    Value tensor = bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                     alloc, true /* restrict */,
                                                     true /* writable */);
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
      if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        return rewriteRank1WraparoundLoad(op, unrealizedCast, rewriter);
      }
      auto mixedDims = op.getMixedMaskDims();
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];
      SmallVector<Value> dynamicResultDims;
      dynamicResultDims.reserve(tensorType.getNumDynamicDims());
      Value init = tensor::EmptyOp::create(rewriter, loc, tensorType.getShape(),
                                           elemType, dynamicResultDims);
      if (Value other = op.getOther()) {
        init = linalg::FillOp::create(rewriter, loc, ValueRange{other},
                                      ValueRange{init})
                   .getResult(0);
      }
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, block1, block2, loc, rewriter);
        auto subview1Type = cast<MemRefType>(subview1.getType());
        auto subview2Type = cast<MemRefType>(subview2.getType());
        auto slice1Type =
            RankedTensorType::get(subview1Type.getShape(), elemType);
        auto slice2Type =
            RankedTensorType::get(subview2Type.getShape(), elemType);
        auto genericSliceType = RankedTensorType::get(
            SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic},
            elemType);
        Value tensor1 =
            createTensorFromMemref(op, subview1, slice1Type, rewriter);
        Value tensor2 =
            createTensorFromMemref(op, subview2, slice2Type, rewriter);
        tensor1 =
            tensor::CastOp::create(rewriter, loc, genericSliceType, tensor1);
        tensor2 =
            tensor::CastOp::create(rewriter, loc, genericSliceType, tensor2);
        Value rowSize = memref::DimOp::create(rewriter, loc, subview1, 0);
        Value colSize1 = memref::DimOp::create(rewriter, loc, subview1, 1);
        Value colSize2 = memref::DimOp::create(rewriter, loc, subview2, 1);
        init = tensor::InsertSliceOp::create(
            rewriter, loc, tensor1, init, ValueRange{c0, c0},
            ValueRange{rowSize, colSize1}, ValueRange{c1, c1});
        init = tensor::InsertSliceOp::create(
            rewriter, loc, tensor2, init, ValueRange{c0, colSize1},
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
        auto genericSliceType = RankedTensorType::get(
            SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic},
            elemType);
        Value tensor1 =
            createTensorFromMemref(op, subview1, slice1Type, rewriter);
        Value tensor2 =
            createTensorFromMemref(op, subview2, slice2Type, rewriter);
        tensor1 =
            tensor::CastOp::create(rewriter, loc, genericSliceType, tensor1);
        tensor2 =
            tensor::CastOp::create(rewriter, loc, genericSliceType, tensor2);
        Value rowSize1 = memref::DimOp::create(rewriter, loc, subview1, 0);
        Value rowSize2 = memref::DimOp::create(rewriter, loc, subview2, 0);
        Value colSize = memref::DimOp::create(rewriter, loc, subview1, 1);
        init = tensor::InsertSliceOp::create(
            rewriter, loc, tensor1, init, ValueRange{c0, c0},
            ValueRange{rowSize1, colSize}, ValueRange{c1, c1});
        init = tensor::InsertSliceOp::create(
            rewriter, loc, tensor2, init, ValueRange{rowSize1, c0},
            ValueRange{rowSize2, colSize}, ValueRange{c1, c1});
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

    auto alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(tensorType.getShape(), elemType));

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
    memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);

    Value tensor = bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                     alloc, true /* restrict */,
                                                     true /* writable */);
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
        arith::IndexCastOp::create(rewriter, loc, indexOffsetTy, gatherOffset)
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
    auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, offsetSize).getResult();
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
        upperBound = arith::ConstantIndexOp::create(rewriter, loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, offsetSize);
        upperBound = arith::MinSIOp::create(rewriter, loc, gatherMaskDimVal,
                                            offsetSizeVal)
                         .getResult();
      }
    }
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    if (resultType.getRank() == 1) {
      SmallVector<Value> dynamicDims;
      if (resultType.isDynamicDim(0)) {
        dynamicDims.push_back(
            tensor::DimOp::create(rewriter, loc, gatherOffset, 0));
      }
      Value result =
          tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                  resultType.getElementType(), dynamicDims);
      if (Value other = op.getOther()) {
        result = linalg::FillOp::create(rewriter, loc, ValueRange{other},
                                        ValueRange{result})
                     .getResult(0);
      }

      auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound,
                                     step, ValueRange{result});
      rewriter.setInsertionPointToStart(loop.getBody());
      Value inductionVar = loop.getInductionVar();
      Value tensorAcc = loop.getRegionIterArgs()[0];
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

      auto buildInsert = [&](OpBuilder &builder, Value acc) -> Value {
        auto gatherOffsetElt = tensor::ExtractOp::create(
            builder, loc, gatherOffset, ValueRange{inductionVar});
        Value srcPtr = rewriteGatherScatterPtrElement(
            staticSizes, ptr, memRefPtr, gatherOffsetElt.getResult(), gatherDim,
            rewriter);
        Value loaded =
            memref::LoadOp::create(builder, loc, srcPtr, ValueRange{zero});
        return tensor::InsertOp::create(builder, loc, loaded, acc,
                                        ValueRange{inductionVar})
            .getResult();
      };

      Value nextTensor;
      if (Value unstructuredMask = ptr.getGatherScatterMask()) {
        auto maskValue = tensor::ExtractOp::create(
            rewriter, loc, unstructuredMask, ValueRange{inductionVar});
        auto ifOp =
            scf::IfOp::create(rewriter, loc, tensorAcc.getType(), maskValue,
                              /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          scf::YieldOp::create(rewriter, loc, buildInsert(rewriter, tensorAcc));
        }
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          scf::YieldOp::create(rewriter, loc, tensorAcc);
        }
        nextTensor = ifOp.getResult(0);
      } else {
        nextTensor = buildInsert(rewriter, tensorAcc);
      }
      scf::YieldOp::create(rewriter, loc, nextTensor);
      rewriter.replaceOp(op, loop.getResult(0));

      return success();
    }

    // Create alloc to save the result.
    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    auto alloc = memref::AllocOp::create(rewriter, loc, allocType);

    // Fill load destination with other value.
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, resultType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    // Create tensor from alloc and use it as the result to replace op.
    Value tensor = bufferization::ToTensorOp::create(
        rewriter, loc, op.getType(), alloc, true /* restrict */,
        true /* writable */);
    rewriter.replaceOp(op, tensor);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // load.
      auto maskValue = tensor::ExtractOp::create(
          rewriter, loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = scf::IfOp::create(rewriter, loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = tensor::ExtractOp::create(
        rewriter, loc, gatherOffset, ValueRange{inductionVar});

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
      srcPtr = memref::SubViewOp::create(rewriter, loc,
                                         cast<MemRefType>(dstSubViewType),
                                         srcPtr, maskOffsets, sizes, oneStrides)
                   .getResult();
    }

    // alloc[inductionVar]
    SmallVector<OpFoldResult> allocOffsets(rank, OpFoldResult(lowerBound));
    allocOffsets[gatherDim] = inductionVar;
    auto dstAllocType = memref::SubViewOp::inferResultType(
        allocType, allocOffsets, sizes, oneStrides);
    auto dstSubview =
        memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstAllocType),
                                  alloc, allocOffsets, sizes, oneStrides);
    // Copy srcPtr to alloc[inductionVar].
    memref::CopyOp::create(rewriter, loc, srcPtr, dstSubview);

    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter,
                bool enableTensorFirstVectorCpu, TritonLoweringConfig config,
                MLIRContext *context)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context),
        emitContract(config.emitContract),
        enableTensorFirstVectorCpu(enableTensorFirstVectorCpu) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto originalPtr = op.getPtr();
    if (auto gatherScatterPtr =
            originalPtr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      if (emitContract)
        return op.emitError("neutral contract gather/scatter load is "
                            "unsupported in contract v1");
      FailureOr<Value> memRefPtr = materializeGatherScatterBaseMemRef(
          gatherScatterPtr, op.getLoc(), rewriter);
      if (failed(memRefPtr)) {
        return rewriter.notifyMatchFailure(
            op, "expected gather/scatter base pointer to materialize memref");
      }
      return rewriteGather(gatherScatterPtr, op, *memRefPtr, rewriter);
    }

    auto ptr = adaptor.getPtr();
    auto makeTPtr = originalPtr.getDefiningOp<tts::MakeTensorPtrOp>();
    if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType()) && makeTPtr) {
      FailureOr<Value> materialized = failure();
      if (makeTPtr.isStructuredPtr()) {
        materialized =
            materializeStructuredTPtrMemRef(makeTPtr, op.getLoc(), rewriter);
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
    if (emitContract) {
      result = rewriteContractLogicalLoad(op, ptr, rewriter);
    } else if (isTensorFirstFastPathCandidate(op, ptr)) {
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
  bool emitContract;

private:
  bool enableTensorFirstVectorCpu;

  LogicalResult
  rewriteRank1WraparoundStore(tts::StoreOp op, Value storeValue,
                              UnrealizedConversionCastOp descriptorCast,
                              ConversionPatternRewriter &rewriter) const {
    auto descriptor = getRank1WraparoundDescriptor(descriptorCast);
    auto tensorType = dyn_cast<RankedTensorType>(storeValue.getType());
    if (!descriptor || !tensorType || tensorType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected a rank-1 linear wraparound descriptor");
    }

    Location loc = op.getLoc();
    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        op.hasMask()
            ? ofrToIndexValue(op.getMixedMaskDims()[0], loc, rewriter)
            : tensor::DimOp::create(rewriter, loc, storeValue, 0).getResult();
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value stored = tensor::ExtractOp::create(
        rewriter, loc, storeValue, ValueRange{loop.getInductionVar()});
    Value elementPtr = createRank1WraparoundElementMemRef(
        loc, *descriptor, loop.getInductionVar(), tensorType.getElementType(),
        rewriter);
    memref::StoreOp::create(rewriter, loc, stored, elementPtr,
                            ValueRange{lowerBound});
    rewriter.setInsertionPointAfter(loop);
    rewriter.eraseOp(op);
    return success();
  }

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

  LogicalResult
  rewriteTensorFirstStore(tts::StoreOp op, Value ptr, Value stVal,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto ptrSubview = createFullSubview1D(loc, ptr, rewriter);
    auto srcTensorType = cast<RankedTensorType>(stVal.getType());
    auto dynamicTensorType = RankedTensorType::get(
        {ShapedType::kDynamic}, srcTensorType.getElementType());
    if (srcTensorType != dynamicTensorType) {
      stVal = tensor::CastOp::create(rewriter, loc, dynamicTensorType, stVal);
    }
    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, stVal, ptrSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  rewriteTensorFirstMaskedStore(tts::StoreOp op, Value ptr, Value stVal,
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
    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, srcSlice, dstSubview);
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

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, dims);

    return tensor::ExtractSliceOp::create(b, loc, dstType, source, offsets,
                                          dims, strides);
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getSideBySideSlices(Value source, Value block1, Value block2, Location loc,
                      ConversionPatternRewriter &rewriter) const {
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value rowSize = tensor::DimOp::create(rewriter, loc, source, 0);
    Value colSize1 = memref::DimOp::create(rewriter, loc, block1, 1);
    Value colSize2 = memref::DimOp::create(rewriter, loc, block2, 1);

    auto slice1 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, c0},
        ValueRange{rowSize, colSize1}, ValueRange{c1, c1});
    auto slice2 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, colSize1},
        ValueRange{rowSize, colSize2}, ValueRange{c1, c1});
    return {slice1, slice2};
  }

  std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>
  getStackedSlices(Value source, Value block1, Value block2, Location loc,
                   ConversionPatternRewriter &rewriter) const {
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value rowSize1 = memref::DimOp::create(rewriter, loc, block1, 0);
    Value rowSize2 = memref::DimOp::create(rewriter, loc, block2, 0);
    Value colSize = tensor::DimOp::create(rewriter, loc, source, 1);

    auto slice1 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{c0, c0},
        ValueRange{rowSize1, colSize}, ValueRange{c1, c1});
    auto slice2 = tensor::ExtractSliceOp::create(
        rewriter, loc, source, ValueRange{rowSize1, c0},
        ValueRange{rowSize2, colSize}, ValueRange{c1, c1});
    return {slice1, slice2};
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
        arith::IndexCastOp::create(rewriter, loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, offsetSize).getResult();
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
        upperBound = arith::ConstantIndexOp::create(rewriter, loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, offsetSize);
        upperBound = arith::MinSIOp::create(rewriter, loc, gatherMaskDimVal,
                                            offsetSizeVal)
                         .getResult();
      }
    }
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    Value inductionVar = loop.getInductionVar();

    if (Value unstructuredMask = ptr.getGatherScatterMask()) {
      // If the gather scatter mask is present, we need to use it to guard the
      // store.
      auto maskValue = tensor::ExtractOp::create(
          rewriter, loc, unstructuredMask, ValueRange{inductionVar});
      auto ifOp = scf::IfOp::create(rewriter, loc, maskValue);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Load the offsetElt first.
    auto gatherOffsetElt = tensor::ExtractOp::create(
        rewriter, loc, gatherOffset, ValueRange{inductionVar});

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
    auto slice = tensor::ExtractSliceOp::create(
        rewriter, loc, stVal, stValOffsets, sizes, oneStrides);

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
          memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstType),
                                    dstPtr, maskOffsets, sizes, oneStrides)
              .getResult();
    }
    // store slice to dstPtr.
    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, slice, dstPtr);
    storeOp.setWritable(true);

    rewriter.eraseOp(op);

    return success();
  }

public:
  StoreConverter(const TypeConverter &typeConverter,
                 bool enableTensorFirstVectorCpu, TritonLoweringConfig config,
                 MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context),
        emitContract(config.emitContract),
        enableTensorFirstVectorCpu(enableTensorFirstVectorCpu) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto originalPtr = op.getPtr();
    if (auto gatherScatterPtr =
            originalPtr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      if (emitContract)
        return op.emitError("neutral contract gather/scatter store is "
                            "unsupported in contract v1");
      FailureOr<Value> memRefPtr = materializeGatherScatterBaseMemRef(
          gatherScatterPtr, op.getLoc(), rewriter);
      if (failed(memRefPtr)) {
        return rewriter.notifyMatchFailure(
            op, "expected gather/scatter base pointer to materialize memref");
      }
      return rewriteScatter(gatherScatterPtr, op, *memRefPtr,
                            adaptor.getValue(), rewriter);
    }

    auto ptr = adaptor.getPtr();
    auto makeTPtr = originalPtr.getDefiningOp<tts::MakeTensorPtrOp>();
    if (!isa<MemRefType, UnrankedMemRefType>(ptr.getType()) && makeTPtr) {
      FailureOr<Value> materialized = failure();
      if (makeTPtr.isStructuredPtr()) {
        materialized =
            materializeStructuredTPtrMemRef(makeTPtr, op.getLoc(), rewriter);
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

    if (emitContract) {
      auto tensorType = cast<RankedTensorType>(adaptor.getValue().getType());
      if (!tensorType.hasStaticShape() || getWraparoundCast(ptr))
        return op.emitError(
            "neutral contract store requires a static rectangular DDR "
            "view; split/wrapping pointers are unsupported");
      auto ranked =
          ensureRankedMemRef(ptr, tensorType.getRank(),
                             tensorType.getElementType(), loc, rewriter);
      if (failed(ranked))
        return op.emitError(
            "neutral contract store requires a ranked DDR view");
      ptr = *ranked;
      // The bridge exposes the ordinary logical producer. Bufferization
      // resolves it before the independent planner chooses a physical route.
      auto bufferType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      bufferization::BufferizationOptions options;
      bufferization::BufferizationState state;
      if (auto inferred =
              bufferization::getBufferType(adaptor.getValue(), options, state);
          succeeded(inferred))
        if (auto existingType = dyn_cast<MemRefType>(*inferred))
          if (existingType.getMemorySpace())
            bufferType = existingType;
      Value source = rewriter.create<bufferization::ToBufferOp>(
          loc, bufferType, adaptor.getValue(), /*read_only=*/true);
      Value destination = ptr;
      SmallVector<OpFoldResult> mask =
          op.hasMask() ? op.getMixedMaskDims() : SmallVector<OpFoldResult>{};
      if (op.hasMask()) {
        source = getSubview(tensorType.getRank(), mask, source, loc, rewriter);
        destination =
            getSubview(tensorType.getRank(), mask, ptr, loc, rewriter);
      }
      auto copy = rewriter.create<memref::CopyOp>(loc, source, destination);
      copy->setAttr("contract.access",
                    memoryProvenance(rewriter, "store", tensorType, mask, ptr,
                                     /*other=*/{}, op));
      rewriter.eraseOp(op);
      if (makeTPtr && makeTPtr->use_empty())
        rewriter.eraseOp(makeTPtr);
      return success();
    }

    if (isTensorFirstFastPathCandidate(op, ptr)) {
      auto res =
          op.hasMask()
              ? rewriteTensorFirstMaskedStore(op, ptr, adaptor.getValue(),
                                              rewriter)
              : rewriteTensorFirstStore(op, ptr, adaptor.getValue(), rewriter);
      if (succeeded(res) && makeTPtr && makeTPtr->use_empty()) {
        rewriter.eraseOp(makeTPtr);
      }
      return res;
    }

    if (auto unrealizedCast = getWraparoundCast(ptr)) {
      if (unrealizedCast->hasAttr(WRAP_LINEAR)) {
        auto result = rewriteRank1WraparoundStore(op, adaptor.getValue(),
                                                  unrealizedCast, rewriter);
        if (succeeded(result) && makeTPtr && makeTPtr->use_empty()) {
          rewriter.eraseOp(makeTPtr);
        }
        return result;
      }
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (op.hasMask()) {
        return rewriter.notifyMatchFailure(
            op, "masked split-pointer store is not implemented");
      }

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [slice1, slice2] = getSideBySideSlices(adaptor.getValue(), block1,
                                                    block2, loc, rewriter);
        auto store1 = bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, slice1, block1);
        store1.setWritable(true);
        auto store2 = bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, slice2, block2);
        store2.setWritable(true);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [slice1, slice2] =
            getStackedSlices(adaptor.getValue(), block1, block2, loc, rewriter);
        auto store1 = bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, slice1, block1);
        store1.setWritable(true);
        auto store2 = bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, slice2, block2);
        store2.setWritable(true);
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
    auto rankedPtr = ensureRankedMemRef(
        ptr, rank, storeTensorType.getElementType(), loc, rewriter);
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

      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, storeValue, ptr);
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
  patterns.add<MaskedElementwiseStoreFusionPattern>(patterns.getContext(),
                                                    PatternBenefit(10));
}

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    bool enableTensorFirstVectorCpu, TritonLoweringConfig config) {
  patterns.add<MakeTensorPtrConverter, MakeGatherScatterTensorPtrConverter>(
      typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(
      typeConverter, enableTensorFirstVectorCpu, config, patterns.getContext());
}
