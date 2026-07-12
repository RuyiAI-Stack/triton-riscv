#include "triton-shared/Transform/ExpandFloat8Conversions/ExpandFloat8Conversions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "expand-float8-conversions"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_EXPANDFLOAT8CONVERSIONS
#include "triton-shared/Transform/ExpandFloat8Conversions/Passes.h.inc"

namespace {

static bool isFloat(Type type) { return isa<FloatType>(type); }

static bool isFixedVectorOf(VectorType type, Type elementType) {
  return type && !type.isScalable() && type.getElementType() == elementType;
}

static bool isScalarOrFixedVectorOf(Type type, Type elementType) {
  if (type == elementType)
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && isFixedVectorOf(vectorType, elementType);
}

static bool isScalarOrFixedVectorOfFloat(Type type) {
  if (isFloat(type))
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && !vectorType.isScalable() &&
         isFloat(vectorType.getElementType());
}

static FloatType getFloatElementType(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type))
    type = vectorType.getElementType();
  return cast<FloatType>(type);
}

static Type withElementType(Type type, Type elementType) {
  if (auto vectorType = dyn_cast<VectorType>(type))
    return vectorType.clone(elementType);
  return elementType;
}

static bool hasSameFixedVectorShape(Type lhs, Type rhs) {
  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (!lhsVector || !rhsVector)
    return !lhsVector && !rhsVector;
  return !lhsVector.isScalable() && !rhsVector.isScalable() &&
         lhsVector.getShape() == rhsVector.getShape();
}

static Value createIntConstant(Location loc, Type type, int64_t value,
                               OpBuilder &builder) {
  Type elementType = type;
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    elementType = vectorType.getElementType();
    auto scalar = builder.getIntegerAttr(elementType, value);
    auto splat = DenseElementsAttr::get(vectorType, scalar);
    return builder.create<arith::ConstantOp>(loc, type, splat);
  }
  return builder.create<arith::ConstantIntOp>(
      loc, cast<IntegerType>(elementType), value);
}

static Value createCmp(Location loc, arith::CmpIPredicate predicate, Value lhs,
                       Value rhs, OpBuilder &builder) {
  return builder.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
}

static Value castToF32(Location loc, Value value, OpBuilder &builder) {
  Type f32Type = withElementType(value.getType(), builder.getF32Type());
  if (value.getType() == f32Type)
    return value;
  return builder.create<arith::ExtFOp>(loc, f32Type, value);
}

static Value castFromF32(Location loc, Value value, Type targetType,
                         OpBuilder &builder) {
  if (value.getType() == targetType)
    return value;
  if (getFloatElementType(targetType).getWidth() > 32)
    return builder.create<arith::ExtFOp>(loc, targetType, value);
  return builder.create<arith::TruncFOp>(loc, targetType, value);
}

static Value expandFp8ToF32(Location loc, Value fp8, OpBuilder &builder) {
  Type i8Type = withElementType(fp8.getType(), builder.getI8Type());
  Type i32Type = withElementType(fp8.getType(), builder.getI32Type());
  Type f32Type = withElementType(fp8.getType(), builder.getF32Type());

  Value bits8 = builder.create<arith::BitcastOp>(loc, i8Type, fp8);
  Value bits = builder.create<arith::ExtUIOp>(loc, i32Type, bits8);
  auto constant = [&](int64_t value) {
    return createIntConstant(loc, i32Type, value, builder);
  };

  Value sign = builder.create<arith::AndIOp>(loc, bits, constant(0x80));
  sign = builder.create<arith::ShLIOp>(loc, sign, constant(24));
  Value exponent = builder.create<arith::ShRUIOp>(loc, bits, constant(3));
  exponent = builder.create<arith::AndIOp>(loc, exponent, constant(0x0f));
  Value mantissa = builder.create<arith::AndIOp>(loc, bits, constant(0x07));

  Value normalExponent =
      builder.create<arith::AddIOp>(loc, exponent, constant(120));
  normalExponent =
      builder.create<arith::ShLIOp>(loc, normalExponent, constant(23));
  Value normalMantissa =
      builder.create<arith::ShLIOp>(loc, mantissa, constant(20));
  Value normal =
      builder.create<arith::OrIOp>(loc, normalExponent, normalMantissa);

  Value highSubMantissa =
      builder.create<arith::SubIOp>(loc, mantissa, constant(4));
  highSubMantissa =
      builder.create<arith::ShLIOp>(loc, highSubMantissa, constant(21));
  Value highSub = builder.create<arith::OrIOp>(
      loc, constant(int64_t{120} << 23), highSubMantissa);

  Value midSubMantissa =
      builder.create<arith::SubIOp>(loc, mantissa, constant(2));
  midSubMantissa =
      builder.create<arith::ShLIOp>(loc, midSubMantissa, constant(22));
  Value midSub = builder.create<arith::OrIOp>(loc, constant(int64_t{119} << 23),
                                              midSubMantissa);

  Value mantissaGe4 =
      createCmp(loc, arith::CmpIPredicate::uge, mantissa, constant(4), builder);
  Value mantissaGe2 =
      createCmp(loc, arith::CmpIPredicate::uge, mantissa, constant(2), builder);
  Value mantissaEq1 =
      createCmp(loc, arith::CmpIPredicate::eq, mantissa, constant(1), builder);
  Value subnormal = builder.create<arith::SelectOp>(
      loc, mantissaEq1, constant(int64_t{118} << 23), constant(0));
  subnormal =
      builder.create<arith::SelectOp>(loc, mantissaGe2, midSub, subnormal);
  subnormal =
      builder.create<arith::SelectOp>(loc, mantissaGe4, highSub, subnormal);

  Value exponentIsZero =
      createCmp(loc, arith::CmpIPredicate::eq, exponent, constant(0), builder);
  Value magnitude =
      builder.create<arith::SelectOp>(loc, exponentIsZero, subnormal, normal);
  Value exponentIs15 =
      createCmp(loc, arith::CmpIPredicate::eq, exponent, constant(15), builder);
  Value mantissaIs7 =
      createCmp(loc, arith::CmpIPredicate::eq, mantissa, constant(7), builder);
  Value isNaN = builder.create<arith::AndIOp>(loc, exponentIs15, mantissaIs7);
  magnitude = builder.create<arith::SelectOp>(loc, isNaN, constant(0x7fc00000),
                                              magnitude);
  Value f32Bits = builder.create<arith::OrIOp>(loc, sign, magnitude);
  return builder.create<arith::BitcastOp>(loc, f32Type, f32Bits);
}

static Value expandFpToFp8(Location loc, Value value, Type resultType,
                           OpBuilder &builder) {
  Value f32 = castToF32(loc, value, builder);
  Type i32Type = withElementType(f32.getType(), builder.getI32Type());
  Type i8Type = withElementType(f32.getType(), builder.getI8Type());
  auto constant = [&](int64_t value) {
    return createIntConstant(loc, i32Type, value, builder);
  };

  Value bits = builder.create<arith::BitcastOp>(loc, i32Type, f32);
  Value sign = builder.create<arith::ShRUIOp>(loc, bits, constant(24));
  sign = builder.create<arith::AndIOp>(loc, sign, constant(0x80));
  Value magnitude =
      builder.create<arith::AndIOp>(loc, bits, constant(0x7fffffff));
  Value exponent = builder.create<arith::ShRUIOp>(loc, magnitude, constant(23));
  exponent = builder.create<arith::AndIOp>(loc, exponent, constant(0xff));
  Value significand =
      builder.create<arith::AndIOp>(loc, magnitude, constant(0x7fffff));
  significand =
      builder.create<arith::OrIOp>(loc, significand, constant(0x800000));

  Value normalRounded =
      builder.create<arith::ShRUIOp>(loc, significand, constant(20));
  Value normalRemainder =
      builder.create<arith::AndIOp>(loc, significand, constant(0xfffff));
  Value normalGreater = createCmp(loc, arith::CmpIPredicate::ugt,
                                  normalRemainder, constant(0x80000), builder);
  Value normalTie = createCmp(loc, arith::CmpIPredicate::eq, normalRemainder,
                              constant(0x80000), builder);
  Value normalOdd =
      builder.create<arith::AndIOp>(loc, normalRounded, constant(1));
  normalOdd =
      createCmp(loc, arith::CmpIPredicate::ne, normalOdd, constant(0), builder);
  Value normalRoundUp =
      builder.create<arith::AndIOp>(loc, normalTie, normalOdd);
  normalRoundUp =
      builder.create<arith::OrIOp>(loc, normalGreater, normalRoundUp);
  Value normalRoundBit =
      builder.create<arith::ExtUIOp>(loc, i32Type, normalRoundUp);
  normalRounded =
      builder.create<arith::AddIOp>(loc, normalRounded, normalRoundBit);
  Value normalCarry = createCmp(loc, arith::CmpIPredicate::eq, normalRounded,
                                constant(16), builder);
  Value normalCarryBit =
      builder.create<arith::ExtUIOp>(loc, i32Type, normalCarry);
  Value encodedExponent =
      builder.create<arith::SubIOp>(loc, exponent, constant(120));
  encodedExponent =
      builder.create<arith::AddIOp>(loc, encodedExponent, normalCarryBit);
  encodedExponent =
      builder.create<arith::ShLIOp>(loc, encodedExponent, constant(3));
  Value encodedMantissa =
      builder.create<arith::AndIOp>(loc, normalRounded, constant(7));
  Value normal =
      builder.create<arith::OrIOp>(loc, encodedExponent, encodedMantissa);

  Value isTooSmall = createCmp(loc, arith::CmpIPredicate::ult, exponent,
                               constant(117), builder);
  Value safeExponent =
      builder.create<arith::SelectOp>(loc, isTooSmall, constant(117), exponent);
  Value shift = builder.create<arith::SubIOp>(loc, constant(141), safeExponent);
  Value subRounded = builder.create<arith::ShRUIOp>(loc, significand, shift);
  Value oneAtShift = builder.create<arith::ShLIOp>(loc, constant(1), shift);
  Value subMask = builder.create<arith::SubIOp>(loc, oneAtShift, constant(1));
  Value subRemainder = builder.create<arith::AndIOp>(loc, significand, subMask);
  Value halfShift = builder.create<arith::SubIOp>(loc, shift, constant(1));
  Value subHalf = builder.create<arith::ShLIOp>(loc, constant(1), halfShift);
  Value subGreater =
      createCmp(loc, arith::CmpIPredicate::ugt, subRemainder, subHalf, builder);
  Value subTie =
      createCmp(loc, arith::CmpIPredicate::eq, subRemainder, subHalf, builder);
  Value subOdd = builder.create<arith::AndIOp>(loc, subRounded, constant(1));
  subOdd =
      createCmp(loc, arith::CmpIPredicate::ne, subOdd, constant(0), builder);
  Value subRoundUp = builder.create<arith::AndIOp>(loc, subTie, subOdd);
  subRoundUp = builder.create<arith::OrIOp>(loc, subGreater, subRoundUp);
  Value subRoundBit = builder.create<arith::ExtUIOp>(loc, i32Type, subRoundUp);
  subRounded = builder.create<arith::AddIOp>(loc, subRounded, subRoundBit);
  Value subnormal =
      builder.create<arith::SelectOp>(loc, isTooSmall, constant(0), subRounded);

  Value isNormal = createCmp(loc, arith::CmpIPredicate::uge, magnitude,
                             constant(0x3c800000), builder);
  Value encoded =
      builder.create<arith::SelectOp>(loc, isNormal, normal, subnormal);
  Value exponentIsAllOnes = createCmp(loc, arith::CmpIPredicate::eq, exponent,
                                      constant(0xff), builder);
  Value hasPayload = createCmp(loc, arith::CmpIPredicate::ne, significand,
                               constant(0x800000), builder);
  Value isNaN =
      builder.create<arith::AndIOp>(loc, exponentIsAllOnes, hasPayload);
  Value overflows = createCmp(loc, arith::CmpIPredicate::ugt, magnitude,
                              constant(0x43e80000), builder);
  encoded =
      builder.create<arith::SelectOp>(loc, overflows, constant(0x7e), encoded);
  encoded =
      builder.create<arith::SelectOp>(loc, isNaN, constant(0x7f), encoded);
  Value isZero =
      createCmp(loc, arith::CmpIPredicate::eq, magnitude, constant(0), builder);
  encoded = builder.create<arith::SelectOp>(loc, isZero, constant(0), encoded);
  encoded = builder.create<arith::OrIOp>(loc, sign, encoded);

  Value bits8 = builder.create<arith::TruncIOp>(loc, i8Type, encoded);
  return builder.create<arith::BitcastOp>(loc, resultType, bits8);
}

class ExpandFloat8ConversionsPass
    : public ::impl::ExpandFloat8ConversionsBase<ExpandFloat8ConversionsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Type fp8Type = Float8E4M3FNType::get(context);

    SmallVector<arith::ExtFOp> fp8ExtOps;
    SmallVector<arith::TruncFOp> fp8TruncOps;
    bool hasUnsupportedConversion = false;
    module.walk([&](arith::ExtFOp op) {
      if (isScalarOrFixedVectorOf(op.getOperand().getType(), fp8Type) &&
          isScalarOrFixedVectorOfFloat(op.getType()) &&
          hasSameFixedVectorShape(op.getOperand().getType(), op.getType()))
        fp8ExtOps.push_back(op);
    });
    module.walk([&](arith::TruncFOp op) {
      if (isScalarOrFixedVectorOfFloat(op.getOperand().getType()) &&
          isScalarOrFixedVectorOf(op.getType(), fp8Type) &&
          hasSameFixedVectorShape(op.getOperand().getType(), op.getType())) {
        if (getFloatElementType(op.getOperand().getType()).getWidth() <= 32) {
          fp8TruncOps.push_back(op);
          return;
        }
        op.emitError("conversion from a float wider than f32 to f8E4M3FN is "
                     "unsupported");
        hasUnsupportedConversion = true;
      }
    });

    if (hasUnsupportedConversion) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(context);
    for (arith::ExtFOp op : fp8ExtOps) {
      rewriter.setInsertionPoint(op);
      Value f32 = expandFp8ToF32(op.getLoc(), op.getOperand(), rewriter);
      rewriter.replaceOp(op,
                         castFromF32(op.getLoc(), f32, op.getType(), rewriter));
    }

    for (arith::TruncFOp op : fp8TruncOps) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOp(op, expandFpToFp8(op.getLoc(), op.getOperand(),
                                           op.getType(), rewriter));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createExpandFloat8ConversionsPass() {
  return std::make_unique<ExpandFloat8ConversionsPass>();
}
