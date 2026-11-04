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
    return arith::ConstantOp::create(builder, loc, type, splat);
  }
  return arith::ConstantIntOp::create(builder, loc,
                                      cast<IntegerType>(elementType), value);
}

static Value createCmp(Location loc, arith::CmpIPredicate predicate, Value lhs,
                       Value rhs, OpBuilder &builder) {
  return arith::CmpIOp::create(builder, loc, predicate, lhs, rhs);
}

static Value castToF32(Location loc, Value value, OpBuilder &builder) {
  Type f32Type = withElementType(value.getType(), builder.getF32Type());
  if (value.getType() == f32Type)
    return value;
  return arith::ExtFOp::create(builder, loc, f32Type, value,
                               arith::FastMathFlagsAttr());
}

static Value castFromF32(Location loc, Value value, Type targetType,
                         OpBuilder &builder) {
  if (value.getType() == targetType)
    return value;
  if (getFloatElementType(targetType).getWidth() > 32)
    return arith::ExtFOp::create(builder, loc, targetType, value,
                                 arith::FastMathFlagsAttr());
  return arith::TruncFOp::create(builder, loc, targetType, value);
}

static Value expandFp8ToF32(Location loc, Value fp8, OpBuilder &builder) {
  Type i8Type = withElementType(fp8.getType(), builder.getI8Type());
  Type i32Type = withElementType(fp8.getType(), builder.getI32Type());
  Type f32Type = withElementType(fp8.getType(), builder.getF32Type());

  Value bits8 = arith::BitcastOp::create(builder, loc, i8Type, fp8);
  Value bits = arith::ExtUIOp::create(builder, loc, i32Type, bits8);
  auto constant = [&](int64_t value) {
    return createIntConstant(loc, i32Type, value, builder);
  };

  Value sign = arith::AndIOp::create(builder, loc, bits, constant(0x80));
  sign = arith::ShLIOp::create(builder, loc, sign, constant(24));
  Value exponent = arith::ShRUIOp::create(builder, loc, bits, constant(3));
  exponent = arith::AndIOp::create(builder, loc, exponent, constant(0x0f));
  Value mantissa = arith::AndIOp::create(builder, loc, bits, constant(0x07));

  Value normalExponent =
      arith::AddIOp::create(builder, loc, exponent, constant(120));
  normalExponent =
      arith::ShLIOp::create(builder, loc, normalExponent, constant(23));
  Value normalMantissa =
      arith::ShLIOp::create(builder, loc, mantissa, constant(20));
  Value normal =
      arith::OrIOp::create(builder, loc, normalExponent, normalMantissa);

  Value highSubMantissa =
      arith::SubIOp::create(builder, loc, mantissa, constant(4));
  highSubMantissa =
      arith::ShLIOp::create(builder, loc, highSubMantissa, constant(21));
  Value highSub = arith::OrIOp::create(
      builder, loc, constant(int64_t{120} << 23), highSubMantissa);

  Value midSubMantissa =
      arith::SubIOp::create(builder, loc, mantissa, constant(2));
  midSubMantissa =
      arith::ShLIOp::create(builder, loc, midSubMantissa, constant(22));
  Value midSub = arith::OrIOp::create(
      builder, loc, constant(int64_t{119} << 23), midSubMantissa);

  Value mantissaGe4 =
      createCmp(loc, arith::CmpIPredicate::uge, mantissa, constant(4), builder);
  Value mantissaGe2 =
      createCmp(loc, arith::CmpIPredicate::uge, mantissa, constant(2), builder);
  Value mantissaEq1 =
      createCmp(loc, arith::CmpIPredicate::eq, mantissa, constant(1), builder);
  Value subnormal = arith::SelectOp::create(
      builder, loc, mantissaEq1, constant(int64_t{118} << 23), constant(0));
  subnormal =
      arith::SelectOp::create(builder, loc, mantissaGe2, midSub, subnormal);
  subnormal =
      arith::SelectOp::create(builder, loc, mantissaGe4, highSub, subnormal);

  Value exponentIsZero =
      createCmp(loc, arith::CmpIPredicate::eq, exponent, constant(0), builder);
  Value magnitude =
      arith::SelectOp::create(builder, loc, exponentIsZero, subnormal, normal);
  Value exponentIs15 =
      createCmp(loc, arith::CmpIPredicate::eq, exponent, constant(15), builder);
  Value mantissaIs7 =
      createCmp(loc, arith::CmpIPredicate::eq, mantissa, constant(7), builder);
  Value isNaN = arith::AndIOp::create(builder, loc, exponentIs15, mantissaIs7);
  magnitude = arith::SelectOp::create(builder, loc, isNaN, constant(0x7fc00000),
                                      magnitude);
  Value f32Bits = arith::OrIOp::create(builder, loc, sign, magnitude);
  return arith::BitcastOp::create(builder, loc, f32Type, f32Bits);
}

static Value expandFpToFp8(Location loc, Value value, Type resultType,
                           OpBuilder &builder) {
  Value f32 = castToF32(loc, value, builder);
  Type i32Type = withElementType(f32.getType(), builder.getI32Type());
  Type i8Type = withElementType(f32.getType(), builder.getI8Type());
  auto constant = [&](int64_t value) {
    return createIntConstant(loc, i32Type, value, builder);
  };

  Value bits = arith::BitcastOp::create(builder, loc, i32Type, f32);
  Value sign = arith::ShRUIOp::create(builder, loc, bits, constant(24));
  sign = arith::AndIOp::create(builder, loc, sign, constant(0x80));
  Value magnitude =
      arith::AndIOp::create(builder, loc, bits, constant(0x7fffffff));
  Value exponent =
      arith::ShRUIOp::create(builder, loc, magnitude, constant(23));
  exponent = arith::AndIOp::create(builder, loc, exponent, constant(0xff));
  Value significand =
      arith::AndIOp::create(builder, loc, magnitude, constant(0x7fffff));
  significand =
      arith::OrIOp::create(builder, loc, significand, constant(0x800000));

  Value normalRounded =
      arith::ShRUIOp::create(builder, loc, significand, constant(20));
  Value normalRemainder =
      arith::AndIOp::create(builder, loc, significand, constant(0xfffff));
  Value normalGreater = createCmp(loc, arith::CmpIPredicate::ugt,
                                  normalRemainder, constant(0x80000), builder);
  Value normalTie = createCmp(loc, arith::CmpIPredicate::eq, normalRemainder,
                              constant(0x80000), builder);
  Value normalOdd =
      arith::AndIOp::create(builder, loc, normalRounded, constant(1));
  normalOdd =
      createCmp(loc, arith::CmpIPredicate::ne, normalOdd, constant(0), builder);
  Value normalRoundUp =
      arith::AndIOp::create(builder, loc, normalTie, normalOdd);
  normalRoundUp =
      arith::OrIOp::create(builder, loc, normalGreater, normalRoundUp);
  Value normalRoundBit =
      arith::ExtUIOp::create(builder, loc, i32Type, normalRoundUp);
  normalRounded =
      arith::AddIOp::create(builder, loc, normalRounded, normalRoundBit);
  Value normalCarry = createCmp(loc, arith::CmpIPredicate::eq, normalRounded,
                                constant(16), builder);
  Value normalCarryBit =
      arith::ExtUIOp::create(builder, loc, i32Type, normalCarry);
  Value encodedExponent =
      arith::SubIOp::create(builder, loc, exponent, constant(120));
  encodedExponent =
      arith::AddIOp::create(builder, loc, encodedExponent, normalCarryBit);
  encodedExponent =
      arith::ShLIOp::create(builder, loc, encodedExponent, constant(3));
  Value encodedMantissa =
      arith::AndIOp::create(builder, loc, normalRounded, constant(7));
  Value normal =
      arith::OrIOp::create(builder, loc, encodedExponent, encodedMantissa);

  Value isTooSmall = createCmp(loc, arith::CmpIPredicate::ult, exponent,
                               constant(117), builder);
  Value safeExponent = arith::SelectOp::create(builder, loc, isTooSmall,
                                               constant(117), exponent);
  Value shift =
      arith::SubIOp::create(builder, loc, constant(141), safeExponent);
  Value subRounded = arith::ShRUIOp::create(builder, loc, significand, shift);
  Value oneAtShift = arith::ShLIOp::create(builder, loc, constant(1), shift);
  Value subMask = arith::SubIOp::create(builder, loc, oneAtShift, constant(1));
  Value subRemainder =
      arith::AndIOp::create(builder, loc, significand, subMask);
  Value halfShift = arith::SubIOp::create(builder, loc, shift, constant(1));
  Value subHalf = arith::ShLIOp::create(builder, loc, constant(1), halfShift);
  Value subGreater =
      createCmp(loc, arith::CmpIPredicate::ugt, subRemainder, subHalf, builder);
  Value subTie =
      createCmp(loc, arith::CmpIPredicate::eq, subRemainder, subHalf, builder);
  Value subOdd = arith::AndIOp::create(builder, loc, subRounded, constant(1));
  subOdd =
      createCmp(loc, arith::CmpIPredicate::ne, subOdd, constant(0), builder);
  Value subRoundUp = arith::AndIOp::create(builder, loc, subTie, subOdd);
  subRoundUp = arith::OrIOp::create(builder, loc, subGreater, subRoundUp);
  Value subRoundBit = arith::ExtUIOp::create(builder, loc, i32Type, subRoundUp);
  subRounded = arith::AddIOp::create(builder, loc, subRounded, subRoundBit);
  Value subnormal = arith::SelectOp::create(builder, loc, isTooSmall,
                                            constant(0), subRounded);

  Value isNormal = createCmp(loc, arith::CmpIPredicate::uge, magnitude,
                             constant(0x3c800000), builder);
  Value encoded =
      arith::SelectOp::create(builder, loc, isNormal, normal, subnormal);
  Value exponentIsAllOnes = createCmp(loc, arith::CmpIPredicate::eq, exponent,
                                      constant(0xff), builder);
  Value hasPayload = createCmp(loc, arith::CmpIPredicate::ne, significand,
                               constant(0x800000), builder);
  Value isNaN =
      arith::AndIOp::create(builder, loc, exponentIsAllOnes, hasPayload);
  Value overflows = createCmp(loc, arith::CmpIPredicate::ugt, magnitude,
                              constant(0x43e80000), builder);
  encoded =
      arith::SelectOp::create(builder, loc, overflows, constant(0x7e), encoded);
  encoded =
      arith::SelectOp::create(builder, loc, isNaN, constant(0x7f), encoded);
  Value isZero =
      createCmp(loc, arith::CmpIPredicate::eq, magnitude, constant(0), builder);
  encoded = arith::SelectOp::create(builder, loc, isZero, constant(0), encoded);
  encoded = arith::OrIOp::create(builder, loc, sign, encoded);

  Value bits8 = arith::TruncIOp::create(builder, loc, i8Type, encoded);
  return arith::BitcastOp::create(builder, loc, resultType, bits8);
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
