#include "triton-shared/Transform/ExpandFloat8Conversions/ExpandFloat8Conversions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "expand-float8-conversions"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Transform/ExpandFloat8Conversions/Passes.h.inc"

namespace {

constexpr StringLiteral kFp8ToF32Name = "__triton_shared_fp8e4nv_to_f32";
constexpr StringLiteral kF32ToFp8Name = "__triton_shared_f32_to_fp8e4nv";

static FlatSymbolRefAttr getOrCreateHelper(ModuleOp module, StringRef name,
                                           TypeRange inputs, TypeRange results,
                                           OpBuilder &builder) {
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto functionType = builder.getFunctionType(inputs, results);
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), name, functionType);
    func.setPrivate();
  }

  return SymbolRefAttr::get(builder.getContext(), name);
}

static bool isFp8E4M3FN(Type type) { return isa<Float8E4M3FNType>(type); }

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

static bool hasSameFixedVectorShape(Type lhs, Type rhs) {
  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (!lhsVector || !rhsVector)
    return !lhsVector && !rhsVector;
  return !lhsVector.isScalable() && !rhsVector.isScalable() &&
         lhsVector.getShape() == rhsVector.getShape();
}

static Value castToF32(Location loc, Value value, Type f32Type,
                       IRRewriter &rewriter) {
  Type type = value.getType();
  if (type == f32Type)
    return value;
  return rewriter.create<arith::ExtFOp>(loc, f32Type, value);
}

static Value castFromF32(Location loc, Value value, Type targetType,
                         IRRewriter &rewriter) {
  if (value.getType() == targetType)
    return value;
  return rewriter.create<arith::TruncFOp>(loc, targetType, value);
}

static Value createF32ToFp8Scalar(Location loc, Value f32,
                                  FlatSymbolRefAttr helper, Type i8Type,
                                  IRRewriter &rewriter) {
  auto call = rewriter.create<func::CallOp>(loc, helper, i8Type, f32);
  return rewriter.create<arith::BitcastOp>(
      loc, Float8E4M3FNType::get(i8Type.getContext()), call.getResult(0));
}

static Value createFp8ToF32Scalar(Location loc, Value fp8,
                                  FlatSymbolRefAttr helper, Type i8Type,
                                  Type f32Type, IRRewriter &rewriter) {
  Value bits = rewriter.create<arith::BitcastOp>(loc, i8Type, fp8);
  auto call = rewriter.create<func::CallOp>(loc, helper, f32Type, bits);
  return call.getResult(0);
}

static Value createFpToFp8Scalar(Location loc, Value value,
                                 FlatSymbolRefAttr helper, Type i8Type,
                                 Type f32Type, IRRewriter &rewriter) {
  Value f32 = castToF32(loc, value, f32Type, rewriter);
  return createF32ToFp8Scalar(loc, f32, helper, i8Type, rewriter);
}

static Value createFp8ToFpScalar(Location loc, Value fp8,
                                 FlatSymbolRefAttr helper, Type i8Type,
                                 Type f32Type, Type targetType,
                                 IRRewriter &rewriter) {
  Value f32 = createFp8ToF32Scalar(loc, fp8, helper, i8Type, f32Type, rewriter);
  return castFromF32(loc, f32, targetType, rewriter);
}

static Value createZeroVector(Location loc, VectorType vectorType, Type i8Type,
                              Type f32Type, IRRewriter &rewriter) {
  Type elementType = vectorType.getElementType();
  Value zero;
  if (isFloat(elementType)) {
    auto floatType = cast<FloatType>(elementType);
    zero = rewriter.create<arith::ConstantFloatOp>(
        loc, floatType, APFloat::getZero(floatType.getFloatSemantics()));
  } else {
    assert(isFp8E4M3FN(elementType) && "unexpected fp8 expansion vector type");
    Value zeroBits = rewriter.create<arith::ConstantIntOp>(loc, i8Type, 0);
    zero = rewriter.create<arith::BitcastOp>(loc, elementType, zeroBits);
  }
  return rewriter.create<vector::SplatOp>(loc, vectorType, zero);
}

static Value expandFpToFp8(Location loc, Value fpValue, Type resultType,
                           FlatSymbolRefAttr helper, Type i8Type, Type f32Type,
                           IRRewriter &rewriter) {
  auto vectorType = dyn_cast<VectorType>(resultType);
  if (!vectorType)
    return createFpToFp8Scalar(loc, fpValue, helper, i8Type, f32Type, rewriter);

  int64_t numElements = vectorType.getNumElements();
  auto v1dType = VectorType::get({numElements}, vectorType.getElementType());
  auto v1dInputType = VectorType::get(
      {numElements}, cast<VectorType>(fpValue.getType()).getElementType());

  Value input1d =
      rewriter.create<vector::ShapeCastOp>(loc, v1dInputType, fpValue);
  Value result1d = createZeroVector(loc, v1dType, i8Type, f32Type, rewriter);

  Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = rewriter.create<arith::ConstantIndexOp>(loc, numElements);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto forOp = rewriter.create<scf::ForOp>(
      loc, lb, ub, step, ValueRange{result1d},
      [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
        Value scalar = b.create<vector::ExtractElementOp>(l, input1d, iv);
        Value converted =
            createFpToFp8Scalar(l, scalar, helper, i8Type, f32Type, rewriter);
        Value updated =
            b.create<vector::InsertElementOp>(l, converted, iterArgs[0], iv);
        b.create<scf::YieldOp>(l, updated);
      });

  return rewriter.create<vector::ShapeCastOp>(loc, vectorType,
                                              forOp.getResult(0));
}

static Value expandFp8ToFp(Location loc, Value fp8Value, Type resultType,
                           FlatSymbolRefAttr helper, Type i8Type, Type f32Type,
                           IRRewriter &rewriter) {
  auto vectorType = dyn_cast<VectorType>(resultType);
  if (!vectorType)
    return createFp8ToFpScalar(loc, fp8Value, helper, i8Type, f32Type,
                               resultType, rewriter);

  int64_t numElements = vectorType.getNumElements();
  auto v1dType = VectorType::get({numElements}, vectorType.getElementType());
  auto v1dInputType = VectorType::get(
      {numElements}, cast<VectorType>(fp8Value.getType()).getElementType());

  Value input1d =
      rewriter.create<vector::ShapeCastOp>(loc, v1dInputType, fp8Value);
  Value result1d = createZeroVector(loc, v1dType, i8Type, f32Type, rewriter);

  Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = rewriter.create<arith::ConstantIndexOp>(loc, numElements);
  Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto forOp = rewriter.create<scf::ForOp>(
      loc, lb, ub, step, ValueRange{result1d},
      [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
        Value scalar = b.create<vector::ExtractElementOp>(l, input1d, iv);
        Value converted =
            createFp8ToFpScalar(l, scalar, helper, i8Type, f32Type,
                                v1dType.getElementType(), rewriter);
        Value updated =
            b.create<vector::InsertElementOp>(l, converted, iterArgs[0], iv);
        b.create<scf::YieldOp>(l, updated);
      });

  return rewriter.create<vector::ShapeCastOp>(loc, vectorType,
                                              forOp.getResult(0));
}

class ExpandFloat8ConversionsPass
    : public ExpandFloat8ConversionsBase<ExpandFloat8ConversionsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    auto i8Type = IntegerType::get(context, 8);
    auto f32Type = Float32Type::get(context);

    SmallVector<arith::ExtFOp> fp8ExtOps;
    SmallVector<arith::TruncFOp> fp8TruncOps;
    module.walk([&](arith::ExtFOp op) {
      if (isScalarOrFixedVectorOf(op.getOperand().getType(),
                                  Float8E4M3FNType::get(context)) &&
          isScalarOrFixedVectorOfFloat(op.getType()) &&
          hasSameFixedVectorShape(op.getOperand().getType(), op.getType()))
        fp8ExtOps.push_back(op);
    });
    module.walk([&](arith::TruncFOp op) {
      if (isScalarOrFixedVectorOfFloat(op.getOperand().getType()) &&
          isScalarOrFixedVectorOf(op.getType(),
                                  Float8E4M3FNType::get(context)) &&
          hasSameFixedVectorShape(op.getOperand().getType(), op.getType()))
        fp8TruncOps.push_back(op);
    });

    if (fp8ExtOps.empty() && fp8TruncOps.empty())
      return;

    OpBuilder builder(context);
    auto fp8ToF32 =
        getOrCreateHelper(module, kFp8ToF32Name, {i8Type}, {f32Type}, builder);
    auto f32ToFp8 =
        getOrCreateHelper(module, kF32ToFp8Name, {f32Type}, {i8Type}, builder);

    IRRewriter rewriter(context);
    for (arith::ExtFOp op : fp8ExtOps) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      rewriter.replaceOp(op,
                         expandFp8ToFp(loc, op.getOperand(), op.getType(),
                                       fp8ToF32, i8Type, f32Type, rewriter));
    }

    for (arith::TruncFOp op : fp8TruncOps) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      rewriter.replaceOp(op,
                         expandFpToFp8(loc, op.getOperand(), op.getType(),
                                       f32ToFp8, i8Type, f32Type, rewriter));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createExpandFloat8ConversionsPass() {
  return std::make_unique<ExpandFloat8ConversionsPass>();
}
