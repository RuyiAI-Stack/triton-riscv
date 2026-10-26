//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "ContractTileSpecialization.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "triton-shared/Analysis/KernelContract.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;
using namespace triton;

#define GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
#define GEN_PASS_DEF_TRITONTOLINALGEXPERIMENTAL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

/// Scalar boundary events use the same logical snapshot ABI as tensor events.
/// Lifting to one lane preserves the address, value, mask and origin, while
/// avoiding an unstructured scalar memory operation in the neutral contract.
static void liftScalarBoundaryAccesses(ModuleOp module) {
  SmallVector<Operation *> accesses;
  module.walk([&](Operation *op) {
    if ((isa<triton::LoadOp>(op) &&
         !isa<ShapedType>(op->getResult(0).getType())) ||
        (isa<triton::StoreOp>(op) &&
         !isa<ShapedType>(cast<triton::StoreOp>(op).getValue().getType())))
      accesses.push_back(op);
  });
  for (Operation *access : accesses) {
    OpBuilder builder(access);
    IRMapping mapping;
    for (Value operand : access->getOperands()) {
      auto type = RankedTensorType::get({1}, operand.getType());
      Value lane =
          triton::SplatOp::create(builder, access->getLoc(), type, operand);
      if (isa<triton::PointerType>(operand.getType())) {
        // A one-element range denotes a genuine unit-stride view. A bare
        // pointer splat would carry stride zero into rectangular access
        // analysis even though its only address is identical.
        Value index = triton::MakeRangeOp::create(
            builder, access->getLoc(),
            RankedTensorType::get({1}, builder.getI32Type()), 0, 1);
        lane = triton::AddPtrOp::create(builder, access->getLoc(), type, lane,
                                        index);
      }
      mapping.map(operand, lane);
    }
    Operation *lifted = builder.clone(*access, mapping);
    if (access->getNumResults()) {
      lifted->getResult(0).setType(
          RankedTensorType::get({1}, access->getResult(0).getType()));
      Value zero = arith::ConstantIndexOp::create(builder, access->getLoc(), 0);
      Value scalar = tensor::ExtractOp::create(
          builder, access->getLoc(), lifted->getResult(0), ValueRange{zero});
      access->getResult(0).replaceAllUsesWith(scalar);
    }
    access->erase();
  }
}

struct FusePointwiseTensorExpression : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool laneExpression(linalg::GenericOp generic) {
    return generic && generic.hasPureTensorSemantics() &&
           generic.getNumLoops() == 1 &&
           generic.getIteratorTypesArray()[0] == utils::IteratorType::parallel;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    if (!laneExpression(generic))
      return failure();
    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!laneExpression(producer) ||
          producer->getBlock() != generic->getBlock() ||
          !linalg::areElementwiseOpsFusable(operand))
        continue;
      auto fused = linalg::fuseElementwiseOps(rewriter, operand);
      if (failed(fused))
        return failure();
      for (auto [original, replacement] : fused->replacements)
        rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
          return use.get().getDefiningOp() != producer;
        });
      rewriter.eraseOp(generic);
      return success();
    }
    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto fill = operand->get().getDefiningOp<linalg::FillOp>();
      if (!fill || !generic.payloadUsesValueFromOperand(operand))
        continue;
      rewriter.modifyOpInPlace(generic, [&] {
        generic.getRegion()
            .front()
            .getArgument(operand->getOperandNumber())
            .replaceAllUsesWith(fill.getInputs()[0]);
      });
      return success();
    }
    return failure();
  }
};

static LogicalResult fusePointwiseTensorExpressions(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  patterns.add<FusePointwiseTensorExpression>(module.getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  // Do not run RemoveOutsDependency: changing a loop's carried tensor into
  // an unrelated empty buffer destroys its content boundary.
  return applyPatternsGreedily(module, std::move(patterns));
}

static std::optional<int64_t> getSplatInteger(Value value) {
  Attribute valueAttr;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = constant.getValue();
  } else if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    if (fill.getInputs().size() != 1)
      return std::nullopt;
    auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    valueAttr = constant.getValue();
  } else {
    return std::nullopt;
  }
  if (auto integer = dyn_cast<IntegerAttr>(valueAttr))
    return integer.getInt();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!dense || !dense.isSplat() || !isa<IntegerType>(dense.getElementType()))
    return std::nullopt;
  return dense.getSplatValue<APInt>().getSExtValue();
}

static bool isSplatZero(Value value) {
  Attribute valueAttr;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    valueAttr = constant.getValue();
  } else if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    if (fill.getInputs().size() != 1)
      return false;
    auto constant = fill.getInputs().front().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return false;
    valueAttr = constant.getValue();
  } else {
    return false;
  }
  if (auto integer = dyn_cast<IntegerAttr>(valueAttr))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(valueAttr))
    return floating.getValue().isZero();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (auto integers = dyn_cast<DenseIntElementsAttr>(dense))
    return integers.isSplat() && integers.getSplatValue<APInt>().isZero();
  if (auto floats = dyn_cast<DenseFPElementsAttr>(dense))
    return floats.isSplat() && floats.getSplatValue<APFloat>().isZero();
  return false;
}

static bool hasIdentityParallelMaps(linalg::GenericOp op) {
  auto maps = op.getIndexingMapsArray();
  return op.getIteratorTypesArray().size() == 1 &&
         op.getIteratorTypesArray()[0] == utils::IteratorType::parallel &&
         maps.size() == op->getNumOperands() &&
         llvm::all_of(maps, [](AffineMap map) { return map.isIdentity(); });
}

static bool hasIdentityParallelMaps(linalg::GenericOp op, unsigned rank) {
  auto maps = op.getIndexingMapsArray();
  auto iterators = op.getIteratorTypesArray();
  return iterators.size() == rank && maps.size() == op->getNumOperands() &&
         llvm::all_of(iterators,
                      [](utils::IteratorType iterator) {
                        return iterator == utils::IteratorType::parallel;
                      }) &&
         llvm::all_of(maps, [rank](AffineMap map) {
           return map.isIdentity() && map.getNumDims() == rank &&
                  map.getNumResults() == rank;
         });
}

static bool matchTensorI32Add(linalg::GenericOp add, Value product,
                              Value &accumulator) {
  if (!add.hasPureTensorSemantics() || add.getInputs().size() != 2 ||
      add.getOutputs().size() != 1 || add.getResults().size() != 1 ||
      !hasIdentityParallelMaps(add, 2))
    return false;
  if (add.getInputs()[0] == product)
    accumulator = add.getInputs()[1];
  else if (add.getInputs()[1] == product)
    accumulator = add.getInputs()[0];
  else
    return false;
  auto productType = dyn_cast<RankedTensorType>(product.getType());
  auto accumulatorType = dyn_cast<RankedTensorType>(accumulator.getType());
  if (!productType || !accumulatorType || productType.getRank() != 2 ||
      accumulatorType != productType ||
      add.getResult(0).getType() != productType ||
      !productType.getElementType().isInteger(32))
    return false;
  Block &body = add.getRegion().front();
  if (body.getNumArguments() != 3 || body.empty() ||
      !isa<linalg::YieldOp>(body.back()) ||
      std::distance(body.begin(), body.end()) != 2)
    return false;
  auto addOp = dyn_cast<arith::AddIOp>(&body.front());
  auto yield = cast<linalg::YieldOp>(body.back());
  if (!addOp || addOp.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != addOp.getResult())
    return false;
  bool direct = addOp.getLhs() == body.getArgument(0) &&
                addOp.getRhs() == body.getArgument(1);
  bool swapped = addOp.getLhs() == body.getArgument(1) &&
                 addOp.getRhs() == body.getArgument(0);
  return direct || swapped;
}

static bool matchTensorI32Add(arith::AddIOp add, Value product,
                              Value &accumulator) {
  if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      (add.getLhs() != product && add.getRhs() != product))
    return false;
  accumulator = add.getLhs() == product ? add.getRhs() : add.getLhs();
  auto productType = dyn_cast<RankedTensorType>(product.getType());
  auto accumulatorType = dyn_cast<RankedTensorType>(accumulator.getType());
  return productType && accumulatorType && productType.getRank() == 2 &&
         accumulatorType == productType &&
         productType.getElementType().isInteger(32);
}

/// Fuse the tensor spelling of `P = matmul(A, B, zero); C = C + P` before
/// structured-to-memref and bufferization.  The proof is intentionally local:
/// both operations stay in one block, P has exactly one consumer, and no
/// effectful operation is crossed.  This preserves branch/loop boundaries and
/// leaves ambiguous or non-private products for the ordinary lowering.
static LogicalResult fuseIntegerTensorAccumulators(ModuleOp module) {
  SmallVector<linalg::MatmulOp> candidates;
  module.walk([&](linalg::MatmulOp matmul) {
    if (matmul.hasPureTensorSemantics() && matmul.getResults().size() == 1)
      candidates.push_back(matmul);
  });

  for (linalg::MatmulOp matmul : candidates) {
    if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1)
      continue;
    auto outputType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!outputType || outputType.getRank() != 2 ||
        !outputType.getElementType().isInteger(32) ||
        !cast<RankedTensorType>(matmul.getInputs()[0].getType())
             .getElementType()
             .isInteger(8) ||
        !cast<RankedTensorType>(matmul.getInputs()[1].getType())
             .getElementType()
             .isInteger(8) ||
        matmul.getCast() != linalg::TypeFn::cast_signed)
      continue;
    Value product = matmul.getResult(0);
    if (!product.hasOneUse())
      continue;

    Operation *consumer = *product.getUsers().begin();
    Value accumulator;
    linalg::GenericOp genericAdd;
    arith::AddIOp scalarAdd;
    if (auto candidate = dyn_cast<linalg::GenericOp>(consumer)) {
      if (!matchTensorI32Add(candidate, product, accumulator))
        continue;
      genericAdd = candidate;
    } else if (auto candidate = dyn_cast<arith::AddIOp>(consumer)) {
      if (!matchTensorI32Add(candidate, product, accumulator))
        continue;
      scalarAdd = candidate;
    } else {
      continue;
    }

    if (!accumulator || accumulator == product ||
        accumulator.getType() != product.getType() ||
        matmul->getBlock() != consumer->getBlock() ||
        !matmul->isBeforeInBlock(consumer))
      continue;
    Value zeroInit = matmul.getOutputs().front();
    if (!isSplatZero(zeroInit))
      continue;
    auto zeroFill = zeroInit.getDefiningOp<linalg::FillOp>();
    // Save the producer before erasing the fill: zeroInit is then invalid.
    auto empty =
        zeroFill
            ? zeroFill.getOutputs().front().getDefiningOp<tensor::EmptyOp>()
            : tensor::EmptyOp();

    bool harmless = true;
    for (Operation *between = matmul->getNextNode(); between != consumer;
         between = between ? between->getNextNode() : nullptr) {
      if (!between || !isMemoryEffectFree(between)) {
        harmless = false;
        break;
      }
    }
    if (!harmless)
      continue;

    // The matmul already has the right result type.  Retarget its init to C,
    // and keep it at the add's position so a locally defined C dominates it.
    // Other uses of the old C tensor remain SSA uses; bufferization must
    // preserve them when deciding whether its storage can be updated in place.
    matmul->moveBefore(consumer);
    matmul.getOutputsMutable()[0].set(accumulator);
    // The dot's old init was zero. Its accumulator source metadata must be
    // rebuilt from C by graph annotation, together with task/value IDs.
    for (StringRef attr : {"contract.accumulator_origin_id",
                           "contract.accumulator_source_load_origin_ids",
                           "contract.accumulator_source_lineage_complete"})
      matmul->removeAttr(attr);
    if (genericAdd)
      genericAdd.getResult(0).replaceAllUsesWith(matmul.getResult(0));
    else
      scalarAdd.getResult().replaceAllUsesWith(matmul.getResult(0));
    consumer->erase();
    if (zeroFill && zeroFill->use_empty())
      zeroFill.erase();
    if (empty && empty->use_empty())
      empty.erase();
  }
  return success();
}

/// A zero tensor used by both a matmul init and a following floating add
/// remains live across the matmul. One-Shot must then copy the init before
/// updating it, even though the add only needs its scalar splat value.
/// Materialize that scalar use explicitly and destination-pass the product.
/// This changes storage, not arithmetic: the add, its operand order, exact
/// signed-zero constant and fastmath attributes are cloned without alteration.
static void normalizeFloatingZeroProductAdds(ModuleOp module) {
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp generic) {
    if (generic.hasPureTensorSemantics())
      candidates.push_back(generic);
  });
  for (linalg::GenericOp generic : candidates) {
    if (generic.getInputs().size() != 2 || generic.getOutputs().size() != 1 ||
        generic.getResults().size() != 1 ||
        !hasIdentityParallelMaps(generic, 2))
      continue;
    auto type = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!type || !isa<FloatType>(type.getElementType()) ||
        generic.getInputs()[0].getType() != type ||
        generic.getInputs()[1].getType() != type)
      continue;
    Block &body = generic.getRegion().front();
    if (body.getNumArguments() != 3 || body.getOperations().size() != 2)
      continue;
    auto add = dyn_cast<arith::AddFOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.back());
    if (!add || !yield || yield.getValues().size() != 1 ||
        yield.getValues()[0] != add.getResult() ||
        !((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getLhs() == body.getArgument(1) &&
           add.getRhs() == body.getArgument(0))))
      continue;
    for (unsigned zeroIndex : {0u, 1u}) {
      Value zeroTensor = generic.getInputs()[zeroIndex];
      Value product = generic.getInputs()[1 - zeroIndex];
      auto matmul = product.getDefiningOp<linalg::MatmulOp>();
      if (!matmul || !matmul.hasPureTensorSemantics() ||
          matmul.getOutputs().size() != 1 ||
          matmul.getOutputs()[0].getType() != type ||
          !isSplatZero(matmul.getOutputs()[0]) ||
          matmul->getBlock() != generic->getBlock() ||
          !matmul->isBeforeInBlock(generic) || !isSplatZero(zeroTensor))
        continue;
      // Keep the proof local. Other product consumers and nonzero C retain
      // their independent SSA snapshots and the ordinary copy decision.
      if (!product.hasOneUse())
        continue;
      OpBuilder builder(generic);
      Value scalar;
      if (auto fill = zeroTensor.getDefiningOp<linalg::FillOp>()) {
        scalar = fill.getInputs()[0];
      } else if (auto constant =
                     zeroTensor.getDefiningOp<arith::ConstantOp>()) {
        auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
        if (!dense || !dense.isSplat())
          continue;
        scalar = builder.create<arith::ConstantOp>(
            generic.getLoc(), type.getElementType(),
            dense.getSplatValue<FloatAttr>());
      }
      if (!scalar || scalar.getType() != type.getElementType())
        continue;
      auto replacement = builder.create<linalg::GenericOp>(
          generic.getLoc(), TypeRange{type}, ValueRange{product},
          ValueRange{product},
          SmallVector<AffineMap>(2, builder.getMultiDimIdentityMap(2)),
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &nested, Location loc, ValueRange arguments) {
            IRMapping mapping;
            mapping.map(body.getArgument(zeroIndex), scalar);
            mapping.map(body.getArgument(1 - zeroIndex), arguments[0]);
            Operation *cloned = nested.clone(*add, mapping);
            nested.create<linalg::YieldOp>(loc, cloned->getResult(0));
          });
      for (NamedAttribute attr : generic->getAttrs())
        if (attr.getName().getValue().starts_with("contract."))
          replacement->setAttr(attr.getName(), attr.getValue());
      generic.getResult(0).replaceAllUsesWith(replacement.getResult(0));
      generic.erase();
      break;
    }
  }
}

static linalg::GenericOp getZeroBasedRange(Value value) {
  auto range = value.getDefiningOp<linalg::GenericOp>();
  if (!range || !range.getInputs().empty() || range.getOutputs().size() != 1 ||
      range.getResults().size() != 1 || range.getResult(0) != value ||
      !range.hasPureTensorSemantics() || !hasIdentityParallelMaps(range))
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0) ||
      type.getDimSize(0) <= 0)
    return {};
  Block &body = range.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 3)
    return {};
  auto indexOp = dyn_cast<linalg::IndexOp>(body.front());
  auto cast = dyn_cast<arith::IndexCastOp>(body.front().getNextNode());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!indexOp || indexOp.getDim() != 0 || !cast ||
      cast.getIn() != indexOp.getResult() || !cast.getType().isInteger(32) ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues()[0] != cast.getResult())
    return {};
  return range;
}

static std::optional<int64_t> matchStaticPrefixPredicate(Value predicate,
                                                         Operation **rangeOp,
                                                         Operation **boundOp) {
  auto compare = predicate.getDefiningOp<linalg::GenericOp>();
  if (!compare || compare.getInputs().size() != 2 ||
      compare.getOutputs().size() != 1 || compare.getResults().size() != 1 ||
      compare.getResult(0) != predicate || !compare.hasPureTensorSemantics() ||
      !hasIdentityParallelMaps(compare))
    return std::nullopt;
  Block &body = compare.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return std::nullopt;
  auto cmp = dyn_cast<arith::CmpIOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != body.getArgument(0) ||
      cmp.getRhs() != body.getArgument(1) || !yield ||
      yield.getValues().size() != 1 || yield.getValues()[0] != cmp.getResult())
    return std::nullopt;
  linalg::GenericOp range = getZeroBasedRange(compare.getInputs()[0]);
  auto predicateType = dyn_cast<RankedTensorType>(predicate.getType());
  auto rangeType =
      range ? dyn_cast<RankedTensorType>(range.getResult(0).getType())
            : RankedTensorType();
  if (!range || !predicateType || !rangeType ||
      predicateType.getShape() != rangeType.getShape())
    return std::nullopt;
  std::optional<int64_t> prefix = getSplatInteger(compare.getInputs()[1]);
  if (!prefix || *prefix <= 0 || *prefix >= rangeType.getDimSize(0))
    return std::nullopt;
  *rangeOp = range;
  if (auto fill = compare.getInputs()[1].getDefiningOp<linalg::FillOp>())
    *boundOp = fill;
  else
    *boundOp = nullptr;
  return prefix;
}

static TypedAttr getSplatScalarConstant(Value value) {
  if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
    if (fill.getInputs().size() != 1)
      return {};
    value = fill.getInputs()[0];
  }
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return {};
  Attribute attr = constant.getValue();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    return dense.isSplat()
               ? dyn_cast<TypedAttr>(dense.getSplatValue<Attribute>())
               : TypedAttr();
  auto scalar = dyn_cast<TypedAttr>(attr);
  return scalar && scalar.getType().isIntOrFloat() ? scalar : TypedAttr();
}

static void normalizeStaticTailSelects(ModuleOp module) {
  module.walk([&](linalg::GenericOp selectGeneric) {
    if (selectGeneric.getInputs().empty() ||
        selectGeneric.getOutputs().size() != 1 ||
        selectGeneric.getResults().size() != 1 ||
        !selectGeneric.hasPureTensorSemantics() ||
        !hasIdentityParallelMaps(selectGeneric))
      return;
    auto outputType =
        dyn_cast<RankedTensorType>(selectGeneric.getResult(0).getType());
    if (!outputType || outputType.getRank() != 1 || outputType.isDynamicDim(0))
      return;
    Block &body = selectGeneric.getRegion().front();
    if (std::distance(body.begin(), body.end()) != 2)
      return;
    auto select = dyn_cast<arith::SelectOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!select || !yield || yield.getValues().size() != 1 ||
        yield.getValues()[0] != select.getResult())
      return;
    auto condition = dyn_cast<BlockArgument>(select.getCondition());
    auto trueValue = dyn_cast<BlockArgument>(select.getTrueValue());
    auto falseValue = dyn_cast<BlockArgument>(select.getFalseValue());
    if (!condition || !trueValue || !falseValue ||
        condition.getArgNumber() >= selectGeneric.getInputs().size() ||
        trueValue.getArgNumber() >= selectGeneric.getInputs().size() ||
        falseValue.getArgNumber() >= selectGeneric.getInputs().size())
      return;
    Value falseInput = selectGeneric.getInputs()[falseValue.getArgNumber()];
    TypedAttr scalar = getSplatScalarConstant(falseInput);
    if (!scalar || scalar.getType() != outputType.getElementType())
      return;
    Operation *range = nullptr;
    Operation *bound = nullptr;
    Value predicate = selectGeneric.getInputs()[condition.getArgNumber()];
    std::optional<int64_t> prefix =
        matchStaticPrefixPredicate(predicate, &range, &bound);
    if (!prefix || outputType.getDimSize(0) <= *prefix ||
        !predicate.hasOneUse() || !range->getResult(0).hasOneUse() ||
        selectGeneric.getInputs()[trueValue.getArgNumber()].getType() !=
            outputType ||
        selectGeneric.getInputs()[falseValue.getArgNumber()].getType() !=
            outputType)
      return;

    // Preserve the selected false constant exactly, including the sign of
    // floating zero.  Multiplication by a predicate is incorrect for NaN/Inf.
    OpBuilder rewriter(selectGeneric);
    Value zero = rewriter.create<arith::ConstantOp>(
        selectGeneric.getLoc(), outputType.getElementType(), scalar);
    int64_t tailSize = outputType.getDimSize(0) - *prefix;
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(*prefix)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(tailSize)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    Value original = selectGeneric.getInputs()[trueValue.getArgNumber()];
    Value tail = rewriter.create<tensor::ExtractSliceOp>(
        selectGeneric.getLoc(), original, offsets, sizes, strides);
    Value zeros =
        rewriter
            .create<linalg::FillOp>(selectGeneric.getLoc(), ValueRange{zero},
                                    ValueRange{tail})
            .getResult(0);
    auto inserted = rewriter.create<tensor::InsertSliceOp>(
        selectGeneric.getLoc(), zeros, original, offsets, sizes, strides);
    selectGeneric.getResult(0).replaceAllUsesWith(inserted.getResult());
    selectGeneric.erase();
  });
}

struct LaunchGridContract {
  Operation *function;
  DenseI32ArrayAttr grid;
};

static FailureOr<LaunchGridContract>
parseLaunchGridContract(ModuleOp module, StringRef specification) {
  size_t colon = specification.find(':');
  if (colon == StringRef::npos ||
      specification.find(':', colon + 1) != StringRef::npos)
    return failure();

  StringRef symbolName = specification.take_front(colon).trim();
  if (symbolName.starts_with("@"))
    symbolName = symbolName.drop_front();
  if (symbolName.empty() || symbolName.contains(' ') ||
      symbolName.contains('\t'))
    return failure();

  SmallVector<StringRef, 4> dimensions;
  specification.drop_front(colon + 1).split(dimensions, ',', -1, true);
  if (dimensions.size() != 3)
    return failure();

  SmallVector<int32_t, 3> extents;
  for (StringRef dimension : dimensions) {
    int64_t extent = 0;
    dimension = dimension.trim();
    if (dimension.empty() || dimension.getAsInteger(10, extent) ||
        extent <= 0 || extent > std::numeric_limits<int32_t>::max())
      return failure();
    extents.push_back(static_cast<int32_t>(extent));
  }

  Operation *function = SymbolTable::lookupSymbolIn(module, symbolName);
  if (!function || function->getName().getStringRef() != "tt.func")
    return failure();

  return LaunchGridContract{
      function, DenseI32ArrayAttr::get(module.getContext(), extents)};
}

static void finalizeSourceContract(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](Operation *op) {
    NamedAttrList provenance;
    if (auto existing =
            op->getAttrOfType<DictionaryAttr>("contract.provenance"))
      provenance.assign(existing.getValue());
    SmallVector<StringAttr> erase;
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef name = attr.getName().getValue();
      if (!name.starts_with("contract."))
        continue;
      StringRef field = name.drop_front(StringRef("contract.").size());
      if (field == "origin_id" || field.ends_with("_origin_id") ||
          field.ends_with("_source_load_origin_ids") ||
          field.ends_with("_source_lineage_complete")) {
        provenance.set(field, attr.getValue());
        erase.push_back(attr.getName());
      } else if (field == "event_kind") {
        erase.push_back(attr.getName());
      }
    }
    for (StringAttr name : erase)
      op->removeAttr(name);
    if (!provenance.empty())
      op->setAttr("contract.provenance",
                  provenance.getDictionary(module.getContext()));
    if (auto access = op->getAttrOfType<DictionaryAttr>("contract.access")) {
      NamedAttrList facts(access);
      facts.erase("logical_shape"); // temporary input to static specialization
      op->setAttr("contract.access", facts.getDictionary(module.getContext()));
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      if (auto grid =
              function->getAttrOfType<DenseI32ArrayAttr>("gpu.known_grid_size"))
        function->setAttr("contract.grid", grid);
      if (!function.isExternal() && function.isPublic())
        function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    }
  });
}

/// Canonicalize bufferizer bookkeeping before building content versions.
/// Boundary events remain explicit. A rank-0 initialization copy can be
/// rematerialized only from a locally proven reaching fill; its scalar value
/// is copied verbatim, preserving floating-point initialization bits/order.
static void normalizeContractBuffers(ModuleOp module) {
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    if (copy->hasAttr("contract.access"))
      continue;
    if (copy.getSource() == copy.getTarget()) {
      copy.erase();
      continue;
    }
    auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!sourceType || !targetType || sourceType.getRank() != 0 ||
        targetType.getRank() != 0 ||
        sourceType.getElementType() != targetType.getElementType() ||
        !copy.getSource().getDefiningOp<memref::AllocOp>() ||
        !copy.getTarget().getDefiningOp<memref::AllocOp>())
      continue;
    linalg::FillOp initializer;
    for (Operation *previous = copy->getPrevNode(); previous;
         previous = previous->getPrevNode()) {
      if (auto fill = dyn_cast<linalg::FillOp>(previous)) {
        if (fill.getOutputs().size() == 1 &&
            fill.getOutputs()[0] == copy.getSource())
          initializer = fill;
        break;
      }
      if (!isa<memref::AllocOp, memref::AllocaOp>(previous) &&
          !isMemoryEffectFree(previous))
        break;
    }
    if (!initializer)
      continue;
    OpBuilder builder(copy);
    auto fill = builder.create<linalg::FillOp>(
        copy.getLoc(), initializer.getInputs(), ValueRange{copy.getTarget()});
    for (NamedAttribute attr : copy->getAttrs())
      fill->setAttr(attr.getName(), attr.getValue());
    copy.erase();
  }
}

class TritonToLinalgExperimentalPass
    : public ::impl::TritonToLinalgExperimentalBase<
          TritonToLinalgExperimentalPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    vector::VectorDialect, tts::TritonStructuredDialect,
                    tptr::TPtrDialect, ptr::PtrDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    TritonLoweringConfig config{emitContract};
    if (emitContract &&
        failed(mlir::triton::contract::checkKernelContractVersions(module)))
      return signalPassFailure();
    if ((!launchGrid.empty() || !callerContract.empty() || dumpFacts) &&
        !emitContract) {
      module.emitError(
          "launch-grid, caller-contract and dump-facts require contract=true");
      return signalPassFailure();
    }
    if (launchGrid.getNumOccurrences() && launchGrid.empty()) {
      module.emitError("invalid empty launch-grid option");
      return signalPassFailure();
    }
    DenseSet<Operation *> assigned;
    if (!launchGrid.empty()) {
      SmallVector<StringRef> entries;
      StringRef(launchGrid).split(entries, ';', -1, true);
      for (StringRef entry : entries) {
        auto facts = parseLaunchGridContract(module, entry);
        if (failed(facts) || !assigned.insert(facts->function).second) {
          module.emitError()
              << "invalid or duplicate launch-grid entry '" << entry
              << "'; expected symbol:x,y,z with positive int32 extents";
          return signalPassFailure();
        }
        auto previous = facts->function->getAttr("gpu.known_grid_size");
        if (previous && previous != facts->grid) {
          module.emitError(
              "launch-grid conflicts with existing gpu.known_grid_size");
          return signalPassFailure();
        }
        facts->function->setAttr("gpu.known_grid_size", facts->grid);
      }
    }
    if (emitContract) {
      Builder builder(&getContext());
      if (!callerContract.empty()) {
        if (callerContract != "args_aligned64_nonalias") {
          module.emitError() << "invalid caller-contract: " << callerContract;
          return signalPassFailure();
        }
        DictionaryAttr abi = builder.getDictionaryAttr(
            {builder.getNamedAttr("args_aligned_bytes",
                                  builder.getI64IntegerAttr(64)),
             builder.getNamedAttr("args_disjoint", builder.getBoolAttr(true))});
        if (auto existing = module->getAttr("contract.abi");
            existing && existing != abi) {
          module.emitError("caller-contract conflicts with contract.abi");
          return signalPassFailure();
        }
        module->setAttr("contract.abi", abi);
      }
      int64_t origin = 0;
      module.walk([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        StringRef kind = name == "tt.load"    ? "load"
                         : name == "tt.store" ? "store"
                         : name == "tt.dot"   ? "dot"
                                              : "";
        if (!kind.empty()) {
          op->setAttr("contract.origin_id",
                      builder.getI64IntegerAttr(origin++));
          op->setAttr("contract.event_kind", builder.getStringAttr(kind));
        }
      });
    }

    bool tensorFirst = structuredLdstMode == "tensor-first-vector-cpu";
    if (!tensorFirst && structuredLdstMode != "legacy_memref_bridge") {
      module.emitError("invalid structured-ldst-mode; expected "
                       "legacy_memref_bridge or tensor-first-vector-cpu");
      return signalPassFailure();
    }
    if (emitContract)
      liftScalarBoundaryAccesses(module);
    PassManager pm(&getContext(), module.getOperationName());
    pm.addPass(createTritonToStructuredPass(enableMakeGatherScatterTensorPtr));
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createTritonToUnstructuredPass());
    pm.addPass(createTritonArithToLinalgPass(true, false, config));
    pm.addPass(
        createStructuredToMemrefPass(tensorFirst && !emitContract, config));
    pm.addPass(createUnstructuredToMemrefPass(config));
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());
    if (emitContract)
      pm.addNestedPass<func::FuncOp>(createRemoveDeadValuesPass());
    else
      pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape)
      pm.addPass(createCollapseShapePass());
    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
    if (!emitContract)
      return;

    if (failed(specializeContractTiles(module)) ||
        failed(fuseIntegerTensorAccumulators(module)))
      return signalPassFailure();
    normalizeFloatingZeroProductAdds(module);
    normalizeStaticTailSelects(module);
    if (failed(fusePointwiseTensorExpressions(module)))
      return signalPassFailure();
    OpPassManager cleanup(ModuleOp::getOperationName());
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    if (failed(runPipeline(cleanup, module)))
      return signalPassFailure();

    // One-Shot's SSA read-after-write analysis preserves all old tensor
    // readers (notably centered while squares is computed). Contract metadata
    // is generated only after this safety decision, never used to force
    // aliasing.
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowReturnAllocsFromLoops = true;
    if (auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi"))
      if (auto alignment = abi.getAs<IntegerAttr>("args_aligned_bytes"))
        options.bufferAlignment = alignment.getInt();
    bufferization::BufferizationState bufferizationState;
    if (failed(bufferization::runOneShotBufferize(module, options,
                                                  bufferizationState)))
      return signalPassFailure();
    OpPassManager finish(ModuleOp::getOperationName());
    finish.addPass(createCanonicalizerPass());
    finish.addPass(createCSEPass());
    if (failed(runPipeline(finish, module)))
      return signalPassFailure();
    normalizeContractBuffers(module);
    if (failed(runPipeline(finish, module)))
      return signalPassFailure();
    finalizeSourceContract(module);
    if (failed(mlir::triton::contract::annotateKernelContract(module,
                                                              "triton-riscv")))
      return signalPassFailure();
    if (failed(mlir::triton::contract::verifyKernelContract(module)))
      return signalPassFailure();
    if (dumpFacts)
      module.print(llvm::errs());
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
