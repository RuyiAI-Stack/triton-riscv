//===- ContractTileSpecialization.cpp - Rectangular tiles
//-------------------===//

#include "ContractTileSpecialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>

using namespace mlir;

namespace {
struct Interval {
  int64_t lo, hi;
};

/// All supported operations must have mathematical signed-integer semantics
/// throughout the launch/loop domain. Index proofs fit both 32 and 64 bits;
/// they do not assume a target index width at this frontend stage.
static bool fits(Type type, Interval range) {
  unsigned width = type.isIndex() ? 32 : type.getIntOrFloatBitWidth();
  return width <= 64 && llvm::isIntN(width, range.lo) &&
         llvm::isIntN(width, range.hi);
}

class Ranges {
public:
  DenseMap<Value, Interval> arguments;

  /// A launch-domain comparison can guard an otherwise static memory extent
  /// (for example `row < ROWS` in a one-program-per-row bias add). Preserve
  /// unknown predicates rather than treating all masked accesses as full.
  std::optional<bool> truth(Value value) const {
    if (auto constant = getConstantIntValue(value))
      return *constant != 0;
    auto cmp = value.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return std::nullopt;
    auto a = get(cmp.getLhs()), b = get(cmp.getRhs());
    if (!a || !b)
      return std::nullopt;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      if (a->lo == a->hi && b->lo == b->hi && a->lo == b->lo)
        return cmp.getPredicate() == arith::CmpIPredicate::eq;
      if (a->hi < b->lo || b->hi < a->lo)
        return cmp.getPredicate() == arith::CmpIPredicate::ne;
      break;
    case arith::CmpIPredicate::slt:
      if (a->hi < b->lo)
        return true;
      if (a->lo >= b->hi)
        return false;
      break;
    case arith::CmpIPredicate::sle:
      if (a->hi <= b->lo)
        return true;
      if (a->lo > b->hi)
        return false;
      break;
    case arith::CmpIPredicate::sgt:
      if (a->lo > b->hi)
        return true;
      if (a->hi <= b->lo)
        return false;
      break;
    case arith::CmpIPredicate::sge:
      if (a->lo >= b->hi)
        return true;
      if (a->hi < b->lo)
        return false;
      break;
    default:
      break;
    }
    return std::nullopt;
  }

  std::optional<Interval> get(Value value, unsigned depth = 0) const {
    if (depth > 64 || !value.getType().isIntOrIndex())
      return std::nullopt;
    if (auto constant = getConstantIntValue(value)) {
      Interval result{*constant, *constant};
      return fits(value.getType(), result) ? std::optional<Interval>(result)
                                           : std::nullopt;
    }
    if (auto found = arguments.find(value); found != arguments.end())
      return found->second;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto loop =
          dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || loop.getInductionVar() != value || loop.getUnsignedCmp())
        return std::nullopt;
      auto lo = getConstantIntValue(loop.getLowerBound());
      auto hi = getConstantIntValue(loop.getUpperBound());
      auto step = getConstantIntValue(loop.getStep());
      int64_t distance, next;
      if (!lo || !hi || !step || *step <= 0 || *hi <= *lo ||
          llvm::SubOverflow(*hi, *lo, distance) ||
          llvm::AddOverflow(*lo, ((distance - 1) / *step) * *step, next))
        return std::nullopt;
      Interval result{*lo, next};
      // Include the final induction increment, which also must not wrap.
      if (llvm::AddOverflow(next, *step, next) ||
          !fits(value.getType(), {*lo, next}))
        return std::nullopt;
      return result;
    }
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<arith::IndexCastOp, arith::ExtSIOp>(op)) {
      auto input = get(op->getOperand(0), depth + 1);
      if (input && fits(value.getType(), *input))
        return input;
      return std::nullopt;
    }
    if (isa<arith::DivSIOp, arith::RemSIOp>(op)) {
      auto numerator = get(op->getOperand(0), depth + 1);
      auto divisor = getConstantIntValue(op->getOperand(1));
      if (!numerator || numerator->lo < 0 || !divisor || *divisor <= 0)
        return std::nullopt;
      Interval result;
      if (isa<arith::DivSIOp>(op)) {
        result = {numerator->lo / *divisor, numerator->hi / *divisor};
      } else if (numerator->lo / *divisor == numerator->hi / *divisor) {
        result = {numerator->lo % *divisor, numerator->hi % *divisor};
      } else {
        result = {0, std::min(numerator->hi, *divisor - 1)};
      }
      return fits(value.getType(), result) ? std::optional<Interval>(result)
                                           : std::nullopt;
    }
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MinSIOp,
             arith::MaxSIOp>(op))
      return std::nullopt;
    auto a = get(op->getOperand(0), depth + 1);
    auto b = get(op->getOperand(1), depth + 1);
    if (!a || !b)
      return std::nullopt;
    Interval result;
    if (isa<arith::AddIOp>(op)) {
      if (llvm::AddOverflow(a->lo, b->lo, result.lo) ||
          llvm::AddOverflow(a->hi, b->hi, result.hi))
        return std::nullopt;
    } else if (isa<arith::SubIOp>(op)) {
      if (llvm::SubOverflow(a->lo, b->hi, result.lo) ||
          llvm::SubOverflow(a->hi, b->lo, result.hi))
        return std::nullopt;
    } else if (isa<arith::MulIOp>(op)) {
      std::array<int64_t, 4> products;
      if (llvm::MulOverflow(a->lo, b->lo, products[0]) ||
          llvm::MulOverflow(a->lo, b->hi, products[1]) ||
          llvm::MulOverflow(a->hi, b->lo, products[2]) ||
          llvm::MulOverflow(a->hi, b->hi, products[3]))
        return std::nullopt;
      result = {*std::min_element(products.begin(), products.end()),
                *std::max_element(products.begin(), products.end())};
    } else if (isa<arith::MinSIOp>(op)) {
      result = {std::min(a->lo, b->lo), std::min(a->hi, b->hi)};
    } else {
      result = {std::max(a->lo, b->lo), std::max(a->hi, b->hi)};
    }
    if (!fits(value.getType(), result))
      return std::nullopt;
    return result;
  }
};

struct TileExtent {
  Value value, start;
  int64_t tile, bound;
};

template <typename Op> static Value otherOperand(Value value, Value operand) {
  auto op = value.getDefiningOp<Op>();
  if (!op)
    return {};
  if (op.getLhs() == operand)
    return op.getRhs();
  if (op.getRhs() == operand)
    return op.getLhs();
  return {};
}

/// Match the rectangular clamp emitted by structured-to-memref. Canonicalize
/// may remove the redundant outer min(..., tile), so accept both spellings:
/// min(max(min(start + tile, bound), start) - start, tile)
/// and max(min(start + tile, bound), start) - start.
static std::optional<TileExtent> matchExtent(Value value,
                                             const Ranges &ranges) {
  if (!ranges.get(value))
    return std::nullopt;
  SmallVector<std::pair<Value, std::optional<int64_t>>> differences;
  if (auto outer = value.getDefiningOp<arith::MinSIOp>()) {
    for (auto [difference, tileValue] :
         {std::pair{outer.getLhs(), outer.getRhs()},
          std::pair{outer.getRhs(), outer.getLhs()}})
      if (auto tile = getConstantIntValue(tileValue); tile && *tile > 0)
        differences.emplace_back(difference, *tile);
  } else {
    differences.emplace_back(value, std::nullopt);
  }
  for (auto [difference, outerTile] : differences) {
    auto sub = difference.getDefiningOp<arith::SubIOp>();
    if (!sub)
      continue;
    Value start = sub.getRhs();
    Value limited = otherOperand<arith::MaxSIOp>(sub.getLhs(), start);
    auto minimum =
        limited ? limited.getDefiningOp<arith::MinSIOp>() : arith::MinSIOp();
    if (!minimum)
      continue;
    for (auto [end, boundValue] :
         {std::pair{minimum.getLhs(), minimum.getRhs()},
          std::pair{minimum.getRhs(), minimum.getLhs()}}) {
      auto bound = getConstantIntValue(boundValue);
      Value increment = otherOperand<arith::AddIOp>(end, start);
      auto tile = increment ? getConstantIntValue(increment) : std::nullopt;
      if (bound && *bound >= 0 && tile && *tile > 0 &&
          (!outerTile || *outerTile == *tile))
        return TileExtent{value, start, *tile, *bound};
    }
  }
  return std::nullopt;
}

static std::optional<int64_t> constantExtent(const TileExtent &extent,
                                             const Ranges &ranges) {
  auto start = ranges.get(extent.start);
  if (!start || start->lo < 0)
    return std::nullopt;
  auto clamp = [&](int64_t offset) {
    return std::clamp(extent.bound - offset, int64_t(0), extent.tile);
  };
  int64_t a = clamp(start->lo), b = clamp(start->hi);
  return a == b ? std::optional<int64_t>(a) : std::nullopt;
}

static SmallVector<TileExtent> memoryExtents(Operation *root,
                                             const Ranges &ranges) {
  llvm::SetVector<Value> sizes;
  root->walk([&](memref::CopyOp copy) {
    if (!copy->hasAttr("contract.access"))
      return;
    for (Value operand : copy->getOperands())
      if (auto view = operand.getDefiningOp<memref::SubViewOp>())
        for (OpFoldResult size : view.getMixedSizes())
          if (auto value = dyn_cast<Value>(size))
            sizes.insert(value);
  });
  SmallVector<TileExtent> extents;
  for (Value size : sizes)
    if (auto extent = matchExtent(size, ranges))
      extents.push_back(*extent);
  return extents;
}

/// Separate a proven full prefix from one partial last iteration while the
/// accumulator is still tensor SSA. This does not change tile shapes or
/// arithmetic: even the last iteration executes its original padded tile.
/// Memory effects retain their iteration order, including inside branches.
static void splitPartialLastIterations(func::FuncOp function,
                                       const Ranges &ranges) {
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    auto domain = ranges.get(loop.getInductionVar());
    auto step = getConstantIntValue(loop.getStep());
    // Ranges also proves that the original final IV increment cannot wrap.
    // A single iteration already has a singleton domain and needs no split.
    if (!domain || !step || *step <= 0 || domain->lo == domain->hi)
      continue;
    Ranges prefixRanges = ranges, tailRanges = ranges;
    prefixRanges.arguments[loop.getInductionVar()] = {domain->lo,
                                                      domain->hi - *step};
    tailRanges.arguments[loop.getInductionVar()] = {domain->hi, domain->hi};
    auto extents = memoryExtents(loop, ranges);
    bool partialLast = llvm::any_of(extents, [&](const TileExtent &extent) {
      auto prefix = constantExtent(extent, prefixRanges);
      auto tail = constantExtent(extent, tailRanges);
      return prefix && *prefix == extent.tile && tail && *tail > 0 &&
             *tail < extent.tile;
    });
    if (!partialLast)
      continue;

    OpBuilder builder(loop);
    Value last = builder.create<arith::ConstantOp>(
        loop.getLoc(),
        builder.getIntegerAttr(loop.getInductionVar().getType(), domain->hi));
    IRMapping prefixMapping;
    auto prefix = cast<scf::ForOp>(builder.clone(*loop, prefixMapping));
    prefix.setUpperBound(last);

    // Build the last iteration from the prefix results, never the original
    // init operands. Replacing the old loop only after cloning keeps those
    // prefix uses separate from the last iteration's yielded results.
    IRMapping tailMapping;
    tailMapping.map(loop.getInductionVar(), last);
    tailMapping.map(loop.getRegionIterArgs(), prefix.getResults());
    for (Operation &op : loop.getBody()->without_terminator())
      builder.clone(op, tailMapping);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [result, value] :
         llvm::zip(loop.getResults(), yield.getOperands()))
      result.replaceAllUsesWith(tailMapping.lookupOrDefault(value));
    loop.erase();
  }
}

/// Linear launch-axis start. Other variables (including loop IVs) cannot
/// introduce a branch; their entire constant iteration range must be proven.
static std::optional<int64_t> programStep(Value start, Value pid,
                                          unsigned depth = 0) {
  if (depth > 16)
    return std::nullopt;
  if (start == pid)
    return 1;
  if (auto cast = start.getDefiningOp<arith::IndexCastOp>())
    return programStep(cast.getIn(), pid, depth + 1);
  if (auto mul = start.getDefiningOp<arith::MulIOp>()) {
    for (auto [variable, constant] : {std::pair{mul.getLhs(), mul.getRhs()},
                                      std::pair{mul.getRhs(), mul.getLhs()}}) {
      auto factor = getConstantIntValue(constant);
      auto step = programStep(variable, pid, depth + 1);
      int64_t product;
      if (factor && step && *factor > 0 &&
          !llvm::MulOverflow(*factor, *step, product))
        return product;
    }
  }
  return std::nullopt;
}

static void replaceExtents(ArrayRef<TileExtent> extents, const Ranges &ranges,
                           const IRMapping &mapping) {
  for (const TileExtent &extent : extents) {
    auto size = constantExtent(extent, ranges);
    if (!size)
      continue;
    Value value = mapping.lookupOrDefault(extent.value);
    OpBuilder builder(value.getDefiningOp());
    Value constant = builder.create<arith::ConstantOp>(
        value.getLoc(), builder.getIntegerAttr(value.getType(), *size));
    value.replaceAllUsesWith(constant);
  }
}

struct Partition {
  Value pid;
  int64_t fullPrograms, grid;
};

static SmallVector<Value>
cloneBranches(OpBuilder &builder, ArrayRef<Operation *> body,
              func::ReturnOp terminator, ArrayRef<Partition> partitions,
              const Ranges &ranges, ArrayRef<TileExtent> extents) {
  auto loc = terminator.getLoc();
  if (partitions.empty()) {
    IRMapping mapping;
    for (Operation *op : body)
      builder.clone(*op, mapping);
    replaceExtents(extents, ranges, mapping);
    SmallVector<Value> results;
    for (Value result : terminator.getOperands())
      results.push_back(mapping.lookupOrDefault(result));
    return results;
  }
  Partition p = partitions.front();
  Value limit = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(p.pid.getType(), p.fullPrograms));
  Value condition = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, p.pid, limit);
  auto branch = builder.create<scf::IfOp>(loc, terminator.getOperandTypes(),
                                          condition, /*withElseRegion=*/true);
  for (bool full : {true, false}) {
    Block *block = full ? branch.thenBlock() : branch.elseBlock();
    // The builder supplies empty yields only for result-less ifs.
    if (!block->empty())
      block->getTerminator()->erase();
    OpBuilder nested = OpBuilder::atBlockEnd(block);
    Ranges branchRanges = ranges;
    branchRanges.arguments[p.pid] = full ? Interval{0, p.fullPrograms - 1}
                                         : Interval{p.fullPrograms, p.grid - 1};
    auto results =
        cloneBranches(nested, body, terminator, partitions.drop_front(),
                      branchRanges, extents);
    nested.create<scf::YieldOp>(loc, results);
  }
  return SmallVector<Value>(branch.getResults());
}

/// Rebuild only memory-event subviews: refining unrelated users' types could
/// require casts at function or region boundaries.
static void refineMemorySubviews(func::FuncOp function) {
  SmallVector<memref::SubViewOp> views;
  function.walk([&](memref::SubViewOp view) {
    if (view.getType().getRank() == view.getSourceType().getRank() &&
        llvm::all_of(view->getUsers(), [](Operation *user) {
          return isa<memref::CopyOp>(user) && user->hasAttr("contract.access");
        }))
      views.push_back(view);
  });
  for (auto view : views) {
    OpBuilder builder(view);
    auto simplify = [&](SmallVector<OpFoldResult> values) {
      for (OpFoldResult &value : values)
        if (auto constant = getConstantIntValue(value))
          value = builder.getIndexAttr(*constant);
      return values;
    };
    auto offsets = simplify(view.getMixedOffsets());
    auto sizes = simplify(view.getMixedSizes());
    auto strides = simplify(view.getMixedStrides());
    auto refined = builder.create<memref::SubViewOp>(
        view.getLoc(), view.getSource(), offsets, sizes, strides);
    view.getResult().replaceAllUsesWith(refined.getResult());
    view.erase();
  }
}

static void updateMemoryEvents(func::FuncOp function) {
  function.walk([&](memref::CopyOp copy) {
    auto memory = copy->getAttrOfType<DictionaryAttr>("contract.access");
    if (!memory)
      return;
    auto logical = memory.getAs<DenseI64ArrayAttr>("logical_shape");
    auto source = dyn_cast<MemRefType>(copy.getSource().getType());
    auto destination = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!logical || !source || !destination ||
        source.getShape() != destination.getShape())
      return;
    NamedAttrList attrs(memory);
    OpBuilder builder(copy);
    bool full = source.getShape() == logical.asArrayRef();
    NamedAttrList mask;
    mask.set("kind", builder.getStringAttr(full ? "none" : "rectangular"));
    if (!full)
      mask.set("valid_shape", builder.getDenseI64ArrayAttr(source.getShape()));
    attrs.set("mask", mask.getDictionary(function.getContext()));
    copy->setAttr("contract.access",
                  attrs.getDictionary(function.getContext()));

    auto kind = memory.getAs<StringAttr>("kind");
    if (!full || !kind || kind.getValue() != "load")
      return;
    Value snapshot = copy.getTarget();
    if (auto view = snapshot.getDefiningOp<memref::SubViewOp>()) {
      if (!areAllConstantIntValue(view.getMixedOffsets(), 0) ||
          !areAllConstantIntValue(view.getMixedStrides(), 1) ||
          view.getSourceType().getShape() != logical.asArrayRef())
        return;
      snapshot = view.getSource();
    }
    // Only an unobserved, fully overwritten padding fill may disappear.
    // Accumulator fills lack contract.fill and are always retained.
    for (Operation *op = copy->getPrevNode(); op; op = op->getPrevNode()) {
      if (auto fill = dyn_cast<linalg::FillOp>(op)) {
        if (fill->hasAttr("contract.fill") && fill.getOutputs().size() == 1 &&
            fill.getOutputs()[0] == snapshot)
          fill.erase();
        break;
      }
      if (!isMemoryEffectFree(op))
        break;
    }
  });
}
} // namespace

LogicalResult mlir::triton::specializeContractTiles(ModuleOp module) {
  for (auto function : module.getOps<func::FuncOp>()) {
    auto knownGrid =
        function->getAttrOfType<DenseI32ArrayAttr>("gpu.known_grid_size");
    if (!knownGrid || knownGrid.size() != 3 || function.isExternal() ||
        !llvm::hasSingleElement(function.getBody()))
      continue;
    auto terminator =
        dyn_cast<func::ReturnOp>(function.front().getTerminator());
    if (!terminator)
      continue;
    Ranges ranges;
    std::array<Value, 3> pids{};
    for (unsigned i = 0; i < function.getNumArguments(); ++i) {
      auto argument =
          function.getArgAttrOfType<DictionaryAttr>(i, "contract.program");
      if (!argument)
        continue;
      auto role = argument.getAs<StringAttr>("role");
      auto axis = argument.getAs<IntegerAttr>("axis");
      if (!role || !axis || axis.getInt() < 0 || axis.getInt() >= 3)
        continue;
      int64_t grid = knownGrid[axis.getInt()];
      Value value = function.getArgument(i);
      if (!value.getType().isInteger(32))
        continue;
      if (role.getValue() == "program_id") {
        ranges.arguments[value] = {0, grid - 1};
        pids[axis.getInt()] = value;
      } else {
        ranges.arguments[value] = {grid, grid};
      }
    }
    SmallVector<arith::SelectOp> knownSelects;
    function.walk([&](arith::SelectOp select) {
      if (select.getType().isIntOrIndex() &&
          ranges.truth(select.getCondition()))
        knownSelects.push_back(select);
    });
    for (auto select : knownSelects) {
      bool condition = *ranges.truth(select.getCondition());
      select.getResult().replaceAllUsesWith(condition ? select.getTrueValue()
                                                      : select.getFalseValue());
      select.erase();
    }
    splitPartialLastIterations(function, ranges);
    SmallVector<TileExtent> extents = memoryExtents(function, ranges);
    SmallVector<Partition> partitions;
    for (unsigned axis = 0; axis < 3; ++axis) {
      Value pid = pids[axis];
      int64_t grid = knownGrid[axis];
      if (!pid || grid < 2)
        continue;
      for (const auto &extent : extents) {
        if (constantExtent(extent, ranges))
          continue;
        auto step = programStep(extent.start, pid);
        // Split only a full prefix and one constant tail program. Wider or
        // unproved tail domains retain the original clamp and diagnostics.
        if (!step || *step != extent.tile || extent.bound / *step != grid - 1 ||
            extent.bound % *step == 0)
          continue;
        partitions.push_back({pid, grid - 1, grid});
        break;
      }
    }
    if (partitions.empty()) {
      replaceExtents(extents, ranges, IRMapping());
    } else {
      SmallVector<Operation *> original;
      for (Operation &op : function.front().without_terminator())
        original.push_back(&op);
      OpBuilder builder(&function.front(), function.front().begin());
      auto results = cloneBranches(builder, original, terminator, partitions,
                                   ranges, extents);
      builder.create<func::ReturnOp>(terminator.getLoc(), results);
      terminator.erase();
      for (Operation *op : llvm::reverse(original))
        op->erase();
    }
    refineMemorySubviews(function);
    updateMemoryEvents(function);
  }
  return success();
}
