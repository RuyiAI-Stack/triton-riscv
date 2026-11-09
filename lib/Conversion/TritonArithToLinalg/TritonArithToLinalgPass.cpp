//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// Convert Triton arithmetic and tensor ops into Linalg.
//
// Inputs: tt.elementwise, reductions, broadcasts, and related arith on tensor
// types.
//
// Outputs: linalg.generic / named ops used by the experimental and BOSCAME
// pipelines.
//
// When used: early in TritonToLinalgExperimental before contract tile
// specialization.
//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-arith-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TritonArithToLinalgPass
    : public triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass> {
  using TritonArithToLinalgBase<
      TritonArithToLinalgPass>::TritonArithToLinalgBase;

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func, bool attachAxisAttrs) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add the corresponding arguments to function body before installing
    // argument attributes, whose interface verifier checks body argument
    // count against the attribute list.
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }

    SmallVector<DictionaryAttr> newArgAttrs;
    if (func.getAllArgAttrs() || attachAxisAttrs) {
      if (func.getAllArgAttrs())
        func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.resize(origInputTypes.size() + TRITON_PROGRAM_INFO_ARG_COUNT);
      if (attachAxisAttrs) {
        for (unsigned axis = 0; axis < LAUNCH_GRID_RANK; ++axis) {
          NamedAttrList numProgramsAttrs(
              newArgAttrs[origInputTypes.size() + axis]);
          numProgramsAttrs.set(
              "contract.program",
              b.getDictionaryAttr(
                  {b.getNamedAttr("role", b.getStringAttr("num_programs")),
                   b.getNamedAttr("axis", b.getI32IntegerAttr(axis))}));
          newArgAttrs[origInputTypes.size() + axis] =
              numProgramsAttrs.getDictionary(b.getContext());

          NamedAttrList programIdAttrs(
              newArgAttrs[origInputTypes.size() + LAUNCH_GRID_RANK + axis]);
          programIdAttrs.set(
              "contract.program",
              b.getDictionaryAttr(
                  {b.getNamedAttr("role", b.getStringAttr("program_id")),
                   b.getNamedAttr("axis", b.getI32IntegerAttr(axis))}));
          newArgAttrs[origInputTypes.size() + LAUNCH_GRID_RANK + axis] =
              programIdAttrs.getDictionary(b.getContext());
        }
      }
      func.setAllArgAttrs(newArgAttrs);
    }
  }

  LogicalResult applyTensorConcatDecomposition() {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

public:
  TritonLoweringConfig config;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (config.emitContract) {
      WalkResult remainingMemory = moduleOp.walk([&](Operation *op) {
        if (isa<triton::CallOp>(op)) {
          op->emitError("neutral contract uninlined calls are unsupported in "
                        "contract v1");
          return WalkResult::interrupt();
        }
        if (!isa<triton::LoadOp, triton::StoreOp>(op))
          return WalkResult::advance();
        op->emitError("neutral contract unstructured/scalar memory access is "
                      "unsupported in contract v1");
        return WalkResult::interrupt();
      });
      if (remainingMemory.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      populateTritonArithToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, tts::TritonStructuredDialect>();

    target.addLegalOp<ModuleOp>();

    target.addLegalOp<triton::FuncOp, triton::ReturnOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          // Lower dense constant to linalg.fill
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType())) {
              return true;
            }

            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType())) {
                return false;
              }
            }
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    if (pidsToFuncArgs) {
      target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    }

    if (addptrToLinalg) {
      target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
        return !isa<ShapedType>(op.getResult().getType());
      });
    }

    target.addDynamicallyLegalOp<triton::BitcastOp>(
        [this](triton::BitcastOp op) {
          if (!tensorPtrToLinalg) {
            return triton::isPtrTypeLike(op.getType());
          } else {
            if (triton::isPtrTypeLike(op.getType())) {
              return !isa<ShapedType>(op.getType());
            }
            return false;
          }
        });

    // TODO: Might want to consolidate this flag with addptrToLinalg later.
    if (tensorPtrToLinalg) {
      target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp,
                                   triton::IntToPtrOp, triton::PtrToIntOp>(
          [](auto op) {
            return !isa<ShapedType>(op->getOperands()[0].getType());
          });
      populateTritonTensorPtrConversionPatterns(patterns);
    }

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    triton::populateTritonArithToLinalgConversionPatterns(
        pidsToFuncArgs, addptrToLinalg, assertToCf, transposeReduceToRank0,
        patterns, config);

    if (pidsToFuncArgs) {
      for (auto func : getOperation().getOps<triton::FuncOp>()) {
        addProgramInfo(func, config.emitContract);
      }
    }

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
    }

    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc =
            func::FuncOp::create(builder, func.getLoc(), name, type);
        // Preserve the visibility attribute
        funcFunc.setVisibility(func.getVisibility());
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);
        if (config.emitContract)
          for (NamedAttribute attr : func->getAttrs())
            if (attr.getName() != SymbolTable::getSymbolAttrName() &&
                attr.getName() != "function_type" &&
                attr.getName() != "arg_attrs" &&
                attr.getName() != "res_attrs" &&
                attr.getName() != "sym_visibility")
              funcFunc->setAttr(attr.getName(), attr.getValue());

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            func::ReturnOp::create(builder, func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass(bool tensorPtrToLinalg,
                                      bool transposeReduceToRank0,
                                      TritonLoweringConfig config) {
  TritonArithToLinalgOptions options;
  options.tensorPtrToLinalg = tensorPtrToLinalg;
  options.transposeReduceToRank0 = transposeReduceToRank0;
  auto pass = std::make_unique<TritonArithToLinalgPass>(options);
  pass->config = std::move(config);
  return pass;
}
