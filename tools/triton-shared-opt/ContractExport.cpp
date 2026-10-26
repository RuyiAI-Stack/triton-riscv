// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ContractExport.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton-shared/Analysis/KernelContract.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace {
struct PrepareInput : PassWrapper<PrepareInput, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareInput)
  PrepareInput() = default;
  PrepareInput(const PrepareInput &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "triton-contract-prepare"; }
  StringRef getDescription() const final {
    return "Check TTIR input and set explicit Contract entry/math promises";
  }
  Option<std::string> entryName{*this, "entry-name", llvm::cl::init("")};
  Option<std::string> mathAccuracy{*this, "math-accuracy", llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(mlir::triton::contract::checkKernelContractVersions(module)))
      return signalPassFailure();
    // Only Triton input, never a later stage or a Gluon module.
    if (module
            .walk([&](Operation *op) -> WalkResult {
              StringRef ns = op->getName().getDialectNamespace();
              if (ns != "builtin" && ns != "tt" && ns != "arith" &&
                  ns != "math" && ns != "scf" && ns != "cf" && ns != "tensor" &&
                  ns != "ub") {
                op->emitOpError(
                    "Contract export accepts Triton TTIR input only");
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            })
            .wasInterrupted())
      return signalPassFailure();
    SmallVector<triton::FuncOp> entries;
    for (auto function : module.getOps<triton::FuncOp>())
      if (!function.isExternal() && function.isPublic())
        entries.push_back(function);
    if (entries.size() != 1) {
      module.emitError(
          "Contract export requires exactly one public TTIR entry");
      return signalPassFailure();
    }
    auto entry = entries.front();
    if (!entryName.empty() && entryName != entry.getSymName()) {
      SymbolTable symbols(module);
      if (failed(symbols.rename(entry, entryName))) {
        entry.emitError(
            "Contract entry rename conflicts with an existing symbol");
        return signalPassFailure();
      }
    }
    if (!mathAccuracy.empty()) {
      auto attr = dyn_cast_or_null<DictionaryAttr>(
          parseAttribute(mathAccuracy, &getContext()));
      if (!attr || attr.empty()) {
        entry.emitError("math accuracy must be a nonempty dictionary");
        return signalPassFailure();
      }
      if (auto previous = entry->getAttr("contract.math");
          previous && previous != attr) {
        entry.emitError("math accuracy conflicts with contract.math");
        return signalPassFailure();
      }
      entry->setAttr("contract.math", attr);
    }
  }
};

static std::string typeString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

static llvm::json::Value jsonAttribute(Attribute attr) {
  if (auto dictionary = dyn_cast_or_null<DictionaryAttr>(attr)) {
    llvm::json::Object object;
    for (auto entry : dictionary)
      object[entry.getName().strref()] = jsonAttribute(entry.getValue());
    return object;
  }
  if (auto boolean = dyn_cast_or_null<BoolAttr>(attr))
    return boolean.getValue();
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return integer.getInt();
  if (auto number = dyn_cast_or_null<FloatAttr>(attr))
    return number.getValueAsDouble();
  if (auto string = dyn_cast_or_null<StringAttr>(attr))
    return string.getValue().str();
  if (auto array = dyn_cast_or_null<DenseI32ArrayAttr>(attr)) {
    llvm::json::Array values;
    for (int32_t value : array.asArrayRef())
      values.push_back(value);
    return values;
  }
  if (auto array = dyn_cast_or_null<DenseI64ArrayAttr>(attr)) {
    llvm::json::Array values;
    for (int64_t value : array.asArrayRef())
      values.push_back(value);
    return values;
  }
  if (auto array = dyn_cast_or_null<ArrayAttr>(attr)) {
    llvm::json::Array values;
    for (Attribute value : array)
      values.push_back(jsonAttribute(value));
    return values;
  }
  return nullptr;
}

struct ReportContract : PassWrapper<ReportContract, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportContract)
  ReportContract() = default;
  ReportContract(const ReportContract &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "triton-contract-report"; }
  StringRef getDescription() const final {
    return "Verify Contract and report actual entry ABI, promises and graph "
           "limits";
  }
  Option<std::string> output{*this, "output", llvm::cl::init("")};
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(mlir::triton::contract::verifyKernelContract(module)))
      return signalPassFailure();
    llvm::json::Array functions;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      llvm::json::Array arguments, results;
      for (Type type : function.getArgumentTypes())
        arguments.push_back(typeString(type));
      for (Type type : function.getResultTypes())
        results.push_back(typeString(type));
      functions.push_back(llvm::json::Object{
          {"name", function.getSymName().str()},
          {"arguments", std::move(arguments)},
          {"results", std::move(results)},
          {"grid", jsonAttribute(function->getAttr("contract.grid"))},
          {"math_accuracy", jsonAttribute(function->getAttr("contract.math"))},
          {"graph",
           jsonAttribute(function->getAttr("contract.compute_graph"))}});
    }
    llvm::json::Object report{
        {"functions", std::move(functions)},
        {"caller_contract", jsonAttribute(module->getAttr("contract.abi"))}};
    std::error_code error;
    llvm::raw_fd_ostream stream(output, error, llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError("cannot write Contract report: ") << error.message();
      return signalPassFailure();
    }
    stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
    stream.flush();
    if (stream.has_error()) {
      module.emitError("cannot flush Contract report");
      signalPassFailure();
    }
  }
};
} // namespace

void registerContractExportPasses() {
  PassRegistration<PrepareInput>();
  PassRegistration<ReportContract>();
}
