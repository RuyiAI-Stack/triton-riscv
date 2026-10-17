#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif

using namespace llvm;

namespace {

struct SanitizerAttributes : RequiredPassInfoMixin<SanitizerAttributes> {
  explicit SanitizerAttributes(Attribute::AttrKind Type) : Type(Type) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    F.addFnAttr(Type);

    // this pass modifies all function attributes
    return PreservedAnalyses::none();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }

private:
  Attribute::AttrKind Type;
};
} // namespace

llvm::PassPluginLibraryInfo getSanitizerAttributesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SanitizerAttributes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  // opt loads pass plugins after parsing command-line options.
                  // Keep the sanitizer type in the pipeline, not a global
                  // option.
                  if (Name == "sanitizer-attributes<asan>") {
                    FPM.addPass(
                        SanitizerAttributes(Attribute::SanitizeAddress));
                    return true;
                  }
                  if (Name == "sanitizer-attributes<tsan>") {
                    FPM.addPass(SanitizerAttributes(Attribute::SanitizeThread));
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSanitizerAttributesPluginInfo();
}
