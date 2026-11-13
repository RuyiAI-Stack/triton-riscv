//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// Driver for triton-shared-opt: dialect registration, contract export CLI, and
// MlirOptMain.
//
// Inputs: command-line passes plus optional --contract-build-info.
//
// Outputs: transformed MLIR on stdout, or the build identity string for pinning
// toolchains.
//
// When used: lit tests and local debugging of BOSCAME AOT conversions.
//

#include "ContractBuildIdentity.h"
#include "ContractExport.h"
#include "RegisterTritonSharedDialects.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--contract-build-info") == 0) {
    llvm::outs() << contractBuildIdentity << "\n";
    return 0;
  }
  registerContractExportPasses();
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Shared test driver\n", registry));
}
