//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// triton-shared-opt contract export: dump verified BOSCAME AOT artifacts from
// MLIR.
//
// Inputs: a module after emitContract / kernel-contract-annotate, plus CLI
// flags.
//
// Outputs: contract MLIR/JSON and optional build-identity strings for
// reproducibility.
//
// When used: `triton-shared-opt --contract-*` and FPGA export scripts.
//

#pragma once

void registerContractExportPasses();
