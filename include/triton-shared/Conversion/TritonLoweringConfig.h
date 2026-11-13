//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//
// Shared knobs for Triton to Linalg/MemRef lowering.
//
// Inputs: pass options from compiler.py and triton-shared-opt (contract
// emission, BOSCAME AOT).
//
// Outputs: a config object consumed by conversion patterns and the experimental
// pipeline.
//
// When used: any shared lowering that must stay target-neutral for FPGA AOT.
//

#ifndef TRITON_SHARED_CONVERSION_TRITONLOWERINGCONFIG_H
#define TRITON_SHARED_CONVERSION_TRITONLOWERINGCONFIG_H

#include <string>

namespace mlir::triton {

/// Internal configuration, independent of any driver or command-line options.
struct TritonLoweringConfig {
  bool emitContract = false;
};

} // namespace mlir::triton

#endif
