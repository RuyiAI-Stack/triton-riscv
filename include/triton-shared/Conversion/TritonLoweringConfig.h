//===- TritonLoweringConfig.h - Shared Triton backend selection ----*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
