//===- KernelContract.h - Target-independent buffer contracts ----*- C++
//-*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_ANALYSIS_KERNELCONTRACT_H
#define TRITON_SHARED_ANALYSIS_KERNELCONTRACT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
class Operation;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::triton::contract {

/// Check every existing module/graph version before any normalization. Missing
/// versions are permitted on first annotation; existing versions must be v1.
mlir::LogicalResult checkKernelContractVersions(mlir::ModuleOp module);

/// Module argument ABI promises apply only to entry functions with no symbol
/// uses in the module. Called/address-taken helpers can receive aliasing or
/// unaligned views, and require separate call-site proofs.
bool hasEntryArgumentABI(mlir::func::FuncOp function);

/// Rebuild scopes, tasks, content versions and loop summaries from buffer-form
/// standard IR. Retain neutral access, ABI, provenance and origin facts. No
/// target dialect, bank, memory-space or route decision is introduced.
mlir::LogicalResult
annotateKernelContract(mlir::ModuleOp module,
                       llvm::StringRef producer = "kernel-contract");

/// Verify the neutral contract against actual operands, payloads and program
/// order. Only standard buffer IR with logical memory space zero is accepted.
/// Diagnostics identify the failing operation. Verification never rewrites IR.
mlir::LogicalResult verifyKernelContract(mlir::ModuleOp module);

std::unique_ptr<mlir::Pass> createKernelContractAnnotatePass();
std::unique_ptr<mlir::Pass> createKernelContractVerifyPass();
void registerKernelContractPasses();

} // namespace mlir::triton::contract

#endif // TRITON_SHARED_ANALYSIS_KERNELCONTRACT_H
