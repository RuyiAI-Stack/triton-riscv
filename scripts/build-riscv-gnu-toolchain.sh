#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_script_dir}/.." && pwd)"
BUDDY_DIR="${BUDDY_DIR:-$(cd "${_repo_root}/../buddy-mlir" && pwd)}"
BUDDY_BUILD_DIR="${BUDDY_BUILD_DIR:-${BUDDY_DIR}/build-for-triton-riscv}"
_toolchain_prefix="${BUDDY_BUILD_DIR}/thirdparty/riscv-gnu-toolchain"
_jobs="${JOBS:-16}"

if [[ ! -d "${BUDDY_DIR}/.git" ]]; then
  echo "Buddy source tree not found: ${BUDDY_DIR}" >&2
  exit 1
fi

_cmake_args=(
  -S "${BUDDY_DIR}"
  -B "${BUDDY_BUILD_DIR}"
  -G Ninja
  -DBUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN=ON
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON
  -DLLVM_ENABLE_ASSERTIONS=ON
  -DCMAKE_BUILD_TYPE=RELEASE
  -DLLVM_DIR="${LLVM_DIR:-${BUDDY_DIR}/llvm/build/lib/cmake/llvm}"
  -DMLIR_DIR="${MLIR_DIR:-${BUDDY_DIR}/llvm/build/lib/cmake/mlir}"
  -DPython3_EXECUTABLE="${Python3_EXECUTABLE:-$(command -v python3)}"
)
cmake "${_cmake_args[@]}"
env -u PYTHONSAFEPATH ninja -C "${BUDDY_BUILD_DIR}" -j"${_jobs}"

"${_toolchain_prefix}/bin/riscv64-unknown-linux-gnu-gcc" --version | head -n1
"${_toolchain_prefix}/bin/qemu-riscv64" --version | head -n1
