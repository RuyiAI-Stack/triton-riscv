#!/usr/bin/env bash
#
# Development environment for a triton-riscv source checkout. Source it, do not
# execute it:
#
#   source scripts/env.sh
#
# Export any of the values below beforehand to override them.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/env.sh" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

_TRITON_RISCV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_VENV="${TRITON_VENV:-${CONDA_PREFIX:-${_TRITON_RISCV_DIR}/.venv}}"

# The toolchains scripts/build.sh downloads.
LLVM_SYSPATH="${LLVM_SYSPATH:-${_TRITON_RISCV_DIR}/.cache/llvm}"
LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${LLVM_SYSPATH}/bin}"
BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-${_TRITON_RISCV_DIR}/.cache/buddy/bin}"

# ---------------------------------------------------------------------------
# Build time
# ---------------------------------------------------------------------------

# Upstream Triton downloads host-specific GPU toolchains that riscv64 has no
# builds for; disable them for every build started from this shell.
# shellcheck source=scripts/_gpu-stub-env.sh
source "${_TRITON_RISCV_DIR}/scripts/_gpu-stub-env.sh" "${_TRITON_RISCV_DIR}/.cache/gpu-deps-stub"
# That stub points every GPU download at an empty directory, which is also what
# skips upstream's AMD codegen fetch during the build. The backend reads this one
# again when it compiles, though, so point it at the library scripts/setup.sh
# builds -- the same file get_amd_codegen_path() falls back to by default.
export TRITON_AMD_CODEGEN_PATH="${_TRITON_RISCV_DIR}/triton/third_party/amd/backend/lib/libtriton_amd_codegen.so"
# Triton's plugin support, which packages backend/ as
# triton.backends.triton_shared and builds the C++ plugin into libtriton.so.
export TRITON_PLUGIN_DIRS="${_TRITON_RISCV_DIR}"
# Build against the Python stable ABI (abi3), which is how upstream builds its
# abi3 wheel: nanobind is compiled with the limited API, and the wheel is tagged
# cp312-abi3, so it covers Python 3.12 and later instead of only the interpreter
# it was built with. Upstream ships no abi3 wheel for free-threaded builds.
export TRITON_STABLE_ABI="${TRITON_STABLE_ABI:-ON}"

# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

export LLVM_SYSPATH LLVM_BINARY_DIR BUDDY_MLIR_BINARY_DIR
export PATH="${TRITON_VENV}/bin:${LLVM_BINARY_DIR}:${BUDDY_MLIR_BINARY_DIR}:${PATH}"
# Keep the repository's own triton/ checkout from shadowing the installed one.
export PYTHONSAFEPATH=1
