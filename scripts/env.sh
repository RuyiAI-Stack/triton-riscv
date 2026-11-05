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

TRITON_RISCV_DIR="${TRITON_RISCV_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRITON_DIR="${TRITON_DIR:-${TRITON_RISCV_DIR}/triton}"
TRITON_VENV="${TRITON_VENV:-${CONDA_PREFIX:-${TRITON_RISCV_DIR}/.venv}}"

# The toolchains scripts/build.sh downloads.
LLVM_SYSPATH="${LLVM_SYSPATH:-${TRITON_RISCV_DIR}/.cache/llvm}"
LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${LLVM_SYSPATH}/bin}"
BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-${TRITON_RISCV_DIR}/.cache/buddy/bin}"

# ---------------------------------------------------------------------------
# Build time
# ---------------------------------------------------------------------------

# Upstream Triton downloads host-specific GPU toolchains that riscv64 has no
# builds for; disable them for every build started from this shell.
# shellcheck source=scripts/_gpu-stub-env.sh
source "${TRITON_RISCV_DIR}/scripts/_gpu-stub-env.sh" "${TRITON_RISCV_DIR}/.cache/gpu-deps-stub"
# Triton's plugin support, which packages backend/ as
# triton.backends.triton_shared and builds the C++ plugin into libtriton.so.
export TRITON_PLUGIN_DIRS="${TRITON_RISCV_DIR}"
# Build against the Python stable ABI (abi3), which is how upstream builds its
# abi3 wheel: nanobind is compiled with the limited API, and the wheel is tagged
# cp312-abi3, so it covers Python 3.12 and later instead of only the interpreter
# it was built with. Upstream ships no abi3 wheel for free-threaded builds.
export TRITON_STABLE_ABI="${TRITON_STABLE_ABI:-ON}"

# ---------------------------------------------------------------------------
# Run time
# ---------------------------------------------------------------------------

# Read by Triton.
TRITON_HOME="${TRITON_HOME:-${HOME}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_HOME}/.triton/cache}"
TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-${TRITON_HOME}/.triton/dump}"
TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR:-${TRITON_HOME}/.triton/override}"

# Read by the backend.
TRITON_SHARED_DUMP_PATH="${TRITON_SHARED_DUMP_PATH:-${TRITON_DUMP_DIR}/shared}"

# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

export TRITON_RISCV_DIR TRITON_DIR TRITON_VENV
export LLVM_SYSPATH LLVM_BINARY_DIR BUDDY_MLIR_BINARY_DIR
export TRITON_HOME TRITON_CACHE_DIR TRITON_DUMP_DIR TRITON_OVERRIDE_DIR
export TRITON_SHARED_DUMP_PATH
export PATH="${TRITON_VENV}/bin:${LLVM_BINARY_DIR}:${BUDDY_MLIR_BINARY_DIR}:${PATH}"
# Keep the repository's own triton/ checkout from shadowing the installed one.
export PYTHONSAFEPATH=1
