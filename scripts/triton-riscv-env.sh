#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/triton-riscv-env.sh" >&2
  exit 1
fi

_triton_riscv_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_triton_riscv_repo_root="$(cd "${_triton_riscv_env_dir}/.." && pwd)"

TRITON_RISCV_DIR="${TRITON_RISCV_DIR:-${_triton_riscv_repo_root}}"
TRITON_DIR="${TRITON_DIR:-$(cd "${TRITON_RISCV_DIR}/../triton" && pwd)}"
BUDDY_DIR="${BUDDY_DIR:-$(cd "${TRITON_RISCV_DIR}/../buddy-mlir" && pwd)}"
TRITON_VENV="${TRITON_VENV:-${TRITON_RISCV_DIR}/.venv}"
TRITON_HOME="${TRITON_HOME:-${HOME}}"
TRITON_RUNTIME_ROOT="${TRITON_RUNTIME_ROOT:-${HOME}/.triton}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_RUNTIME_ROOT}/cache}"
TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-${TRITON_RUNTIME_ROOT}/dump}"
TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR:-${TRITON_RUNTIME_ROOT}/override}"
TRITON_SHARED_DUMP_PATH="${TRITON_SHARED_DUMP_PATH:-${TRITON_DUMP_DIR}/shared}"

if [[ -z "${BUILD_DIR:-}" ]]; then
  _detected_python_tag=""
  if [[ -x "${TRITON_VENV}/bin/python" ]]; then
    _detected_python_tag="$("${TRITON_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  fi

  if [[ -n "${_detected_python_tag}" ]]; then
    _detected_build_dir="$(
      find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name "cmake.linux-*-cpython-${_detected_python_tag}" 2>/dev/null | sort | head -n1
    )"
  fi

  if [[ -z "${_detected_build_dir:-}" ]]; then
    _detected_build_dir="$(
      find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name 'cmake.linux-*-cpython-*' 2>/dev/null | sort | head -n1
    )"
  fi

  BUILD_DIR="${_detected_build_dir}"
fi

if [[ -z "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  _candidate_triton_shared_opt_paths=(
    "${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
  )
  for _candidate_triton_shared_opt_path in "${_candidate_triton_shared_opt_paths[@]}"; do
    if [[ -n "${_candidate_triton_shared_opt_path}" && -x "${_candidate_triton_shared_opt_path}" ]]; then
      TRITON_SHARED_OPT_PATH="${_candidate_triton_shared_opt_path}"
      break
    fi
  done
fi

LLVM_SYSPATH="${LLVM_SYSPATH:-${BUDDY_DIR}/llvm/build}"
LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${LLVM_SYSPATH}/bin}"
BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-${BUDDY_DIR}/build/bin}"
TRITON_RISCV_LOWERING_MODE="${TRITON_RISCV_LOWERING_MODE:-linalg_loops}"

export TRITON_RISCV_DIR
export TRITON_DIR
export TRITON_VENV
export TRITON_HOME
export TRITON_RUNTIME_ROOT
export TRITON_CACHE_DIR
export TRITON_DUMP_DIR
export TRITON_OVERRIDE_DIR
export BUILD_DIR
export TRITON_PLUGIN_DIRS="${TRITON_RISCV_DIR}"
export LLVM_SYSPATH
export LLVM_BINARY_DIR
export BUDDY_MLIR_BINARY_DIR
export TRITON_SHARED_OPT_PATH
export TRITON_SHARED_DUMP_PATH
export TRITON_RISCV_LOWERING_MODE
export PATH="${TRITON_VENV}/bin:${LLVM_BINARY_DIR}:${BUDDY_MLIR_BINARY_DIR}:${PATH}"

unset _detected_build_dir
unset _detected_python_tag
unset _candidate_triton_shared_opt_path
unset _candidate_triton_shared_opt_paths
unset _triton_riscv_env_dir
unset _triton_riscv_repo_root
