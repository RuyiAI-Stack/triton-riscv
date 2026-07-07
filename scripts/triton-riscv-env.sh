#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/triton-riscv-env.sh" >&2
  exit 1
fi

_triton_riscv_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_triton_riscv_repo_root="$(cd "${_triton_riscv_env_dir}/.." && pwd)"

TRITON_RISCV_DIR="${TRITON_RISCV_DIR:-${_triton_riscv_repo_root}}"
_default_triton_venv="${TRITON_RISCV_DIR}/.venv"

_resolve_first_dir() {
  local _candidate
  for _candidate in "$@"; do
    if [[ -d "${_candidate}" ]]; then
      cd "${_candidate}" && pwd
      return 0
    fi
  done
  return 1
}

if [[ -z "${TRITON_DIR:-}" ]]; then
  TRITON_DIR="$(
    _resolve_first_dir \
      "${TRITON_RISCV_DIR}/../triton" \
      "${TRITON_RISCV_DIR}/triton" \
      2>/dev/null || true
  )"
  TRITON_DIR="${TRITON_DIR:-${TRITON_RISCV_DIR}/../triton}"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" && ( -z "${TRITON_VENV:-}" || "${TRITON_VENV}" == "${_default_triton_venv}" ) ]]; then
  TRITON_VENV="${CONDA_PREFIX}"
elif [[ -z "${TRITON_VENV:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    TRITON_VENV="${CONDA_PREFIX}"
  elif [[ -x "${_default_triton_venv}/bin/python" ]]; then
    TRITON_VENV="${_default_triton_venv}"
  else
    TRITON_VENV="${_default_triton_venv}"
  fi
fi

if [[ -z "${BUDDY_DIR:-}" ]]; then
  BUDDY_DIR="$(
    _resolve_first_dir \
      "${TRITON_RISCV_DIR}/../buddy-mlir" \
      "${TRITON_RISCV_DIR}/buddy-mlir" \
      "${TRITON_RISCV_DIR}/.cache/buddy" \
      2>/dev/null || true
  )"
  BUDDY_DIR="${BUDDY_DIR:-${TRITON_RISCV_DIR}/../buddy-mlir}"
fi
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
      find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name "cmake.linux-*-cpython-${_detected_python_tag}" 2>/dev/null | sort | head -n1 || true
    )"
  fi

  if [[ -z "${_detected_build_dir:-}" ]]; then
    _detected_build_dir="$(
      find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name 'cmake.linux-*-cpython-*' 2>/dev/null | sort | head -n1 || true
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

if [[ -z "${LLVM_SYSPATH:-}" ]]; then
  LLVM_SYSPATH="$(
    _resolve_first_dir \
      "${BUDDY_DIR}/llvm/build" \
      "${TRITON_RISCV_DIR}/.cache/llvm" \
      2>/dev/null || true
  )"
  LLVM_SYSPATH="${LLVM_SYSPATH:-${BUDDY_DIR}/llvm/build}"
fi
JSON_SYSPATH="${JSON_SYSPATH:-${TRITON_RUNTIME_ROOT}/json}"
LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${LLVM_SYSPATH}/bin}"
if [[ -z "${BUDDY_MLIR_BINARY_DIR:-}" ]]; then
  BUDDY_MLIR_BINARY_DIR="$(
    _resolve_first_dir \
      "${BUDDY_DIR}/build/bin" \
      "${BUDDY_DIR}/bin" \
      "${TRITON_RISCV_DIR}/.cache/buddy/bin" \
      2>/dev/null || true
  )"
  BUDDY_MLIR_BINARY_DIR="${BUDDY_MLIR_BINARY_DIR:-${BUDDY_DIR}/build/bin}"
fi
if [[ -z "${RISCV_GNU_TOOLCHAIN_DIR:-}" ]]; then
  RISCV_GNU_TOOLCHAIN_DIR="$(
    _resolve_first_dir \
      "${BUDDY_DIR}/build-for-triton-riscv/thirdparty/riscv-gnu-toolchain" \
      "${TRITON_RISCV_DIR}/.cache/riscv-toolchain" \
      2>/dev/null || true
  )"
  RISCV_GNU_TOOLCHAIN_DIR="${RISCV_GNU_TOOLCHAIN_DIR:-${BUDDY_DIR}/build-for-triton-riscv/thirdparty/riscv-gnu-toolchain}"
fi
if [[ -x "${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-gcc" ]]; then
  TRITON_RISCV_CC="${TRITON_RISCV_CC:-${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-gcc}"
  TRITON_RISCV_OBJDUMP="${TRITON_RISCV_OBJDUMP:-${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-objdump}"
  TRITON_RISCV_SYSROOT="${TRITON_RISCV_SYSROOT:-${RISCV_GNU_TOOLCHAIN_DIR}/sysroot}"
elif [[ -x "${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-linux-gnu-gcc-wrap" ]]; then
  TRITON_RISCV_CC="${TRITON_RISCV_CC:-${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-linux-gnu-gcc-wrap}"
  TRITON_RISCV_OBJDUMP="${TRITON_RISCV_OBJDUMP:-${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-linux-gnu-objdump-wrap}"
  TRITON_RISCV_SYSROOT="${TRITON_RISCV_SYSROOT:-${RISCV_GNU_TOOLCHAIN_DIR}/usr/riscv64-linux-gnu}"
fi
if [[ -x "${RISCV_GNU_TOOLCHAIN_DIR}/bin/qemu-riscv64" ]]; then
  TRITON_RISCV_QEMU="${TRITON_RISCV_QEMU:-${RISCV_GNU_TOOLCHAIN_DIR}/bin/qemu-riscv64}"
elif [[ -x "${TRITON_RISCV_DIR}/.cache/qemu/bin/qemu-riscv64" ]]; then
  TRITON_RISCV_QEMU="${TRITON_RISCV_QEMU:-${TRITON_RISCV_DIR}/.cache/qemu/bin/qemu-riscv64}"
fi
if [[ -n "${TRITON_RISCV_QEMU:-}" ]]; then
  TRITON_RISCV_QEMU_CPU="${TRITON_RISCV_QEMU_CPU:-rv64,v=true,vlen=256,elen=64,vext_spec=v1.0}"
fi
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
export JSON_SYSPATH
export LLVM_BINARY_DIR
export BUDDY_MLIR_BINARY_DIR
export RISCV_GNU_TOOLCHAIN_DIR
export TRITON_RISCV_CC
export TRITON_RISCV_OBJDUMP
export TRITON_RISCV_SYSROOT
export TRITON_RISCV_QEMU
export TRITON_RISCV_QEMU_CPU
export TRITON_SHARED_OPT_PATH
export TRITON_SHARED_DUMP_PATH
export TRITON_RISCV_LOWERING_MODE
export PATH="${TRITON_VENV}/bin:${LLVM_BINARY_DIR}:${BUDDY_MLIR_BINARY_DIR}:${PATH}"

export PYTHONSAFEPATH=1

unset _candidate_build_dir
unset _default_triton_venv
unset _detected_build_dir
unset _detected_python_tag
unset _candidate_triton_shared_opt_path
unset _candidate_triton_shared_opt_paths
unset _triton_riscv_env_dir
unset _triton_riscv_repo_root
unset -f _resolve_first_dir
