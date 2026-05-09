#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRITON_RISCV_DIR="${ROOT_DIR}"
TRITON_DIR="${ROOT_DIR}/triton"
TRITON_HASH="$(cat "${TRITON_RISCV_DIR}/triton-hash.txt")"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

TOOLCHAIN_DIR="${TRITON_RISCV_DIR}/.cache"
LLVM_EXTRACT_DIR="${TOOLCHAIN_DIR}/llvm"
BUDDY_EXTRACT_DIR="${TOOLCHAIN_DIR}/buddy"
LLVM_BINARY_DIR="${LLVM_EXTRACT_DIR}/bin"
BUDDY_MLIR_BINARY_DIR="${BUDDY_EXTRACT_DIR}/bin"
PY_TAG="${TRITON_PYTHON_TAG:-cp312-cp312}"

case "$(uname -m)" in
    amd64|x86_64)
        LLVM_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/release/v0.0.3/llvm-22.0.0git-${PY_TAG}-manylinux_2_28_x86_64.tar.gz"
        BUDDY_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/release/v0.0.3/buddy-0.0.3-${PY_TAG}-manylinux_2_28_x86_64.tar.gz"
        ;;
    riscv64)
        LLVM_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/release/v0.0.3/llvm-22.0.0git-${PY_TAG}-manylinux_2_39_riscv64.tar.gz"
        BUDDY_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/release/v0.0.3/buddy-0.0.3-${PY_TAG}-manylinux_2_39_riscv64.tar.gz"
        ;;
    *)
        echo "unsupported arch: $(uname -m)" >&2
        exit 1
        ;;
esac

LLVM_URL="${TRITON_LLVM_PACKAGE_URL:-${LLVM_DEFAULT_URL}}"
BUDDY_URL="${TRITON_BUDDY_PACKAGE_URL:-${BUDDY_DEFAULT_URL}}"

if [ ! -x "${PYTHON_BIN}" ]; then
    python3 -m venv "${ROOT_DIR}/.venv"
fi

if [ ! -d "${TRITON_DIR}" ]; then
    git clone https://github.com/triton-lang/triton.git "${TRITON_DIR}"
fi

git -C "${TRITON_DIR}" reset --hard "${TRITON_HASH}"

"${TRITON_RISCV_DIR}/scripts/apply_patches.sh" "${TRITON_DIR}"

mkdir -p "${TOOLCHAIN_DIR}"

if [ ! -d "${LLVM_BINARY_DIR}" ]; then
    rm -rf "${LLVM_EXTRACT_DIR}"
    mkdir -p "${LLVM_EXTRACT_DIR}"
    curl -L "${LLVM_URL}" -o "${TOOLCHAIN_DIR}/llvm.tar.gz"
    tar -xzf "${TOOLCHAIN_DIR}/llvm.tar.gz" -C "${LLVM_EXTRACT_DIR}"
    rm -f "${TOOLCHAIN_DIR}/llvm.tar.gz"
fi

if [ ! -d "${BUDDY_MLIR_BINARY_DIR}" ]; then
    rm -rf "${BUDDY_EXTRACT_DIR}"
    mkdir -p "${BUDDY_EXTRACT_DIR}"
    curl -L "${BUDDY_URL}" -o "${TOOLCHAIN_DIR}/buddy.tar.gz"
    tar -xzf "${TOOLCHAIN_DIR}/buddy.tar.gz" -C "${BUDDY_EXTRACT_DIR}"
    rm -f "${TOOLCHAIN_DIR}/buddy.tar.gz"
fi

[ -d "${LLVM_BINARY_DIR}" ]
[ -d "${BUDDY_MLIR_BINARY_DIR}" ]

export TRITON_PLUGIN_DIRS="${TRITON_RISCV_DIR}"
export LLVM_BINARY_DIR
export LLVM_SYSPATH="$(dirname "${LLVM_BINARY_DIR}")"
export BUDDY_MLIR_BINARY_DIR

cd "${TRITON_DIR}"
echo "Installing Triton into ${PYTHON_BIN}..."


case "$(uname -m)" in
    riscv64)
        pip install numpy --index-url https://gitlab.com/api/v4/projects/56254198/packages/pypi/simple
        ;;
    *)
        pip install numpy
        ;;
esac

"${PYTHON_BIN}" -m pip install -U \
    "setuptools>=40.8.0" \
    "cmake>=3.20,<4.0" \
    "ninja>=1.11.1" \
    "pybind11>=2.13.1" \
    "pytest"
"${PYTHON_BIN}" -m pip install --no-build-isolation -e .
