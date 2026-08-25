#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRITON_RISCV_DIR="${ROOT_DIR}"
TRITON_DIR="${ROOT_DIR}/triton"
TRITON_HASH="$(cat "${TRITON_RISCV_DIR}/triton-hash.txt")"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

TOOLCHAIN_DIR="${TRITON_RISCV_DIR}/.cache"
LLVM_EXTRACT_DIR="${TOOLCHAIN_DIR}/llvm"
BUDDY_EXTRACT_DIR="${TOOLCHAIN_DIR}/buddy"
LLVM_BINARY_DIR="${LLVM_EXTRACT_DIR}/bin"
BUDDY_MLIR_BINARY_DIR="${BUDDY_EXTRACT_DIR}/bin"

if [ ! -x "${PYTHON}" ]; then
    python3 -m venv "${ROOT_DIR}/.venv"
fi

BUDDY_RELEASE_TAG="$(tail -n 1 "${TRITON_RISCV_DIR}/buddy-hash.txt")"
BUDDY_PACKAGE_VERSION="${BUDDY_RELEASE_TAG#release/v}"
BUDDY_PACKAGE_VERSION="${BUDDY_PACKAGE_VERSION#nightly/v}"
BUDDY_PYTHON_TAG="cp312-abi3"

case "$(uname -m)" in
    amd64|x86_64)
        LLVM_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/${BUDDY_RELEASE_TAG}/llvm-24.0.0git-${BUDDY_PYTHON_TAG}-manylinux_2_28_x86_64.tar.gz"
        BUDDY_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/${BUDDY_RELEASE_TAG}/buddy-${BUDDY_PACKAGE_VERSION}-${BUDDY_PYTHON_TAG}-manylinux_2_28_x86_64.tar.gz"
        ;;
    riscv64)
        LLVM_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/${BUDDY_RELEASE_TAG}/llvm-24.0.0git-${BUDDY_PYTHON_TAG}-manylinux_2_39_riscv64.tar.gz"
        BUDDY_DEFAULT_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/${BUDDY_RELEASE_TAG}/buddy-${BUDDY_PACKAGE_VERSION}-${BUDDY_PYTHON_TAG}-manylinux_2_39_riscv64.tar.gz"
        ;;
    *)
        echo "unsupported arch: $(uname -m)" >&2
        exit 1
        ;;
esac

LLVM_URL="${TRITON_LLVM_PACKAGE_URL:-${LLVM_DEFAULT_URL}}"
BUDDY_URL="${TRITON_BUDDY_PACKAGE_URL:-${BUDDY_DEFAULT_URL}}"

if [ ! -d "${TRITON_DIR}" ]; then
    git clone https://github.com/triton-lang/triton.git "${TRITON_DIR}"
fi

git config --global --add safe.directory "${TRITON_RISCV_DIR}"
git config --global --add safe.directory "${TRITON_DIR}"
if ! git -C "${TRITON_DIR}" cat-file -e "${TRITON_HASH}^{commit}" 2>/dev/null; then
    git -C "${TRITON_DIR}" fetch origin
fi
if ! git -C "${TRITON_DIR}" cat-file -e "${TRITON_HASH}^{commit}" 2>/dev/null; then
    git -C "${TRITON_DIR}" fetch origin "${TRITON_HASH}"
fi
git -C "${TRITON_DIR}" reset --hard "${TRITON_HASH}"

"${TRITON_RISCV_DIR}/scripts/apply_patches.sh" "${TRITON_DIR}"

mkdir -p "${TOOLCHAIN_DIR}"

if [ ! -d "${LLVM_BINARY_DIR}" ]; then
    rm -rf "${LLVM_EXTRACT_DIR}"
    mkdir -p "${LLVM_EXTRACT_DIR}"
    curl -fL "${LLVM_URL}" -o "${TOOLCHAIN_DIR}/llvm.tar.gz"
    tar -xzf "${TOOLCHAIN_DIR}/llvm.tar.gz" -C "${LLVM_EXTRACT_DIR}"
    rm -f "${TOOLCHAIN_DIR}/llvm.tar.gz"
fi

if [ ! -d "${BUDDY_MLIR_BINARY_DIR}" ]; then
    rm -rf "${BUDDY_EXTRACT_DIR}"
    mkdir -p "${BUDDY_EXTRACT_DIR}"
    curl -fL "${BUDDY_URL}" -o "${TOOLCHAIN_DIR}/buddy.tar.gz"
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
echo "Installing Triton into ${PYTHON}..."

"${PYTHON}" -m pip install --upgrade pip
case "$(uname -m)" in
    riscv64)
        # Connection to https://ruyirepo.ruyicommunity.cn/pypi/simple/ is poor
        "${PYTHON}" -m pip install numpy "cmake>=3.20,<4.0" --index-url https://gitlab.com/api/v4/projects/56254198/packages/pypi/simple
        ;;
    *)
        "${PYTHON}" -m pip install numpy "cmake>=3.20,<4.0"
        ;;
esac

"${PYTHON}" -m pip install -U \
    "setuptools>=40.8.0" \
    "ninja>=1.11.1" \
    "pybind11>=2.13.1" \
    "pytest"
"${PYTHON}" -m pip install --no-build-isolation -e .
