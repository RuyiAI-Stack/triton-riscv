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

# Keep the token out of the repository's remote URL while authenticating CI
# clones and fetches. Without a token, Git remains usable for local builds.
GIT_AUTH_ARGS=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    GIT_AUTH_HEADER="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')"
    GIT_AUTH_ARGS=(-c "http.extraHeader=AUTHORIZATION: basic ${GIT_AUTH_HEADER}")
fi

if [ ! -d "${TRITON_DIR}" ]; then
    git "${GIT_AUTH_ARGS[@]}" clone https://github.com/triton-lang/triton.git "${TRITON_DIR}"
fi

git config --global --add safe.directory "${TRITON_RISCV_DIR}"
git config --global --add safe.directory "${TRITON_DIR}"
if ! git -C "${TRITON_DIR}" cat-file -e "${TRITON_HASH}^{commit}" 2>/dev/null; then
    git "${GIT_AUTH_ARGS[@]}" -C "${TRITON_DIR}" fetch origin "${TRITON_HASH}"
fi
git -C "${TRITON_DIR}" reset --hard "${TRITON_HASH}"

"${TRITON_RISCV_DIR}/scripts/apply_patches.sh" "${TRITON_DIR}"

mkdir -p "${TOOLCHAIN_DIR}"

download_package() {
    local binary_dir="$1"
    local extract_dir="$2"
    local url="$3"
    local archive="$4"

    if [ ! -d "${binary_dir}" ]; then
        rm -rf "${extract_dir}"
        mkdir -p "${extract_dir}"
        curl -fL "${url}" -o "${TOOLCHAIN_DIR}/${archive}"
        tar -xzf "${TOOLCHAIN_DIR}/${archive}" -C "${extract_dir}"
        rm -f "${TOOLCHAIN_DIR}/${archive}"
    fi
}

download_package "${LLVM_BINARY_DIR}" "${LLVM_EXTRACT_DIR}" "${LLVM_URL}" llvm.tar.gz
download_package "${BUDDY_MLIR_BINARY_DIR}" "${BUDDY_EXTRACT_DIR}" "${BUDDY_URL}" buddy.tar.gz

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
    "nanobind==2.10.2" \
    "pytest"
"${PYTHON}" -m pip install --no-build-isolation -e .
