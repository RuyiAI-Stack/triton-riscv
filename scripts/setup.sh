#!/usr/bin/env bash
#
# Bootstrap a triton-riscv checkout: the pinned Triton checkout and the LLVM and
# buddy toolchains that scripts/build.sh links against. Safe to re-run; everything
# already in place is reused.
#
# Environment: TRITON_DIR and LLVM_SYSPATH override where the checkout and the
# LLVM toolchain land. The build dependencies are in requirements.txt; install
# them into the interpreter you build with.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_DIR="${TRITON_DIR:-${ROOT_DIR}/triton}"
LLVM_SYSPATH="${LLVM_SYSPATH:-${ROOT_DIR}/.cache/llvm}"

TRITON_HASH="$(<"${ROOT_DIR}/triton-hash.txt")"
BUDDY_PYTHON_TAG="cp312-abi3"
BUDDY_RELEASE_TAG="$(tail -n 1 "${ROOT_DIR}/buddy-hash.txt")"
BUDDY_RELEASE_VERSION="${BUDDY_RELEASE_TAG#release/v}"
BUDDY_RELEASE_VERSION="${BUDDY_RELEASE_VERSION#nightly/v}"
BUDDY_RELEASE_URL="https://github.com/buddy-compiler/buddy-mlir/releases/download/${BUDDY_RELEASE_TAG}"

case "$(uname -m)" in
    x86_64|amd64)
        BUDDY_PLATFORM="manylinux_2_28_x86_64"
        ;;
    riscv64)
        BUDDY_PLATFORM="manylinux_2_39_riscv64"
        ;;
    *)
        echo "unsupported arch: $(uname -m)" >&2
        exit 1
        ;;
esac

# Unpack a buddy-mlir release into <dir>, mirroring the <dir>/bin layout the
# tooling expects. Skips the download when already unpacked.
download_release() {
    local name="$1"
    local url="$2"
    local dir="$3"

    if [ -d "${dir}/bin" ]; then
        return
    fi
    echo "Downloading ${name} from ${url}..."
    rm -rf "${dir}"
    mkdir -p "${dir}"
    curl -fL "${url}" | tar -xzf - -C "${dir}"
}

# Triton links the NVPTX/AMDGPU codegen libraries and lld unconditionally, and
# the buddy-mlir release provides them.
download_release llvm \
    "${BUDDY_RELEASE_URL}/llvm-24.0.0git-${BUDDY_PYTHON_TAG}-${BUDDY_PLATFORM}.tar.gz" \
    "${LLVM_SYSPATH}"
download_release buddy \
    "${BUDDY_RELEASE_URL}/buddy-${BUDDY_RELEASE_VERSION}-${BUDDY_PYTHON_TAG}-${BUDDY_PLATFORM}.tar.gz" \
    "${ROOT_DIR}/.cache/buddy"

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
git config --global --add safe.directory "${ROOT_DIR}"
git config --global --add safe.directory "${TRITON_DIR}"
if ! git -C "${TRITON_DIR}" cat-file -e "${TRITON_HASH}^{commit}" 2>/dev/null; then
    git "${GIT_AUTH_ARGS[@]}" -C "${TRITON_DIR}" fetch origin "${TRITON_HASH}"
fi
git -C "${TRITON_DIR}" reset --hard "${TRITON_HASH}"
# The AMD codegen library. Upstream either downloads a prebuilt copy or builds one
# against a separately pinned LLVM, and there is no riscv64 build of either; the
# library itself is small and builds against the LLVM already here, in seconds.
# Without it the AMD backend refuses to compile anything. scripts/_gpu-stub-env.sh
# points TRITON_AMD_CODEGEN_PATH at exactly this path.
amd_codegen_lib="${TRITON_DIR}/third_party/amd/backend/lib/libtriton_amd_codegen.so"
llvm_cmake="${LLVM_SYSPATH}/lib/cmake/llvm"
if [ ! -e "${amd_codegen_lib}" ] && [ -f "${llvm_cmake}/LLVMConfig.cmake" ]; then
    echo "Building the AMD codegen library..."
    cmake -G Ninja -S "${TRITON_DIR}/third_party/amd/backend/codegen" \
        -B "${ROOT_DIR}/.cache/amd-codegen" \
        -DLLVM_DIR="${llvm_cmake}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRITON_AMD_LLVM_REVISION=local >/dev/null
    cmake --build "${ROOT_DIR}/.cache/amd-codegen" --target triton_amd_codegen >/dev/null
    cp "${ROOT_DIR}/.cache/amd-codegen/libtriton_amd_codegen.so" "${amd_codegen_lib}"
fi
