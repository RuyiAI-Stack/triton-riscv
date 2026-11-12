#!/usr/bin/env bash
#
# Build and install Triton with the triton-shared backend. Run scripts/setup.sh
# once first -- the Makefile's `all` target and `make setup` do that for you.
#
# Environment: scripts/env.sh, plus the GPU dependency stubs sourced first.
# Set PYTHON to build into another interpreter.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_DIR="${TRITON_DIR:-${ROOT_DIR}/triton}"

# Stub the GPU toolchain downloads; env.sh runs after and repoints the AMD codegen.
source "${ROOT_DIR}/scripts/_gpu-stub-env.sh" "${ROOT_DIR}/.cache/gpu-deps-stub"
source "${ROOT_DIR}/scripts/env.sh"

PYTHON="${PYTHON:-python3}"
# Triton resolves lit with shutil.which('lit'), so the interpreter we install
# into has to come first on PATH: it picks the pip lit instead of the one in the
# LLVM tree, which calls main() at module level and crashes under Python 3.14,
# where Linux defaults to the forkserver start method.
export PATH="$(dirname "$(command -v "${PYTHON}")"):${PATH}"

amd_codegen_dir="${TRITON_DIR}/third_party/amd/backend/codegen"
amd_codegen_lib="${TRITON_DIR}/third_party/amd/backend/lib/libtriton_amd_codegen.so"
llvm_cmake="${LLVM_SYSPATH}/lib/cmake/llvm"
if [ -f "${llvm_cmake}/LLVMConfig.cmake" ]; then
    cmake -G Ninja -S "${amd_codegen_dir}" -B "${amd_codegen_dir}/build" \
        -DLLVM_DIR="${llvm_cmake}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRITON_AMD_LLVM_REVISION=local
    cmake --build "${amd_codegen_dir}/build" --target triton_amd_codegen >/dev/null
    cp "${amd_codegen_dir}/build/libtriton_amd_codegen.so" "${amd_codegen_lib}"
else
    echo "warning: ${llvm_cmake}/LLVMConfig.cmake not found; skipping the AMD codegen library" >&2
fi

# Build against the Python stable ABI (abi3), which is how upstream builds its
# abi3 wheel: nanobind is compiled with the limited API, and the wheel is tagged
# cp312-abi3, so it covers Python 3.12 and later instead of only the interpreter
# it was built with. Upstream ships no abi3 wheel for free-threaded builds.
export TRITON_STABLE_ABI="${TRITON_STABLE_ABI:-ON}"

# Build with clang, as the official Triton wheels do: clang tolerates the
# "deprecated aggregate builder" pattern MLIR's generated code uses, while gcc
# turns it into an error under Triton's -Werror. Select it via CC/CXX rather
# than TRITON_BUILD_WITH_CLANG_LLD, which would also force lld.
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

# Keep the plugin's POST_BUILD copy of triton-shared-opt out of an editable
# source tree; the tool is staged below instead.
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_WHEEL_DIR=${TRITON_DIR}/build/wheel-staging -DTRITON_BUILD_UT=OFF${TRITON_APPEND_CMAKE_ARGS:+ ${TRITON_APPEND_CMAKE_ARGS}}"

# Packages backend/ as triton.backends.triton_shared; cmake caches it, so only the build needs it.
export TRITON_PLUGIN_DIRS="${ROOT_DIR}"
"${PYTHON}" -m pip install --no-build-isolation -e "${TRITON_DIR}"

# Stage the freshly built tool where the backend looks for it.
_python_tag="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
mkdir -p "${ROOT_DIR}/backend/bin"
ln -sfn ${TRITON_DIR}/build/cmake.linux-*-cpython-${_python_tag}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt \
    "${ROOT_DIR}/backend/bin/triton-shared-opt"
unset _python_tag

# The LLVM and buddy tools the backend shells out to, so the wheel can carry them.
for tool in opt llc llvm-objdump llvm-readobj; do
    ln -sfn "${LLVM_SYSPATH}/bin/${tool}" "${ROOT_DIR}/backend/bin/${tool}"
done
buddy_bin="${BUDDY_MLIR_BINARY_DIR:-${ROOT_DIR}/.cache/buddy/bin}"
[ -d "${buddy_bin}" ] || buddy_bin="$(dirname "$(command -v buddy-opt 2>/dev/null || echo /nonexistent)")"
for tool in buddy-opt buddy-llc buddy-translate; do
    ln -sfn "${buddy_bin}/${tool}" "${ROOT_DIR}/backend/bin/${tool}"
done
