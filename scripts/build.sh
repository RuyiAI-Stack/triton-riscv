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

# Upstream Triton downloads host-specific GPU toolchains that riscv64 has no
# builds for, and builds Proton and the NVIDIA sanitizer runtime, none of which
# this backend needs. Stub them before env.sh, so that whatever env.sh points at
# a real file -- the AMD codegen library -- wins over the stub.
source "${ROOT_DIR}/scripts/_gpu-stub-env.sh" "${ROOT_DIR}/.cache/gpu-deps-stub"
source "${ROOT_DIR}/scripts/env.sh"

PYTHON="${PYTHON:-python3}"
# Triton resolves lit with shutil.which('lit'), so the interpreter we install
# into has to come first on PATH: it picks the pip lit instead of the one in the
# LLVM tree, which calls main() at module level and crashes under Python 3.14,
# where Linux defaults to the forkserver start method.
export PATH="$(dirname "$(command -v "${PYTHON}")"):${PATH}"

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
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_WHEEL_DIR=${TRITON_DIR}/build/wheel-staging${TRITON_APPEND_CMAKE_ARGS:+ ${TRITON_APPEND_CMAKE_ARGS}}"

# Triton's plugin support, which packages backend/ as triton.backends.triton_shared
# and builds the C++ plugin into libtriton.so. setup.py turns this into
# -DTRITON_PLUGIN_DIRS, which cmake caches, so only the build needs it.
export TRITON_PLUGIN_DIRS="${ROOT_DIR}"
"${PYTHON}" -m pip install --no-build-isolation -e "${TRITON_DIR}"

# Stage the freshly built tool where the backend looks for it.
_python_tag="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
mkdir -p "${ROOT_DIR}/backend/bin"
ln -sfn ${TRITON_DIR}/build/cmake.linux-*-cpython-${_python_tag}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt \
    "${ROOT_DIR}/backend/bin/triton-shared-opt"
unset _python_tag
