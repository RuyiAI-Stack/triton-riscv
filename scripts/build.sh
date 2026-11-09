#!/usr/bin/env bash
#
# Build and install Triton with the triton-shared backend. Run scripts/setup.sh
# once first -- the Makefile's `all` target and `make setup` do that for you.
#
# Environment: see scripts/env.sh. Set PYTHON to build into another interpreter.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/env.sh
source "${ROOT_DIR}/scripts/env.sh"

PYTHON="${PYTHON:-${TRITON_VENV}/bin/python}"

# Build with clang, as the official Triton wheels do: clang tolerates the
# "deprecated aggregate builder" pattern MLIR's generated code uses, while gcc
# turns it into an error under Triton's -Werror. Select it via CC/CXX rather
# than TRITON_BUILD_WITH_CLANG_LLD, which would also force lld.
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

# Keep the plugin's POST_BUILD copy of triton-shared-opt out of an editable
# source tree; the tool is staged below instead.
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_WHEEL_DIR=${TRITON_DIR}/build/wheel-staging${TRITON_APPEND_CMAKE_ARGS:+ ${TRITON_APPEND_CMAKE_ARGS}}"

# TRITON_PLUGIN_DIRS (set by env.sh) is what makes Triton package backend/ as
# triton.backends.triton_shared, register its entry point, and build the C++
# plugin into libtriton.so.
"${PYTHON}" -m pip install --no-build-isolation -e "${TRITON_DIR}"

# Stage the freshly built tool where the backend looks for it.
_python_tag="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
mkdir -p "${TRITON_RISCV_DIR}/backend/bin"
ln -sfn ${TRITON_DIR}/build/cmake.linux-*-cpython-${_python_tag}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt \
    "${TRITON_RISCV_DIR}/backend/bin/triton-shared-opt"
unset _python_tag

# Do not `import triton` here: discovering nvidia/amd drivers while
# triton.backends is still initializing hits a circular import on some
# hosts. The plugin tool plus the package metadata are enough to prove
# the editable install.
"${PYTHON}" -c "import importlib.metadata as m; print(m.version('triton'))"
"${TRITON_RISCV_DIR}/backend/bin/triton-shared-opt" --version
