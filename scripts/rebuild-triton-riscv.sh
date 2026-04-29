#!/usr/bin/env bash

set -euo pipefail

_triton_riscv_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_triton_riscv_repo_root="$(cd "${_triton_riscv_script_dir}/.." && pwd)"

source "${_triton_riscv_repo_root}/scripts/triton-riscv-env.sh"

if [[ ! -d "${TRITON_DIR}" ]]; then
  echo "TRITON_DIR does not exist: ${TRITON_DIR}" >&2
  echo "Set TRITON_DIR before running this script." >&2
  exit 1
fi

if [[ ! -x "${TRITON_VENV}/bin/pip" ]]; then
  echo "pip not found in TRITON_VENV: ${TRITON_VENV}" >&2
  echo "Create the virtual environment and install build deps first." >&2
  exit 1
fi

if [[ ! -d "${LLVM_SYSPATH}" ]]; then
  echo "LLVM_SYSPATH does not exist: ${LLVM_SYSPATH}" >&2
  echo "Build buddy-mlir LLVM first, or override LLVM_SYSPATH/BUDDY_DIR." >&2
  exit 1
fi

_expected_python_tag="$("${TRITON_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [[ -n "${BUILD_DIR:-}" && -d "${BUILD_DIR}" ]]; then
  _reconfigure_build_dir=0

  if [[ "${BUILD_DIR}" != *"cpython-${_expected_python_tag}" ]]; then
    _reconfigure_build_dir=1
  fi

  if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    _cached_build_root="$(sed -n 's|^# For build in directory: ||p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
    if [[ -n "${_cached_build_root}" && "${_cached_build_root}" != "${BUILD_DIR}" ]]; then
      _reconfigure_build_dir=1
    fi
  fi

  if [[ "${_reconfigure_build_dir}" -eq 1 ]]; then
    echo "Removing stale Triton build directory: ${BUILD_DIR}" >&2
    rm -rf "${BUILD_DIR}"
    unset BUILD_DIR
    unset TRITON_SHARED_OPT_PATH
  fi
fi

if [[ -n "${BUILD_DIR:-}" && -d "${BUILD_DIR}" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

cd "${TRITON_DIR}"

# setuptools/wheel may reuse stale Python staging directories under build/.
# If an earlier build accidentally staged experiment dumps under
# triton/backends/triton_shared, bdist_wheel will copy them into the new wheel
# even after the source backend no longer contains those files.
if [[ -d build ]]; then
  find build -maxdepth 1 -type d \( -name 'lib.*' -o -name 'bdist.*' \) -print -exec rm -rf {} +
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 "${TRITON_VENV}/bin/pip" install --no-build-isolation -vvv .

"${TRITON_VENV}/bin/python" - "${TRITON_RISCV_DIR}" <<'PY'
import pathlib
import shutil
import sys

import triton.backends.triton_shared.compiler as compiler

repo_root = pathlib.Path(sys.argv[1]).resolve()
src_dir = repo_root / "backend"
dst_dir = pathlib.Path(compiler.__file__).resolve().parent

for name in ("compiler.py", "driver.py"):
    src = src_dir / name
    dst = dst_dir / name
    if not src.is_file():
        raise SystemExit(f"Missing backend source file: {src}")
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    if src.read_bytes() != dst.read_bytes():
        raise SystemExit(f"Installed backend file does not match source: {dst}")

pycache = dst_dir / "__pycache__"
if pycache.is_dir():
    for pattern in ("compiler*.pyc", "driver*.pyc"):
        for pyc in pycache.glob(pattern):
            pyc.unlink()

print(f"Synced triton_shared backend Python files to {dst_dir}")
PY

if [[ -z "${BUILD_DIR:-}" || ! -d "${BUILD_DIR}" || ! -x "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  BUILD_DIR="$(
    find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name "cmake.linux-*-cpython-${_expected_python_tag}" | sort | head -n1
  )"
  if [[ -z "${BUILD_DIR:-}" ]]; then
    BUILD_DIR="$(
      find "${TRITON_DIR}/build" -maxdepth 1 -mindepth 1 -type d -name 'cmake.linux-*-cpython-*' | sort | head -n1
    )"
  fi
  export BUILD_DIR
  export TRITON_SHARED_OPT_PATH="${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
fi

if [[ -z "${BUILD_DIR:-}" || ! -d "${BUILD_DIR}" ]]; then
  echo "Unable to locate Triton build directory under ${TRITON_DIR}/build" >&2
  exit 1
fi

if [[ ! -x "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  echo "triton-shared-opt not found or not executable: ${TRITON_SHARED_OPT_PATH:-}" >&2
  exit 1
fi

"${TRITON_VENV}/bin/python" -c "import triton; import triton.backends.triton_shared.compiler as c; print(triton.__version__); print(c.__file__)"
"${TRITON_SHARED_OPT_PATH}" --version

unset _cached_build_root
unset _expected_python_tag
unset _reconfigure_build_dir
unset _triton_riscv_repo_root
unset _triton_riscv_script_dir
