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

if [[ ! -x "${TRITON_VENV}/bin/python" ]]; then
  echo "python not found in TRITON_VENV: ${TRITON_VENV}" >&2
  echo "Activate your environment (venv or conda) and install build deps first." >&2
  exit 1
fi

if [[ ! -d "${LLVM_SYSPATH}" ]]; then
  echo "LLVM_SYSPATH does not exist: ${LLVM_SYSPATH}" >&2
  echo "Build buddy-mlir LLVM first, or override LLVM_SYSPATH/BUDDY_DIR." >&2
  exit 1
fi

if [[ -n "${BUILD_DIR:-}" ]]; then
  # Custom build directory names need not contain an interpreter tag. Let
  # CMake validate their configuration; never delete an explicitly chosen tree.
  if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    _cached_build_root="$(sed -n 's|^# For build in directory: ||p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
    if [[ -n "${_cached_build_root}" && "${_cached_build_root}" != "${BUILD_DIR}" ]]; then
      echo "Moved CMake cache in ${BUILD_DIR}; select a fresh TRITON_BUILD_DIR." >&2
      exit 1
    fi
  fi
  export TRITON_BUILD_DIR="${BUILD_DIR}"
fi

cd "${TRITON_DIR}"

# setuptools/wheel may reuse stale Python staging directories under build/.
# If an earlier build accidentally staged experiment dumps under
# triton/backends/triton_shared, bdist_wheel will copy them into the new wheel
# even after the source backend no longer contains those files.
if [[ -d build ]]; then
  find build -maxdepth 1 -type d \( -name 'lib.*' -o -name 'bdist.*' \) -print -exec rm -rf {} +
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 "${TRITON_VENV}/bin/python" -m pip install --no-build-isolation -vvv .

"${TRITON_VENV}/bin/python" - "${TRITON_RISCV_DIR}" <<'PY'
import pathlib
import shutil
import sys

import triton.backends.triton_shared.compiler as compiler

repo_root = pathlib.Path(sys.argv[1]).resolve()
src_dir = repo_root / "backend"
dst_dir = pathlib.Path(compiler.__file__).resolve().parent

for name in ("compiler.py", "driver.py", "riscv.py", "paths.py", "contract.py"):
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
    for pattern in ("compiler*.pyc", "driver*.pyc", "riscv*.pyc", "paths*.pyc", "contract*.pyc"):
        for pyc in pycache.glob(pattern):
            pyc.unlink()

print(f"Synced triton_shared backend Python files to {dst_dir}")
PY

if [[ -z "${BUILD_DIR:-}" ]]; then
  # Ask the Python build helper for this interpreter's directory. Never select
  # an older build by sorting directory names.
  BUILD_DIR="$(PYTHONPATH="${TRITON_DIR}/python" "${TRITON_VENV}/bin/python" -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')"
  export BUILD_DIR
fi
export TRITON_SHARED_OPT_PATH="${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"

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
unset _triton_riscv_repo_root
unset _triton_riscv_script_dir
