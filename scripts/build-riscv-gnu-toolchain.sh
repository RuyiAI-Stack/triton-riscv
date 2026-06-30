#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_script_dir}/.." && pwd)"
BUDDY_DIR="${BUDDY_DIR:-$(cd "${_repo_root}/../buddy-mlir" && pwd)}"
_toolchain_src="${BUDDY_DIR}/thirdparty/riscv-gnu-toolchain"
_toolchain_prefix="${RISCV_GNU_TOOLCHAIN_DIR:-${BUDDY_DIR}/build/thirdparty/riscv-gnu-toolchain}"
_jobs="${JOBS:-16}"

if [[ ! -d "${BUDDY_DIR}/.git" ]]; then
  echo "Buddy source tree not found: ${BUDDY_DIR}" >&2
  exit 1
fi

git -C "${BUDDY_DIR}" submodule update --init thirdparty/riscv-gnu-toolchain
git -C "${_toolchain_src}" submodule update --init binutils gcc glibc qemu

cd "${_toolchain_src}"
if [[ ! -f Makefile ]]; then
  ./configure \
    --prefix="${_toolchain_prefix}" \
    --enable-linux \
    --disable-gdb \
    --with-arch=rv64gcv_zfh_zvfh_zba_zbb \
    --with-abi=lp64d
fi

env -u PYTHONSAFEPATH make -j"${_jobs}" linux
env -u PYTHONSAFEPATH make -j"${_jobs}" build-qemu

"${_toolchain_prefix}/bin/riscv64-unknown-linux-gnu-gcc" --version | head -n1
"${_toolchain_prefix}/bin/qemu-riscv64" --version | head -n1
