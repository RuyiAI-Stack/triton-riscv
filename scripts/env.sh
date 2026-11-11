#!/usr/bin/env bash
#
# Development environment for a triton-riscv source checkout. Source it, do not
# execute it:
#
#   source scripts/env.sh
#
# Export any of the values below beforehand to override them.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/env.sh" >&2
  exit 1
fi

_TRITON_RISCV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The LLVM build scripts/setup.sh downloads, which the Triton build links against.
export LLVM_SYSPATH="${LLVM_SYSPATH:-${_TRITON_RISCV_DIR}/.cache/llvm}"

# Build time
# ---------------------------------------------------------------------------

# Only when build.sh has built it, so a fresh checkout does not shadow the packaged copy.
amd_codegen="${_TRITON_RISCV_DIR}/triton/third_party/amd/backend/lib/libtriton_amd_codegen.so"
if [ -f "${amd_codegen}" ]; then
    export TRITON_AMD_CODEGEN_PATH="${amd_codegen}"
fi
unset amd_codegen

# The build clones things (triton, and googletest through its cmake); authenticate
# git with the token when CI provides one, the way actions/checkout does.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    _gitauth="$(mktemp)"
    printf '[http "https://github.com/"]\n\textraheader = AUTHORIZATION: basic %s\n' \
        "$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')" > "${_gitauth}"
    export GIT_CONFIG_GLOBAL="${_gitauth}"
    unset _gitauth
fi

# Both toolchains are found on PATH, the same way an installed wheel finds them.
export PATH="${LLVM_SYSPATH}/bin:${_TRITON_RISCV_DIR}/.cache/buddy/bin:${PATH}"
# Keep the repository's own triton/ checkout from shadowing the installed one.
export PYTHONSAFEPATH=1
