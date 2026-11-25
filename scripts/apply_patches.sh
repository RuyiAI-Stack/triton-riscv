#!/usr/bin/env bash
#
# Apply patches/*.patch to the pinned Triton checkout. scripts/setup.sh runs it
# right after resetting that checkout to triton-hash.txt, so every build -- local
# or CI -- sees the same sources.
#
# Keep this set tiny: a patch here only bridges an LLVM API difference between
# the Triton we pin and the LLVM the buddy release ships. Anything that changes
# behaviour belongs upstream instead.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_DIR="${TRITON_DIR:-${ROOT_DIR}/triton}"

shopt -s nullglob
for patch in "${ROOT_DIR}"/patches/*.patch; do
    if git -C "${TRITON_DIR}" apply --check "${patch}" 2>/dev/null; then
        git -C "${TRITON_DIR}" apply "${patch}"
        echo "applied $(basename "${patch}")"
    elif git -C "${TRITON_DIR}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "already applied $(basename "${patch}")"
    else
        # Applying it is what makes the build work, so never skip it silently.
        echo "error: $(basename "${patch}") does not apply to $(git -C "${TRITON_DIR}" rev-parse --short HEAD); regenerate it" >&2
        exit 1
    fi
done
