#!/usr/bin/env bash
# Apply triton-shared patches to the triton main repository.
#
# Usage:
#   ./scripts/apply_patches.sh [TRITON_DIR]
#
# If TRITON_DIR is not specified, the script looks for a sibling directory
# named "triton" next to the triton-shared repo root.
#
# The patches remove the hard-coded NVIDIA/AMD backend assumptions from
# triton's build system so that triton can be built as a pure frontend
# (AST -> TTIR) with triton-shared acting as the only backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCHES_DIR="${REPO_ROOT}/patches"

# Resolve the triton source directory.
if [ $# -ge 1 ]; then
    TRITON_DIR="$(realpath "$1")"
else
    TRITON_DIR="$(realpath "${REPO_ROOT}/../triton")"
fi

if [ ! -d "${TRITON_DIR}" ]; then
    echo "ERROR: triton directory not found at '${TRITON_DIR}'"
    echo "  Pass the path explicitly:  $0 /path/to/triton"
    exit 1
fi

echo "Applying triton-shared patches to: ${TRITON_DIR}"

# Later patches can change the context of earlier patches. Unwind the stack
# in a scratch directory to recognize those earlier patches without touching
# the user's checkout, then validate the complete stack before applying it.
SCRATCH_DIR="$(mktemp -d)"
trap 'rm -rf "${SCRATCH_DIR}"' EXIT
git -C "${SCRATCH_DIR}" init --quiet
patches=("${PATCHES_DIR}"/*.patch)
for patch in "${patches[@]}"; do
    while IFS=$'\t' read -r added removed path; do
        if [[ -e "${TRITON_DIR}/${path}" && ! -e "${SCRATCH_DIR}/${path}" ]]; then
            mkdir -p "${SCRATCH_DIR}/$(dirname "${path}")"
            cp "${TRITON_DIR}/${path}" "${SCRATCH_DIR}/${path}"
        fi
    done < <(git -C "${TRITON_DIR}" apply --numstat "${patch}")
done

declare -A applied
for ((i=${#patches[@]}-1; i>=0; i--)); do
    patch="${patches[i]}"
    if git -C "${SCRATCH_DIR}" apply --check --reverse "${patch}" 2>/dev/null; then
        git -C "${SCRATCH_DIR}" apply --reverse "${patch}"
        applied["${patch}"]=1
    fi
done

for patch in "${patches[@]}"; do
    if ! git -C "${SCRATCH_DIR}" apply --check "${patch}"; then
        echo "ERROR: $(basename "${patch}") conflicts with the Triton checkout." >&2
        exit 1
    fi
    git -C "${SCRATCH_DIR}" apply "${patch}"
done

for patch in "${PATCHES_DIR}"/*.patch; do
    echo ""
    echo "  Applying $(basename "${patch}") ..."
    if [[ "${applied["${patch}"]:-0}" == 1 ]]; then
        echo "  SKIPPED (patch already applied)"
    else
        git -C "${TRITON_DIR}" apply "${patch}"
        echo "  OK"
    fi
done

echo ""
echo "All patches applied successfully."
echo ""
echo "Next steps:"
echo "  export TRITON_PLUGIN_DIRS=${REPO_ROOT}"
echo "  pip install --no-build-isolation -e ${TRITON_DIR}"
