#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <py_tag> <x86_64|riscv64>" >&2
    echo "example: $0 cp312-cp312 x86_64" >&2
    echo "         $0 cp312 x86_64" >&2
    exit 1
fi

PY_TAG="$1"
TARGET_ARCH="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${PY_TAG}" =~ ^cp[0-9]+$ ]]; then
    PY_TAG="${PY_TAG}-${PY_TAG}"
fi

OUTPUT_DIR="${REPO_ROOT}/build/release/${TARGET_ARCH}/${PY_TAG}"
CONTAINER_PLUGIN_DIR=/workspace/triton-riscv
CONTAINER_TRITON_DIR="${CONTAINER_PLUGIN_DIR}/triton"
CONTAINER_OUTPUT_DIR="${CONTAINER_PLUGIN_DIR}/build/release/${TARGET_ARCH}/${PY_TAG}"

case "${TARGET_ARCH}" in
    x86_64)
        MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64:latest"
        DOCKER_PLATFORM="linux/amd64"
        ;;
    riscv64)
        MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_39_riscv64:latest"
        DOCKER_PLATFORM="linux/riscv64"
        ;;
    *)
        echo "unsupported target arch: ${TARGET_ARCH}" >&2
        exit 1
        ;;
esac

if [ -z "${IN_DOCKER:-}" ]; then
    mkdir -p "${OUTPUT_DIR}"

    DOCKER_RUN_ARGS=(run --rm -i)

    HOST_ARCH_RAW="$(uname -m)"
    case "${HOST_ARCH_RAW}" in
        amd64)
            HOST_ARCH="x86_64"
            ;;
        *)
            HOST_ARCH="${HOST_ARCH_RAW}"
            ;;
    esac

    # Support cross-compile
    if [ "${HOST_ARCH}" != "${TARGET_ARCH}" ]; then
        DOCKER_RUN_ARGS+=(--platform "${DOCKER_PLATFORM}")
    fi

    DOCKER_ENV_ARGS=()
    for proxy_var in http_proxy https_proxy ftp_proxy no_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY ALL_PROXY all_proxy; do
        if [ -n "${!proxy_var:-}" ]; then
            DOCKER_ENV_ARGS+=(-e "${proxy_var}=${!proxy_var}")
        fi
    done

    docker "${DOCKER_RUN_ARGS[@]}" \
        "${DOCKER_ENV_ARGS[@]}" \
        -e IN_DOCKER=1 \
        -e PY_TAG="${PY_TAG}" \
        -e TARGET_ARCH="${TARGET_ARCH}" \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        -e HOME=/workspace \
        -v "${REPO_ROOT}:${CONTAINER_PLUGIN_DIR}" \
        -v /tmp/triton_riscv_docker_dnf_cache:/var/cache/dnf \
        -v /tmp/triton_riscv_docker_pip_cache:/workspace/.cache/pip \
        -w "${CONTAINER_PLUGIN_DIR}" \
        "${MANYLINUX_IMAGE}" \
        /bin/bash ./scripts/release.sh "${PY_TAG}" "${TARGET_ARCH}"

    echo "wheels are in ${OUTPUT_DIR}"
    exit 0
fi

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [ -n "${RAW_WHEEL_DIR:-}" ]; then
        rm -rf "${RAW_WHEEL_DIR}"
    fi
    if [ -e "${CONTAINER_OUTPUT_DIR}" ]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${CONTAINER_OUTPUT_DIR}"
    fi
    if [ -e "${CONTAINER_TRITON_DIR}/build" ]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${CONTAINER_TRITON_DIR}/build"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

export PYTHON="/opt/python/${PY_TAG}/bin/python"
export PATH="/opt/python/${PY_TAG}/bin:$PATH"

if [ -f /opt/rh/gcc-toolset-14/enable ]; then
    source /opt/rh/gcc-toolset-14/enable
fi

dnf install -y clang lld git

export TRITON_PYTHON_TAG="${PY_TAG}"
export TRITON_PLUGIN_DIRS="${CONTAINER_PLUGIN_DIR}"
export TRITON_BUILD_WITH_CLANG_LLD=1
export LLVM_BINARY_DIR="${CONTAINER_PLUGIN_DIR}/.cache/llvm/bin"
export LLVM_SYSPATH="${CONTAINER_PLUGIN_DIR}/.cache/llvm"
export BUDDY_MLIR_BINARY_DIR="${CONTAINER_PLUGIN_DIR}/.cache/buddy/bin"

# scripts/build.sh does an editable install to verify the patched Triton tree.
rm -rf "${CONTAINER_TRITON_DIR}/build"
"${CONTAINER_PLUGIN_DIR}/scripts/build.sh"

# Build the release wheel from a fresh CMake tree and empty output directory.
rm -rf "${CONTAINER_OUTPUT_DIR}" "${CONTAINER_TRITON_DIR}/build"
mkdir -p "${CONTAINER_OUTPUT_DIR}"
RAW_WHEEL_DIR=""
RAW_WHEEL_DIR="$(mktemp -d)"

cd "${CONTAINER_TRITON_DIR}"
"${PYTHON}" setup.py bdist_wheel -d "${RAW_WHEEL_DIR}"
auditwheel repair "${RAW_WHEEL_DIR}"/*.whl -w "${CONTAINER_OUTPUT_DIR}"
rm -rf "${RAW_WHEEL_DIR}"
