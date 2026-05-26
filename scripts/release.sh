#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <py_tag> <x86_64|riscv64>" >&2
    echo "example: $0 cp312-cp312 x86_64" >&2
    exit 1
fi

PY_TAG="$1"
TARGET_ARCH="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

if [ -z "${IN_MANYLINUX:-}" ]; then
    mkdir -p "${OUTPUT_DIR}"

    docker run --rm -i \
        --platform "${DOCKER_PLATFORM}" \
        -e IN_MANYLINUX=1 \
        -e PY_TAG="${PY_TAG}" \
        -e TARGET_ARCH="${TARGET_ARCH}" \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        -e HOME=/workspace \
        -v "${REPO_ROOT}:${CONTAINER_PLUGIN_DIR}" \
        -w "${CONTAINER_PLUGIN_DIR}" \
        "${MANYLINUX_IMAGE}" \
        /bin/bash ./scripts/release.sh "${PY_TAG}" "${TARGET_ARCH}"

    echo "wheels are in ${OUTPUT_DIR}"
    exit 0
fi

PYBIN="/opt/python/${PY_TAG}/bin/python"
export PATH="/opt/python/${PY_TAG}/bin:$PATH"

if [ -f /opt/rh/gcc-toolset-14/enable ]; then
    source /opt/rh/gcc-toolset-14/enable
fi

dnf install -y clang lld git
"${PYBIN}" -m pip install --upgrade pip
case "$(uname -m)" in
    riscv64)
        # Connection to https://ruyirepo.ruyicommunity.cn/pypi/simple/ is poor
        "${PYBIN}" -m pip install numpy "cmake>=3.20,<4.0" --index-url https://gitlab.com/api/v4/projects/56254198/packages/pypi/simple
        ;;
    *)
        "${PYBIN}" -m pip install numpy "cmake>=3.20,<4.0"
        ;;
esac

export TRITON_PYTHON_TAG="${PY_TAG}"
export TRITON_PLUGIN_DIRS="${CONTAINER_PLUGIN_DIR}"
export TRITON_BUILD_WITH_CLANG_LLD=1

rm -rf "${CONTAINER_OUTPUT_DIR}" "${CONTAINER_TRITON_DIR}/build"
"${CONTAINER_PLUGIN_DIR}/scripts/build.sh"
rm -rf "${CONTAINER_OUTPUT_DIR}" "${CONTAINER_TRITON_DIR}/build"
mkdir -p "${CONTAINER_OUTPUT_DIR}"
RAW_WHEEL_DIR="$(mktemp -d)"

cd "${CONTAINER_TRITON_DIR}"
"${PYBIN}" setup.py bdist_wheel -d "${RAW_WHEEL_DIR}"
auditwheel repair "${RAW_WHEEL_DIR}"/*.whl -w "${CONTAINER_OUTPUT_DIR}"
rm -rf "${RAW_WHEEL_DIR}"
chown -R "${HOST_UID}:${HOST_GID}" "${CONTAINER_OUTPUT_DIR}" || true
