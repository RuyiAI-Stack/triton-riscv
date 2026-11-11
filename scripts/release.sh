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
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        DOCKER_ENV_ARGS+=(-e GITHUB_TOKEN)
    fi
    # Forwarded only when set: gives a nightly build a unique wheel version.
    if [ -n "${TRITON_WHEEL_VERSION_SUFFIX:-}" ]; then
        DOCKER_ENV_ARGS+=(-e TRITON_WHEEL_VERSION_SUFFIX)
    fi

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

dnf install -y clang lld git libstdc++-devel

# clang needs a GCC installation to find crtbeginS.o, libgcc and libstdc++ when
# it links, and on this image it does not detect the riscv64 layout by itself --
# the whole link is handed to ld with bare -lstdc++ / -lgcc, which fails. Ask gcc
# where it lives and pass that on explicitly. Images that ship the compiler as an
# SCL toolset under /opt/rh (which is also where clang's dependency chain used to
# land it) are enabled first, so `gcc` below is the right one either way.
toolset_root=""
for enable in $(ls -1 /opt/rh/gcc-toolset-*/enable 2>/dev/null | sort -V); do
    source "${enable}"
    toolset_root="$(dirname "${enable}")/root/usr"
done
gcc_install_dir="$(dirname "$(gcc -print-libgcc-file-name 2>/dev/null)" 2>/dev/null || true)"
if [ -n "${gcc_install_dir}" ] && [ -d "${gcc_install_dir}" ]; then
    export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DCMAKE_CXX_FLAGS=--gcc-install-dir=${gcc_install_dir} -DCMAKE_C_FLAGS=--gcc-install-dir=${gcc_install_dir}"
fi
ls -d /usr/lib/gcc/*/* 2>/dev/null | sed 's/^/[release] gccdir /'
echo "[release] toolset=${toolset_root:-<none>} gcc-install-dir=${gcc_install_dir:-<none>} g++=$(command -v g++ || echo none)"

export TRITON_PYTHON_TAG="${PY_TAG}"
# Same stable-ABI build as an ordinary `make build`, so the released wheel is
# tagged cp312-abi3 and covers Python 3.12 and later.
export TRITON_STABLE_ABI="${TRITON_STABLE_ABI:-ON}"
# Makes setup.py package backend/ into the wheel and register its
# triton.backends entry point, which is how Triton discovers backends.
export TRITON_PLUGIN_DIRS="${CONTAINER_PLUGIN_DIR}"

source "${CONTAINER_PLUGIN_DIR}/scripts/_gpu-stub-env.sh" "${CONTAINER_PLUGIN_DIR}/.cache/gpu-deps-stub"

# setup.sh fetches the toolchains and the pinned Triton checkout; build.sh builds
# Triton with the triton-shared plugin, which also produces the triton-shared-opt
# executable the backend invokes at compile time.
rm -rf "${CONTAINER_TRITON_DIR}/build"
"${CONTAINER_PLUGIN_DIR}/scripts/setup.sh"

${CONTAINER_PLUGIN_DIR}/scripts/build.sh"

# Statically linked against MLIR/LLVM with debug info: stripping takes the tool
# from hundreds of MB down to well under a hundred. It is bundled in the wheel,
# so strip through the staged symlink before packaging.
strip --strip-unneeded "${CONTAINER_PLUGIN_DIR}/backend/bin/triton-shared-opt" 2>/dev/null || true

# The wheel carries Triton, the plugin built into libtriton.so, and the backend
# under triton/backends/triton_shared -- all of it from the tree build.sh just
# populated, so nothing is rebuilt here.
rm -rf "${CONTAINER_OUTPUT_DIR}"
mkdir -p "${CONTAINER_OUTPUT_DIR}"
RAW_WHEEL_DIR="$(mktemp -d)"

cd "${CONTAINER_TRITON_DIR}"
"${PYTHON}" setup.py bdist_wheel -d "${RAW_WHEEL_DIR}"
# A wheel that does not register the backend installs fine and then fails on the
# first kernel, so check it here rather than in a bug report.
"${PYTHON}" - "${RAW_WHEEL_DIR}"/*.whl <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    entries = [name for name in wheel.namelist() if name.endswith("dist-info/entry_points.txt")]
    text = wheel.read(entries[0]).decode() if entries else ""
if "triton_shared = triton.backends.triton_shared" not in text:
    sys.exit(f"{sys.argv[1]} does not register the triton_shared backend")
PY
if [[ "${TRITON_RISCV_SKIP_REPAIR:-0}" == "1" ]]; then
    cp "${RAW_WHEEL_DIR}"/*.whl "${CONTAINER_OUTPUT_DIR}/"
else
    auditwheel repair "${RAW_WHEEL_DIR}"/*.whl -w "${CONTAINER_OUTPUT_DIR}"
fi
rm -rf "${RAW_WHEEL_DIR}"
