#!/usr/bin/env bash
#
# Sourced helper (not meant to be executed directly).
#
# Upstream Triton unconditionally downloads host-specific GPU toolchains at
# build time (nvcc, cuobjdump, AMD codegen, proton, CUPTI). riscv64 builds of
# those do not exist, so building the stock checkout there requires telling
# Triton not to look for them. Pointing every download location at an empty stub
# directory makes Triton's download_and_copy* helpers skip the download, and the
# components that need real CUDA/ROCm headers are disabled.
#
# Usage:
#   source scripts/_gpu-stub-env.sh <stub-dir>

_gpu_stub_dir="${1:?usage: source scripts/_gpu-stub-env.sh <stub-dir>}"
mkdir -p "${_gpu_stub_dir}"

export TRITON_AMD_CODEGEN_PATH="${_gpu_stub_dir}"
export TRITON_PTXAS_PATH="${_gpu_stub_dir}"
export TRITON_CUOBJDUMP_PATH="${_gpu_stub_dir}"
export TRITON_NVDISASM_PATH="${_gpu_stub_dir}"
export TRITON_CUDACRT_PATH="${_gpu_stub_dir}"
export TRITON_CUDART_PATH="${_gpu_stub_dir}"
export TRITON_CUPTI_INCLUDE_PATH="${_gpu_stub_dir}"
export TRITON_CUPTI_LIB_PATH="${_gpu_stub_dir}"
export TRITON_CUPTI_LIB_BLACKWELL_PATH="${_gpu_stub_dir}"

export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DTRITON_BUILD_NVIDIA_GSAN_RUNTIME=OFF -DTRITON_BUILD_PROTON=OFF"

unset _gpu_stub_dir
