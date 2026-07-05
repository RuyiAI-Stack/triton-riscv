import torch

import triton
import triton.language as tl
import benchmark
from triton.backends.triton_shared.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def nested_use_same_level_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]

    for i1 in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j1 in range(0, 2):
            a_ptrs += 2 * stride_n

        for i6 in range(0, 2):
            a1 = tl.load(a_ptrs)
            a_ptrs += 2 * stride_n
            a3 = tl.load(a_ptrs)
            tl.store(c_ptrs, a1)
            c_ptrs += 2 * stride_n

            c_ptrs += 2 * stride_n
            tl.store(c_ptrs, a3)
            c_ptrs += 2 * stride_n
            a_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n


def run_nested_use_same_level_loop_results(n_rows, n_cols):
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape(
        [n_rows, n_cols]
    )
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)

    def grid(meta):
        return (1,)

    nested_use_same_level_loop_results[grid](x, output, x.stride(0), x.stride(1))
    return output


def nested_use_same_level_loop_results_reference(n_rows, n_cols):
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape(
        [n_rows, n_cols]
    )
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    store_cols = [0, 4, 6, 10, 12, 16, 18, 22]
    load_cols = [4, 6, 8, 10, 18, 20, 22, 24]
    for store_col, load_col in zip(store_cols, load_cols):
        output[0:2, store_col : store_col + 2] = x[0:2, load_col : load_col + 2]
    return output


def bench_nested_use_same_level_loop_results(n_rows, n_cols):
    benchmark.compare_providers(
        f"bench_nested_use_same_level_loop_results(n_rows={n_rows}, n_cols={n_cols})",
        {
            "torch": lambda: nested_use_same_level_loop_results_reference(
                n_rows, n_cols
            ),
            "triton-riscv": lambda: run_nested_use_same_level_loop_results(
                n_rows, n_cols
            ),
        },
    )


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for n_rows, n_cols in [(2**10, 2**10), (2**12, 2**12), (2**14, 2**14)]:
        bench_nested_use_same_level_loop_results(n_rows, n_cols)
