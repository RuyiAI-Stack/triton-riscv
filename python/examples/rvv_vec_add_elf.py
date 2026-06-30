"""Build a Triton vector-add kernel as RVV ELF and execute it with QEMU."""

import triton
import triton.language as tl
from triton.backends.triton_shared.riscv import standalone_kernel_cli


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main() -> None:
    block_size = 16
    size = 64
    x = [float(index) for index in range(size)]
    y = [float(2 * index) for index in range(size)]
    standalone_kernel_cli(
        vector_add_kernel,
        arguments={
            "x_ptr": x,
            "y_ptr": y,
            "output_ptr": [0.0] * size,
            "n_elements": size,
        },
        constexprs={"BLOCK_SIZE": block_size},
        grid=(triton.cdiv(size, block_size),),
        expected={"output_ptr": [lhs + rhs for lhs, rhs in zip(x, y)]},
        default_output="artifacts/riscv/rvv-vector-add.elf",
    )


if __name__ == "__main__":
    main()
