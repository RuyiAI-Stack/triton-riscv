"""Compile representative FlagGems kernels to RV64 ELF and verify in QEMU."""

from pathlib import Path

from flaggems.upsample_nearest1d import upsample_nearest1d_kernel
from flaggems.var import var_scalar_rows_kernel
from flaggems.vdot import dot_scalar_kernel
from flaggems.where import where_kernel
from triton.backends.triton_shared.riscv import compile_and_run_kernel


def _run(
    kernel,
    name,
    arguments,
    *,
    grid=(1,),
    constexprs=None,
    signature=None,
    expected,
):
    output = Path("artifacts/issue35/qemu-flaggems") / f"{name}.elf"
    output.parent.mkdir(parents=True, exist_ok=True)
    compile_and_run_kernel(
        kernel,
        arguments,
        grid=grid,
        output=output,
        constexprs=constexprs,
        signature=signature,
        expected=expected,
        atol=1e-5,
    )
    print(f"{name}: PASS ({output})")


def main():
    _run(
        where_kernel,
        "where",
        {
            "cond_ptr": [1, 0, 1, 0],
            "x_ptr": [1.0, 2.0, 3.0, 4.0],
            "y_ptr": [10.0, 20.0, 30.0, 40.0],
            "out_ptr": [0.0] * 4,
            "n_elements": 4,
        },
        constexprs={"BLOCK_SIZE": 4},
        signature={"cond_ptr": "*u8"},
        expected={"out_ptr": [1.0, 20.0, 3.0, 40.0]},
    )
    _run(
        dot_scalar_kernel,
        "vdot",
        {
            "inp_ptr": [1.0, 2.0, 3.0, 4.0],
            "other_ptr": [2.0, 3.0, 4.0, 5.0],
            "out_ptr": [0.0],
            "n_elements": 4,
            "inp_stride": 1,
            "other_stride": 1,
        },
        expected={"out_ptr": [40.0]},
    )
    _run(
        var_scalar_rows_kernel,
        "var",
        {
            "X": [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            "Var": [0.0, 0.0],
            "M": 2,
            "N": 4,
            "correction": 1.0,
        },
        grid=(2,),
        expected={"Var": [1.6666666667, 6.6666666667]},
    )
    _run(
        upsample_nearest1d_kernel,
        "upsample_nearest1d",
        {
            "ptr_o": [0.0] * 8,
            "ptr_i": [1.0, 2.0, 3.0, 4.0],
            "N": 1,
            "C": 1,
            "OL": 8,
            "IL": 4,
            "reciprocal_scale_l": 0.5,
        },
        constexprs={
            "BLOCK_SIZE": 8,
            "SAME_L": False,
            "USE_INT32_IDX": True,
        },
        expected={"ptr_o": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]},
    )


if __name__ == "__main__":
    main()
