"""Compile representative FlagGems kernels and verify their RV64 results in QEMU."""

from pathlib import Path

import numpy as np

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
    output_dir,
    grid=(1,),
    constexprs=None,
    signature=None,
    expected,
):
    """Run one RV64 kernel and compare its buffers with a CPU-computed reference."""
    output = Path(output_dir) / f"{name}.elf"
    output.parent.mkdir(parents=True, exist_ok=True)
    compile_and_run_kernel(
        kernel,
        arguments,
        grid=grid,
        output=output,
        constexprs=constexprs,
        signature=signature,
        expected={
            key: np.asarray(value).ravel().tolist() for key, value in expected.items()
        },
        atol=1e-5,
    )
    print(f"{name}: numerical PASS ({output})")


def _run_where(output_dir):
    cond = np.array([1, 0, 1, 0], dtype=np.uint8)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    reference = np.where(cond != 0, x, y)
    _run(
        where_kernel,
        "where",
        {
            "cond_ptr": cond,
            "x_ptr": x,
            "y_ptr": y,
            "out_ptr": np.zeros_like(reference),
            "n_elements": reference.size,
        },
        output_dir=output_dir,
        constexprs={"BLOCK_SIZE": 4},
        signature={"cond_ptr": "*u8"},
        expected={"out_ptr": reference},
    )


def _run_vdot(output_dir):
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    other = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    reference = np.array([np.dot(inp, other)], dtype=np.float32)
    _run(
        dot_scalar_kernel,
        "vdot",
        {
            "inp_ptr": inp,
            "other_ptr": other,
            "out_ptr": np.zeros_like(reference),
            "n_elements": inp.size,
            "inp_stride": 1,
            "other_stride": 1,
        },
        output_dir=output_dir,
        expected={"out_ptr": reference},
    )


def _run_var(output_dir):
    values = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    reference = np.var(values, axis=1, ddof=1, dtype=np.float32)
    _run(
        var_scalar_rows_kernel,
        "var",
        {
            "X": values.ravel(),
            "Var": np.zeros_like(reference),
            "M": values.shape[0],
            "N": values.shape[1],
            "correction": 1.0,
        },
        output_dir=output_dir,
        grid=(values.shape[0],),
        expected={"Var": reference},
    )


def _run_upsample_nearest1d(output_dir):
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    reference = np.repeat(values, 2)
    _run(
        upsample_nearest1d_kernel,
        "upsample_nearest1d",
        {
            "ptr_o": np.zeros_like(reference),
            "ptr_i": values,
            "N": 1,
            "C": 1,
            "OL": reference.size,
            "IL": values.size,
            "reciprocal_scale_l": 0.5,
        },
        output_dir=output_dir,
        constexprs={
            "BLOCK_SIZE": 8,
            "SAME_L": False,
            "USE_INT32_IDX": True,
        },
        expected={"ptr_o": reference},
    )


def test_qemu_where(tmp_path):
    _run_where(tmp_path)


def test_qemu_vdot(tmp_path):
    _run_vdot(tmp_path)


def test_qemu_var(tmp_path):
    _run_var(tmp_path)


def test_qemu_upsample_nearest1d(tmp_path):
    _run_upsample_nearest1d(tmp_path)


def main():
    output_dir = Path("artifacts/issue35/qemu-flaggems")
    _run_where(output_dir)
    _run_vdot(output_dir)
    _run_var(output_dir)
    _run_upsample_nearest1d(output_dir)


if __name__ == "__main__":
    main()
