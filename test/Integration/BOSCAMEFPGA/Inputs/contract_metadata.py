"""Check the actual exported kernels at the backend-independent boundary."""

import re
import subprocess
import sys
from pathlib import Path


def main():
    directory, prefix, opt = sys.argv[1:]
    outputs = {}
    for fixture, grid in (
        ("matmul-i8.ttir", "triton_matmul_i8_128x1024x128:8,64,1"),
        ("matmul-tiled-i8.ttir", "triton_matmul_i8_196x768x768:7,6,1"),
        ("layernorm.ttir", "triton_layernorm_197x768:197,1,1"),
        ("matmul-lineage.ttir", None),
        ("conditional-address.ttir", None),
    ):
        mode = "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias"
        if grid:
            mode += " launch-grid=" + grid
        command = [opt, str(Path(directory) / fixture)]
        result = subprocess.run(
            command + [mode, "--kernel-contract-verify"],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0 and not result.stderr, (
            fixture,
            result.stderr,
        )
        output = result.stdout
        assert all(
            word not in output
            for word in ("bosc_ame.", "plan.", "tensor<", "bufferization.")
        ), fixture
        report = subprocess.run(
            command + [mode + " dump-facts=true"],
            text=True,
            capture_output=True,
        )
        assert report.returncode == 0 and report.stdout == output, fixture
        assert report.stderr.strip() == output.strip(), fixture
        Path(prefix + "." + fixture + ".mlir").write_text(output)
        outputs[fixture] = output
        if grid:
            assert "contract.compute_graph = {complete = true" in output, (
                fixture
            )
    matmul = outputs["matmul-i8.ttir"]
    assert matmul.count("linalg.matmul") == 1
    assert "a_source_load_origin_ids = array<i64: 0>" in matmul
    assert "b_source_load_origin_ids = array<i64: 1>" in matmul
    assert "contract.loop_carried_slots" in matmul
    # Initial accumulator contents are represented by the third dot input.
    task = re.search(r"linalg.matmul.*contract.task = \{([^}]+)\}", matmul)[1]
    inputs = re.search(r"input_values = array<i64: ([^>]+)>", task)[1]
    assert len(inputs.split(",")) == 3
    layernorm = outputs["layernorm.ttir"]
    assert layernorm.count("linalg.reduce") == 2
    assert layernorm.count("contract.access") == 4
    assert (
        layernorm.count('kind = "rectangular", valid_shape = array<i64: 768>')
        == 4
    )
    assert "other = 1.000000e+00 : f32" in layernorm
    assert "other = 0.000000e+00 : f32" in layernorm
    stores = [
        line for line in layernorm.splitlines() if 'kind = "store"' in line
    ]
    assert (
        len(stores) == 1
        and "source_load_origin_ids = array<i64: 0, 1, 2>" in stores[0]
    )
    lineage = outputs["matmul-lineage.ttir"]
    assert "a_source_load_origin_ids = array<i64: 0>" in lineage
    assert "b_source_load_origin_ids = array<i64: 1>" in lineage
    assert "a_source_lineage_complete = false" in lineage
    assert "b_source_lineage_complete = false" in lineage
    assert "complete = false, limitation =" in lineage


if __name__ == "__main__":
    main()
