# RUN: %PYTHON %s
"""Collection rechecks actual values; PASS labels and hashes are insufficient."""

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(REPO / "examples/BOSCAMEFPGA/vit-base-patch16-224/tools")
)
from nonlinear_simcheck import (  # noqa: E402
    _sources,
    collect_nonlinear_case,
    nonlinear_program_mask,
    nonlinear_shards,
)
from oracles import (  # noqa: E402
    make_inputs,
    nonlinear_error_metrics,
    nonlinear_reference64,
    reference,
)
from simcheck import scenarios_for  # noqa: E402

from build import argument_shapes, describe, digest  # noqa: E402


def sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


# Numerical fixtures exercise evidence rejection only. They are confined to a
# temporary directory and do not establish compilation or emitted execution.
case = describe("gelu_197x3072")
case = {
    **case,
    "shape": [1, 9],
    "constexprs": {"COUNT": 9, "BLOCK": 4},
    "grid": [3, 1, 1],
}
with tempfile.TemporaryDirectory(
    prefix="nonlinear-evidence-test-"
) as directory:
    output_root = Path(directory)
    output = output_root / case["name"]
    output.mkdir()
    module = output / "kernel.boscame.mlir"
    module.write_text(
        "// numerical evidence fixture, not an executable module\n"
    )
    module_hash, kernel_hash = digest(module), digest(case["kernel_source"])
    shards = [
        {"programs": [list(p) for p in points], "runs": []}
        for points in nonlinear_shards(case, 2)
    ]
    runs, archive = [], {}
    for index, (name, stress, order) in enumerate(scenarios_for(case)):
        arrays = make_inputs(case, argument_shapes(case), stress=stress)
        actual = reference(case, arrays)["Out"]
        mathematical = nonlinear_reference64(case, arrays)["Out"]
        archive["round" + str(index)] = actual
        runs.append(
            {
                "name": name,
                "order": "shard-" + order,
                "programs": 3,
                "total_programs": 3,
                "outputs": 9,
                "module_sha256": module_hash,
                "kernel_module_sha256": kernel_hash,
                "input_sha256": sha(arrays["X"]),
                "output_sha256": {"Out": sha(actual)},
                "ordered_f32_reference_sha256": sha(actual),
                "mathematical_reference_sha256": sha(mathematical),
                "nonlinear_error": nonlinear_error_metrics(
                    actual, mathematical
                ),
                "nonlinear_ddr_bytes": {"%arg0": 36, "%arg1": 36},
            }
        )
        for shard in shards:
            points = shard["programs"]
            mask = nonlinear_program_mask(case, points)
            errors = nonlinear_error_metrics(actual[mask], mathematical[mask])
            errors.update(
                mathematical_reference_sha256=sha(mathematical[mask]),
                ordered_f32_reference_sha256=sha(actual[mask]),
            )
            shard["runs"].append(
                {
                    "name": name,
                    "order": order,
                    "selected_programs": points[::-1]
                    if order == "reverse"
                    else points,
                    "programs": len(points),
                    "outputs": int(np.count_nonzero(mask)),
                    "output_sha256": {"Out": sha(actual[mask])},
                    "module_sha256": module_hash,
                    "kernel_module_sha256": kernel_hash,
                    "input_sha256": sha(arrays["X"]),
                    "nonlinear_error": errors,
                    "nonlinear_ddr_bytes": dict.fromkeys(
                        ("%arg0", "%arg1"), int(np.count_nonzero(mask)) * 4
                    ),
                }
            )
    archive_path = output / "nonlinear-outputs.npz"
    np.savez_compressed(archive_path, **archive)
    saved = {
        "name": case["name"],
        "status": "PASS",
        "module_sha256": module_hash,
        "kernel_module_sha256": kernel_hash,
        "simulator_sources": _sources(),
        "programs": 9,
        "total_programs": 9,
        "outputs": 27,
        "max_abs": max(r["nonlinear_error"]["max_abs"] for r in runs),
        "workers": 2,
        "shard_count": 2,
        "shards": shards,
        "runs": runs,
        "output_archive": archive_path.name,
        "output_archive_sha256": digest(archive_path),
    }
    path = output / "simcheck-result.json"
    with patch("simcheck.check_artifacts", return_value=module):
        path.write_text(json.dumps(saved))
        assert collect_nonlinear_case(case, output_root)["status"] == "PASS"
        corrupt = copy.deepcopy(saved)
        corrupt["shards"][0]["runs"][1]["selected_programs"].reverse()
        path.write_text(json.dumps(corrupt))
        assert collect_nonlinear_case(case, output_root)["status"] == "BLOCKED"
        # Even updating the archive hash cannot legitimize altered numerical
        # data: collection recomputes comparisons from stored actual values.
        archive["round0"][0, 0] += 1
        np.savez_compressed(archive_path, **archive)
        corrupt = copy.deepcopy(saved)
        corrupt["output_archive_sha256"] = digest(archive_path)
        path.write_text(json.dumps(corrupt))
        assert collect_nonlinear_case(case, output_root)["status"] == "BLOCKED"
