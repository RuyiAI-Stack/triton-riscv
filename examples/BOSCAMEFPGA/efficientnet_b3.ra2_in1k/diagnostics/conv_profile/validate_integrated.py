#!/usr/bin/env python3
"""Validate final tile-cached W8A8 convolution artifacts with explicit coverage."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from export import argument_shapes, inventory  # noqa: E402
from oracles import make_inputs  # noqa: E402
from quantized_conv_pipeline import pipeline_layout  # noqa: E402
from simcheck import simulate_case, check_artifacts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/w8a8-opt/integrated"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "build/w8a8-opt/integrated-numerical-report.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": "ordinary tiny full; all other cases sampled first/middle/last spatial blocks and depthwise channels",
        "results": [],
    }
    dest = args.report.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for case in [
        c for c in inventory() if c["family"] in ("conv2d_f32", "dwconv2d_f32")
    ]:
        lay = pipeline_layout(case)
        mod = check_artifacts(case, output)
        points = (
            None
            if lay["groups"] * lay["blocks"] == 1
            else sorted(
                itertools.product(
                    sorted({0, lay["groups"] // 2, lay["groups"] - 1}),
                    sorted({0, lay["blocks"] // 2, lay["blocks"] - 1}),
                )
            )
        )
        actual, metrics = simulate_case(
            case,
            make_inputs(case, argument_shapes(case)),
            output,
            progress=False,
            selected_conv_tiles=points,
        )
        d = output / case["name"]
        frontend = json.loads((d / "frontend.json").read_text())
        functions = set(
            re.findall(
                r"^\s*func[.]func(?:\s+\w+)?\s+@([^ (]+)[(]", mod.read_text(), re.M
            )
        )
        assert (
            functions == {x["symbol"] for x in frontend["stages"]}
            and len(functions) == 4
        )
        assert frontend["stages"][1]["grid"][0] == lay["blocks"]
        assert "mqma.b.mm" in (d / "kernel.nr.s").read_text()
        record = {
            "case": case["name"],
            "status": "PASS",
            "selected_tiles": points,
            "total_tiles": lay["groups"] * lay["blocks"],
            "metrics": metrics,
            "bin_sha256": hashlib.sha256(
                (d / (case["name"] + ".bin")).read_bytes()
            ).hexdigest(),
            "boscame_sha256": hashlib.sha256(mod.read_bytes()).hexdigest(),
            "stage_count": 4,
            "mma_i8_i8_i32": True,
        }
        report["results"].append(record)
        dest.write_text(json.dumps(report, indent=2) + chr(10))
        print(
            "PASS",
            case["name"],
            "tiles=" + str("full" if points is None else len(points)),
            "outputs=" + str(metrics["outputs"]),
            flush=True,
        )
    report["pass_count"] = len(report["results"])
    report["output_count"] = sum(x["metrics"]["outputs"] for x in report["results"])
    report["status"] = "PASS"
    dest.write_text(json.dumps(report, indent=2) + chr(10))
    print(
        "PASS all", report["pass_count"], "outputs", report["output_count"], flush=True
    )


if __name__ == "__main__":
    main()
