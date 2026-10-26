#!/usr/bin/env python3
"""Check actual full-grid nonlinear outputs with an independent 80-digit oracle.

The strict simulator produces the required output archives. This tool only
checks those outputs; it cannot supply successful emitted execution evidence.
Dense/sampled observations are not a formal approximation error bound.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mpmath
import numpy as np
from nonlinear_simcheck import collect_nonlinear_case
from oracles import (
    gelu_boundary_inputs,
    make_inputs,
    nonlinear_error_metrics,
    nonlinear_reference64,
    nonlinear_reference_mp,
)
from simcheck import scenarios_for

from build import ROOT, argument_shapes, describe, digest


def sample_round(case, arrays, actual):
    if case["family"] == "gelu":
        points = np.unique(
            np.concatenate(
                (
                    np.arange(min(len(gelu_boundary_inputs()), actual.size)),
                    np.linspace(
                        0,
                        actual.size - 1,
                        min(257, actual.size),
                        dtype=np.int64,
                    ),
                )
            )
        )
        sample = arrays["X"].ravel()[points]
        got = actual.ravel()[points]
        selection = "first complete boundary-pattern period plus 257 stratified positions"
    else:
        length = case["constexprs"]["LENGTH"]
        count = actual.size // length
        points = np.unique(
            np.concatenate(
                (
                    np.arange(min(16, count)),
                    np.linspace(0, count - 1, min(33, count), dtype=np.int64),
                )
            )
        )
        sample = arrays["X"].reshape(-1, length)[points]
        got = actual.reshape(-1, length)[points]
        selection = "first sixteen special-pattern rows plus 33 stratified complete rows"
    inputs = {"X": sample}
    precise = nonlinear_reference_mp(case, inputs, dps=80)["Out"]
    double = nonlinear_reference64(case, inputs)["Out"]
    np.testing.assert_allclose(double, precise, rtol=1e-13, atol=1e-300)
    metrics = nonlinear_error_metrics(got, precise)
    finite = np.isfinite(double) & np.isfinite(precise)
    metrics.update(
        reference="mpmath 80 decimal digits, rounded to binary64 for comparison",
        mpmath_version=mpmath.__version__,
        precision_decimal_digits=80,
        selection=selection,
        selected_units=len(points),
        sampled_outputs=got.size,
        selection_sha256=hashlib.sha256(
            points.astype("<i8").tobytes()
        ).hexdigest(),
        input_sha256=hashlib.sha256(sample.tobytes()).hexdigest(),
        actual_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
        reference_sha256=hashlib.sha256(precise.tobytes()).hexdigest(),
        max_binary64_reference_difference=float(
            np.max(np.abs(double[finite] - precise[finite]), initial=0)
        ),
        evidence_kind="sampled numerical observations; no claimed strict global error bound",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=("gelu_197x3072", "softmax_12x197x197"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/nonlinear-numerics.json"
    )
    args = parser.parse_args()
    records = []
    for name in args.case or ("gelu_197x3072", "softmax_12x197x197"):
        case = describe(name)
        record = {"name": name, "status": "BLOCKED", "reason": ""}
        try:
            collected = collect_nonlinear_case(case, args.output)
            if collected["status"] != "PASS":
                raise RuntimeError(
                    "strict full-grid evidence unavailable: "
                    + collected["reason"]
                )
            output = args.output / name
            record.update(
                module_sha256=collected["module_sha256"],
                output_archive_sha256=collected["output_archive_sha256"],
                numerical_checker_sha256=digest(Path(__file__)),
                execution_report_sha256=digest(output / "simcheck-result.json"),
            )
            rounds = []
            with np.load(
                output / "nonlinear-outputs.npz", allow_pickle=False
            ) as archive:
                for index, (label, stress, _) in enumerate(scenarios_for(case)):
                    arrays = make_inputs(
                        case, argument_shapes(case), stress=stress
                    )
                    rounds.append(
                        {
                            "name": label,
                            **sample_round(
                                case, arrays, archive["round" + str(index)]
                            ),
                        }
                    )
            record.update(status="PASS", rounds=rounds)
        except Exception as error:
            record.update(
                status="FAIL", reason=f"{type(error).__name__}: {error}"
            )
        records.append(record)
        print(
            f"{record['status']}: {name} independent high-precision samples",
            flush=True,
        )
    report = {
        "status": "PASS"
        if all(r["status"] == "PASS" for r in records)
        else "FAIL",
        "cases": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
