# RUN: %PYTHON %s
"""Independent nonlinear references, IEEE cases and executed-slice coverage."""

import copy
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(REPO / "examples/BOSCAMEFPGA/vit-base-patch16-224/tools")
)
from nonlinear_simcheck import (  # noqa: E402
    assemble_nonlinear_round,
    audit_nonlinear_program,
    nonlinear_program_mask,
    nonlinear_shards,
    selected_nonlinear_comparison,
)
from oracles import (  # noqa: E402
    NONLINEAR_TOLERANCE,
    compare,
    gelu_boundary_inputs,
    make_inputs,
    nonlinear_error_metrics,
    nonlinear_reference64,
    nonlinear_reference_mp,
    reference,
)

from export import argument_shapes, describe  # noqa: E402


def rejects(function, *args):
    try:
        function(*args)
    except (AssertionError, ValueError):
        return
    raise AssertionError(
        "invalid nonlinear numerical/coverage evidence accepted"
    )


def test_references():
    assert NONLINEAR_TOLERANCE == 2e-5
    case = describe("gelu_197x3072")
    x = np.array(
        [
            [
                -12,
                -8,
                -4,
                -0.0,
                0,
                np.finfo(np.float32).tiny,
                np.inf,
                -np.inf,
                np.nan,
            ]
        ],
        np.float32,
    )
    arrays = {"X": x}
    with np.errstate(invalid="ignore"):
        ordered = reference(case, arrays)["Out"]
        mathematical = nonlinear_reference64(case, arrays)["Out"]
        precise = nonlinear_reference_mp(case, arrays)["Out"]
    # The independent erfc reference retains the negative tail lost when the
    # source expression rounds erf to f32 before adding one.
    assert ordered[0, 1] == 0 and mathematical[0, 1] < 0
    np.testing.assert_allclose(mathematical, precise, rtol=1e-13, atol=1e-300)
    assert np.signbit(precise[0, 3]) and not np.signbit(precise[0, 4])
    assert np.isposinf(precise[0, 6]) and np.isnan(precise[0, 7:]).all()
    values = gelu_boundary_inputs()
    assert (
        np.isposinf(values).any()
        and np.isneginf(values).any()
        and np.isnan(values).any()
    )
    assert np.any((values < -4) & (values > -10))
    assert np.any((values > 0) & (values < np.finfo(np.float32).tiny))
    # Existing absolute/relative acceptance is unchanged and still detects an
    # excessive error near zero, incorrect nonfinite classes and signed zero.
    nonlinear_error_metrics(np.array([1e-5]), np.array([0.0]))
    rejects(nonlinear_error_metrics, np.array([4e-5]), np.array([0.0]))
    rejects(nonlinear_error_metrics, np.array([np.inf]), np.array([np.nan]))
    rejects(
        compare,
        case,
        {"Out": np.array([0], np.float32)},
        {"Out": np.array([-0.0], np.float32)},
    )


def test_softmax_reference():
    case = describe("softmax_12x197x197")
    case = {
        **case,
        "shape": [1, 16, 7],
        "constexprs": {**case["constexprs"], "LENGTH": 7, "BLOCK": 8},
        "grid": [16, 1, 1],
    }
    arrays = make_inputs(case, argument_shapes(case), stress=True)
    mathematical = nonlinear_reference64(case, arrays)["Out"]
    precise = nonlinear_reference_mp(case, arrays)["Out"]
    np.testing.assert_allclose(mathematical, precise, rtol=1e-13, atol=1e-300)
    rows = mathematical.reshape(16, 7)
    assert np.isnan(rows[[7, 8, 9, 10]]).all()
    assert np.all(rows[6, ::2] == 0)
    assert not np.signbit(rows[6, ::2]).any()
    finite_rows = np.all(np.isfinite(rows), axis=1)
    np.testing.assert_allclose(
        rows[finite_rows].sum(axis=1), 1, rtol=1e-15, atol=1e-15
    )
    # Large negative exponents remain positive when representable in binary64;
    # the reference does not pretruncate their mass to make a recipe pass.
    assert 0 < rows[5, 0] < 1e-40
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        expected = reference(case, arrays)["Out"]
    nonlinear_error_metrics(expected, precise)


def test_real_grid_coverage():
    for name, programs, outputs in (
        ("gelu_197x3072", 4728, 605184),
        ("softmax_12x197x197", 2364, 465708),
    ):
        case = describe(name)
        shards = nonlinear_shards(case, 48)
        points = [p for shard in shards for p in shard]
        assert len(points) == len(set(points)) == programs
        assert set(points) == set(
            itertools.product(*(range(n) for n in case["grid"]))
        )
        assert np.count_nonzero(nonlinear_program_mask(case, points)) == outputs
        for bad in (
            [],
            [points[0], points[0]],
            [(case["grid"][0], 0, 0)],
            [(0, 1, 0)],
        ):
            rejects(nonlinear_program_mask, case, bad)


def test_partial_programs_and_assembly():
    case = describe("gelu_197x3072")
    case = {
        **case,
        "shape": [1, 9],
        "constexprs": {"COUNT": 9, "BLOCK": 4},
        "grid": [3, 1, 1],
    }
    arrays = {
        "X": np.linspace(-4, 4, 9, dtype=np.float32).reshape(1, 9),
        "Out": np.full((1, 9), np.nan, np.float32),
    }
    expected = reference(case, arrays)["Out"]
    mathematical = nonlinear_reference64(case, arrays)["Out"]
    accesses = [("%arg0", 32, 36, "read"), ("%arg1", 32, 36, "write")]
    assert audit_nonlinear_program(case, (2, 0, 0), accesses) == {
        "%arg0": 4,
        "%arg1": 4,
    }
    rejects(audit_nonlinear_program, case, (2, 0, 0), [*accesses, accesses[0]])
    rejects(
        audit_nonlinear_program,
        case,
        (2, 0, 0),
        [("%arg0", 28, 36, "read"), accesses[1]],
    )
    mask = nonlinear_program_mask(case, [(2, 0, 0)])
    selected = arrays["Out"].copy()
    selected[mask] = expected[mask]
    selected_nonlinear_comparison(
        case, [(2, 0, 0)], {"Out": selected}, {"Out": expected}, arrays
    )
    selected[0, 0] = 0
    rejects(
        selected_nonlinear_comparison,
        case,
        [(2, 0, 0)],
        {"Out": selected},
        {"Out": expected},
        arrays,
    )
    pieces = []
    for points in nonlinear_shards(case, 2):
        mask = nonlinear_program_mask(case, points)
        values = expected[mask].copy()
        pieces.append(
            {
                "programs": points,
                "values": values,
                "metrics": {
                    "programs": len(points),
                    "outputs": values.size,
                    "selected_programs": points,
                    "output_sha256": {
                        "Out": hashlib.sha256(values.tobytes()).hexdigest()
                    },
                },
            }
        )
    actual, _ = assemble_nonlinear_round(case, pieces, expected, mathematical)
    np.testing.assert_array_equal(actual, expected)
    rejects(assemble_nonlinear_round, case, pieces[:1], expected, mathematical)
    rejects(
        assemble_nonlinear_round,
        case,
        [*pieces, pieces[0]],
        expected,
        mathematical,
    )
    corrupt = copy.deepcopy(pieces)
    corrupt[0]["values"][0] += 1
    rejects(assemble_nonlinear_round, case, corrupt, expected, mathematical)


test_references()
test_softmax_reference()
test_real_grid_coverage()
test_partial_programs_and_assembly()
