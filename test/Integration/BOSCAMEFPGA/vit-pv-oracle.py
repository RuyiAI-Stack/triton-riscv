# RUN: %PYTHON %s
"""PV references preserve zero products, per-block adds and f32 rounding."""

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
from oracles import (  # noqa: E402
    _matmul_blocked_sequential,
    _matmul_sequential,
    make_inputs,
    reference,
)
from simcheck import (  # noqa: E402
    SimulationBlocked,
    assemble_pv_round,
    audit_pv_program,
    check_pv_selected_output,
    pv_program_mask,
    pv_shards,
    scenarios_for,
    validate_pv_shard_evidence,
)

from export import argument_shapes, describe, digest  # noqa: E402


def test_block_order():
    for k in (197, 192):
        a = np.zeros((1, k), np.float32)
        b = np.ones((k, 1), np.float32)
        a[0, 0], a[0, 64], a[0, 65] = 2**25, -(2**25), 1
        assert _matmul_blocked_sequential(a, b, 64).item() == 0
        assert _matmul_sequential(a, b).item() == 1
    a = np.array([[1, 2, -4, 8, 0.5]], np.float32)
    b = np.ones((5, 1), np.float32)
    assert _matmul_blocked_sequential(a, b, 64).item() == 7.5
    try:
        _matmul_blocked_sequential(a, b, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive block accepted")


def test_no_fma():
    a = np.array([[-1, 1 + 2**-23]], np.float32)
    b = np.array([[1], [1 - 2**-23]], np.float32)
    assert _matmul_blocked_sequential(a, b, 64).item() == 0
    fused_control = np.float32(float(a[0, 1]) * float(b[1, 0]) - 1)
    assert fused_control == np.float32(-(2**-46))


def test_padding_and_special_values():
    a = np.array([[np.inf, 0, -0.0, 0, 0]], np.float32)
    b = np.ones((5, 1), np.float32)
    original = _matmul_sequential
    shapes = []
    import oracles

    def observe(left, right):
        shapes.append((left.shape, right.shape))
        assert np.all(left[..., 5:] == 0)
        assert not np.any(np.signbit(left[..., 5:]))
        assert np.all(right[..., 5:, :] == 0)
        assert not np.any(np.signbit(right[..., 5:, :]))
        return original(left, right)

    oracles._matmul_sequential = observe
    try:
        assert np.isposinf(_matmul_blocked_sequential(a, b, 64).item())
        a[0, 0] = np.nan
        assert np.isnan(_matmul_blocked_sequential(a, b, 64).item())
    finally:
        oracles._matmul_sequential = original
    assert shapes == [((1, 64), (64, 1))] * 2


def test_real_case_stress():
    case = describe("attention_pv_12x197x64x197")
    arrays = make_inputs(case, argument_shapes(case), stress=True)
    assert np.all(np.isnan(arrays["C"]))
    expected = reference(case, arrays)["C"]
    assert np.all(np.isfinite(expected))
    assert np.any(expected != _matmul_sequential(arrays["A"], arrays["B"]))
    assert np.array_equal(
        expected, reference(case, {**arrays, "C": expected})["C"]
    )


def test_logical_row_audit():
    case = describe("attention_pv_12x197x64x197")
    accesses = [
        ("%arg0", row * 788, (row + 1) * 788, "read") for row in range(16)
    ]
    accesses += [
        ("%arg1", row * 256, row * 256 + 64, "read") for row in range(197)
    ]
    accesses += [
        ("%arg2", row * 256, row * 256 + 64, "write") for row in range(16)
    ]
    dma = [("%arg1", "spm", 0, 0, 64, 0)] * 197
    matrix = [
        {
            "source": "%arg0",
            "source_offset": 0,
            "rows": 16,
            "row_bytes": width,
            "source_pitch": 788,
            "target_pitch": 256,
            "instruction_bits": 32,
            "register_class": "acc",
        }
        for width in (256, 256, 256, 20)
    ]
    audit = audit_pv_program(case, (0, 0, 0), accesses, dma, matrix)
    assert sum(audit["ddr_bytes"].values()) == 26240
    # This still fits the complete A backing buffer, but crosses a K row.
    crossing = [("%arg0", 4, 792, "read"), *accesses[1:]]
    old_c = [*accesses, ("%arg2", 0, 4, "read")]
    for invalid in (crossing, old_c):
        try:
            audit_pv_program(case, (0, 0, 0), invalid, dma, matrix)
        except AssertionError as error:
            assert "logical row violation" in str(error)
        else:
            raise AssertionError("invalid PV logical access passed")
    assert [item[2] for item in scenarios_for(case)] == [
        "forward",
        "tail-first",
        "reverse",
    ]


def test_shard_coverage():
    case = describe("attention_pv_12x197x64x197")
    shards = pv_shards(case)
    points = [point for shard in shards for point in shard]
    assert len(shards) == 48 and all(len(shard) == 13 for shard in shards)
    assert len(points) == len(set(points)) == 624
    assert set(points) == set(itertools.product(range(13), range(4), range(12)))
    assert np.all(pv_program_mask(case, points))
    for invalid in (
        [],
        [points[0], points[0]],
        [(13, 0, 0)],
        [(0, 4, 0)],
        [(0, 0, 12)],
    ):
        try:
            pv_program_mask(case, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shard selection accepted")


def test_selected_output_and_assembly():
    case = describe("attention_pv_12x197x64x197")
    case = {
        **case,
        "shape": [1, 3, 2, 1],
        "grid": [2, 1, 1],
        "constexprs": {**case["constexprs"], "BM": 2, "BN": 2},
    }
    expected = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    initial = np.full_like(expected, np.nan)
    mask = pv_program_mask(case, [(0, 0, 0)])
    actual = initial.copy()
    actual[mask] = expected[mask]
    check_pv_selected_output(case, [(0, 0, 0)], actual, expected, initial)
    actual[~mask] = expected[~mask]
    try:
        check_pv_selected_output(case, [(0, 0, 0)], actual, expected, initial)
    except AssertionError as error:
        assert "unauthorized" in str(error)
    else:
        raise AssertionError("write outside selected programs accepted")
    pieces = []
    for point in ((0, 0, 0), (1, 0, 0)):
        selected = pv_program_mask(case, [point])
        values = expected[selected].copy()
        hashes = {"C": hashlib.sha256(values.tobytes()).hexdigest()}
        pieces.append(
            {
                "programs": [point],
                "values": values,
                "metrics": {
                    "programs": 1,
                    "outputs": len(values),
                    "selected_programs": [point],
                    "output_sha256": hashes,
                    "reference_sha256": hashes,
                    "bit_exact_f32": True,
                },
            }
        )
    assert np.array_equal(assemble_pv_round(case, pieces, expected), expected)
    for invalid in (pieces[:1], [pieces[0], pieces[0], pieces[1]]):
        try:
            assemble_pv_round(case, invalid, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("incomplete/duplicate aggregate accepted")
    for corrupt in ("hash", "value", "executed-points"):
        invalid = copy.deepcopy(pieces)
        if corrupt == "hash":
            invalid[0]["metrics"]["output_sha256"] = {"C": "invalid"}
        elif corrupt == "value":
            invalid[0]["values"][0] = 100
        else:
            invalid[0]["metrics"]["selected_programs"] = [(1, 0, 0)]
        try:
            assemble_pv_round(case, invalid, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt executed-slice evidence accepted")


def test_collect_shard_metadata():
    case = describe("attention_pv_12x197x64x197")
    case = {
        **case,
        "shape": [2, 3, 2, 1],
        "grid": [2, 1, 2],
        "constexprs": {**case["constexprs"], "BM": 2, "BN": 2},
    }
    shards = []
    for programs in pv_shards(case):
        mask = pv_program_mask(case, programs)
        runs = []
        for name, stress, order in scenarios_for(case):
            result = reference(
                case, make_inputs(case, argument_shapes(case), stress=stress)
            )["C"][mask]
            hashes = {"C": hashlib.sha256(result.tobytes()).hexdigest()}
            runs.append(
                {
                    "name": name,
                    "order": order,
                    "module_sha256": "current",
                    "kernel_module_sha256": digest(case["kernel_source"]),
                    "programs": len(programs),
                    "outputs": len(result),
                    "selected_programs": programs
                    if order == "forward"
                    else programs[::-1],
                    "bit_exact_f32": True,
                    "output_sha256": hashes,
                    "reference_sha256": hashes,
                    "pv_access_audit": {},
                }
            )
        shards.append({"programs": programs, "runs": runs})
    saved = {"shards": shards, "shard_count": 2}
    validate_pv_shard_evidence(case, saved, "current")
    mixed_heads = copy.deepcopy(saved)
    (
        mixed_heads["shards"][0]["programs"][1],
        mixed_heads["shards"][1]["programs"][1],
    ) = (
        mixed_heads["shards"][1]["programs"][1],
        mixed_heads["shards"][0]["programs"][1],
    )
    wrong_order = copy.deepcopy(saved)
    wrong_order["shards"][0]["runs"][0]["selected_programs"].reverse()
    wrong_reverse = copy.deepcopy(saved)
    wrong_reverse["shards"][0]["runs"][2]["selected_programs"].reverse()
    for invalid in (mixed_heads, wrong_order, wrong_reverse):
        try:
            validate_pv_shard_evidence(case, invalid, "current")
        except SimulationBlocked:
            pass
        else:
            raise AssertionError(
                "false shard partition/order evidence accepted"
            )


test_block_order()
test_no_fma()
test_padding_and_special_values()
test_real_case_stress()
test_logical_row_audit()
test_shard_coverage()
test_selected_output_and_assembly()
test_collect_shard_metadata()
