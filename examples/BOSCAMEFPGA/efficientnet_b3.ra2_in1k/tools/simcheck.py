#!/usr/bin/env python3
"""Numeric simcheck driver for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.

Inputs: exported kernels and CPU oracles / reference tensors for each registered
case.

Outputs: pass/fail and max-error per operator without needing the FPGA board.

When used: after AOT export, before or instead of a board run, from the model
directory.
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from boscame_paths import buddy_tools
from export import (
    ROOT,
    argument_shapes,
    describe,
    digest,
    inventory,
    summary_report_path,
    tool,
)
from nonlinear_simcheck import (
    audit_nonlinear_program,
    check_nonlinear_sharded,
    collect_nonlinear_case,
    nonlinear_program_mask,
    selected_nonlinear_comparison,
)
from oracles import DTYPES, compare, make_inputs, output_names, reference
from quantized_conv_oracle import (
    is_quantized_conv,
    quantization_error_metrics,
    quantize_rows,
    stage_reference,
    weight_rows,
)
from quantized_conv_oracle import reference as quantized_conv_reference
from quantized_linear_pipeline import is_quantized_linear
from quantized_linear_simcheck import program_counts as linear_program_counts

sys.path.insert(0, str(buddy_tools()))
from boscame_buffer_sim import Machine, SimError


class SimulationBlocked(RuntimeError):
    """The real pipeline or strict simulator cannot execute the requested case."""


def scenarios_for(case):
    if is_quantized_linear(case):
        return [("normal", False, "adapter"), ("stress", True, "adapter")]
    if is_quantized_conv(case):
        return [("normal", False, "adapter")]
    if case["family"] == "conv2d_i8":
        return [
            ("normal", False, "forward"),
            ("signed-extremes-tail-first", True, "tail-first"),
            ("signed-extremes-reverse", True, "reverse"),
        ]
    if case["family"] in ("gelu", "softmax"):
        return [
            ("normal", False, "forward"),
            ("boundaries-special-reverse", True, "reverse"),
            ("boundaries-special-forward", True, "forward"),
        ]
    if case["family"] in ("matmul_i8", "matmul_i8_overwrite"):
        return [
            ("normal", False, "forward"),
            ("signed-wrap-tail-first", True, "tail-first"),
            ("signed-wrap-reverse", True, "reverse"),
        ]
    if case["family"] == "attention_pv":
        return [
            ("normal", False, "forward"),
            ("ordered-blocks-tail-first", True, "tail-first"),
            ("ordered-blocks-reverse", True, "reverse"),
        ]
    if case["family"] == "per_token_quantization" and case["shape"][0] == 1:
        return [
            ("all-zero", False, "forward"),
            ("nonzero-random", False, "reverse"),
            ("rounding-boundaries", True, "forward"),
        ]
    return [("normal", False, "reverse")]


def quantize_pattern(case, name):
    """Single-row quantization needs a separate input set for each scenario."""
    if case["family"] != "per_token_quantization" or case["shape"][0] != 1:
        return None
    return {
        "all-zero": "zeros",
        "nonzero-random": "random",
        "rounding-boundaries": "ties",
    }[name]


def pv_program_mask(case, programs):
    """Validate a nonempty disjoint subset of real PV launch programs."""
    if case["family"] != "attention_pv":
        raise ValueError("program subsets are restricted to PV validation")
    _, m, n, _ = case["shape"]
    bm, bn = (case["constexprs"][key] for key in ("BM", "BN"))
    mask = np.zeros(argument_shapes(case)[2], dtype=bool)
    if not programs:
        raise ValueError("empty PV program selection")
    for point in programs:
        if len(point) != 3 or any(
            not isinstance(value, int) or not 0 <= value < extent
            for value, extent in zip(point, case["grid"])
        ):
            raise ValueError("out-of-grid PV program: " + str(point))
        x, y, head = point
        region = (
            head,
            slice(x * bm, min((x + 1) * bm, m)),
            slice(y * bn, min((y + 1) * bn, n)),
        )
        if np.any(mask[region]):
            raise ValueError("overlapping PV programs: " + str(point))
        mask[region] = True
    return mask


def check_pv_selected_output(case, programs, actual, expected, initial):
    """Compare authorized tiles and require all other output bytes unchanged."""
    mask = pv_program_mask(case, programs)
    bits = actual.view(np.uint32)
    if not np.array_equal(bits[mask], expected.view(np.uint32)[mask]):
        raise AssertionError("PV ordered f32 output is not bit-exact")
    if not np.array_equal(bits[~mask], initial.view(np.uint32)[~mask]):
        raise AssertionError("PV program subset wrote an unauthorized C tile")
    return mask


def audit_pv_program(case, point, accesses, dma, matrix):
    """Check this launch's exact logical rows, not only backing-buffer bounds."""
    _, m, n, k = case["shape"]
    bm, bn, bk = (case["constexprs"][key] for key in ("BM", "BN", "BK"))
    x, y, head = point
    m0, n0 = x * bm, y * bn
    rows, columns = min(bm, m - m0), min(bn, n - n0)
    regions = {
        "%arg0": (m, k, m0, m0 + rows, 0, k, "read"),
        "%arg1": (k, n, 0, k, n0, n0 + columns, "read"),
        "%arg2": (m, n, m0, m0 + rows, n0, n0 + columns, "write"),
    }
    byte_counts = dict.fromkeys(regions, 0)
    intervals = {name: [] for name in regions}
    for name, start, end, action in accesses:
        height, width, r0, r1, c0, c1, expected_action = regions[name]
        first, last = start // 4, end // 4 - 1
        row, column = divmod(first - head * height * width, width)
        last_row, last_column = divmod(last - head * height * width, width)
        if (
            start % 4
            or end % 4
            or action != expected_action
            or not r0 <= row < r1
            or last_row != row
            or not c0 <= column <= last_column < c1
        ):
            raise AssertionError(
                f"PV logical row violation at {point}: {(name, start, end, action)}"
            )
        byte_counts[name] += end - start
        intervals[name].append((start, end))
    expected_bytes = {
        "%arg0": rows * k * 4,
        "%arg1": k * columns * 4,
        "%arg2": rows * columns * 4,
    }
    if byte_counts != expected_bytes:
        raise AssertionError(
            f"PV transfer coverage at {point}: {byte_counts} != {expected_bytes}"
        )
    for name, spans in intervals.items():
        ordered = sorted(spans)
        if any(right[0] < left[1] for left, right in itertools.pairwise(ordered)):
            raise AssertionError(f"PV duplicate DDR access at {point}: {name}")
    # The current strided A cannot use the 64-byte-aligned DMA route. ACC32
    # copies move raw f32 bits; all arithmetic remains the ordered RVV path.
    geometry = {}
    if (k * 4) % 64:
        a_copies = [event for event in matrix if event["source"] == "%arg0"]
        if len(a_copies) != math.ceil(k / bk):
            raise AssertionError(f"PV A matrix-copy count at {point}: {len(a_copies)}")
        for block, event in enumerate(a_copies):
            valid = min(bk, k - block * bk)
            expected = {
                # Offsets are relative to the already offset DDR subview;
                # accesses above independently validate absolute addresses.
                "source_offset": 0,
                "rows": rows,
                "row_bytes": valid * 4,
                "source_pitch": k * 4,
                "target_pitch": bk * 4,
                "instruction_bits": 32,
                "register_class": "acc",
            }
            if any(event[key] != value for key, value in expected.items()):
                raise AssertionError(f"PV A matrix-copy geometry at {point}: {event}")
            label = (
                ("full-M" if rows == bm else "tail-M")
                + "/"
                + ("full-K" if valid == bk else "tail-K")
            )
            geometry[label] = geometry.get(label, 0) + 1
        if any(event[0] == "%arg0" for event in dma):
            raise AssertionError("PV unaligned A incorrectly used DMA")
    b_packets = [event for event in dma if event[0] == "%arg1"]
    if len(b_packets) != k or any(event[4] != columns * 4 for event in b_packets):
        raise AssertionError(f"PV B DMA packets at {point}: {len(b_packets)}")
    if any(event["source"] == "%arg1" for event in matrix):
        raise AssertionError("PV legal B DMA unexpectedly became a matrix transfer")
    return {
        "a_matrix_geometry": geometry,
        "b_dma_packets": len(b_packets),
        "ddr_bytes": byte_counts,
    }


def check_artifacts(case, output_root):
    output = Path(output_root) / case["name"]
    result_path = output / "build-result.json"
    if not result_path.is_file():
        raise SimulationBlocked("missing build-result.json; build this case first")
    result = json.loads(result_path.read_text())
    if result.get("status") != "PASS" or result.get("stage") == "ttir":
        raise SimulationBlocked(
            f"lowering {result.get('stage', 'unknown')}: {result.get('reason', 'no emitted BOSCAME module')}"
        )
    module = output / "kernel.boscame.mlir"
    manifest_path = output / "frontend.json"
    if not module.is_file() or not manifest_path.is_file():
        raise SimulationBlocked(
            "PASS build is missing emitted module or frontend manifest"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("accumulation") != case.get("accumulation"):
        raise SimulationBlocked("stale build: destination semantics differ")
    if manifest.get("kernel_module_sha256") != digest(case["kernel_source"]):
        raise SimulationBlocked(
            "stale build: current Triton source differs from frontend manifest"
        )
    for key in (
        "name",
        "family",
        "kernel",
        "kernel_module",
        "implementation",
        "symbol",
        "signature",
        "constexprs",
        "grid",
        "shape",
        "arguments",
    ):
        if manifest.get(key) != case.get(key):
            raise SimulationBlocked(
                "stale build: current case metadata differs at " + key
            )
    if is_quantized_conv(case):
        from quantized_conv_pipeline import pipeline_layout, scratch_initialization

        pipeline_manifest = output / "contract-manifest.json"
        if not pipeline_manifest.is_file():
            raise SimulationBlocked("composite build is missing its stage manifest")
        composite = json.loads(pipeline_manifest.read_text())
        if (
            composite.get("kind") != "composite-im2col-w8a8-convolution"
            or composite.get("entry") != "_mlir_ciface_kernel_" + case["name"]
            or manifest.get("pipeline_layout") != pipeline_layout(case)
            or manifest.get("scratch_initialization") != scratch_initialization(case)
            or composite.get("scratch_initialization") != scratch_initialization(case)
            or manifest.get("contract_manifest_sha256") != digest(pipeline_manifest)
            or composite.get("outputs", {}).get("contract_sha256")
            != digest(output / "kernel.frontend.mlir")
            or composite.get("outputs", {}).get("ttir_sha256")
            != digest(output / "kernel.ttir")
        ):
            raise SimulationBlocked("stale or malformed composite build manifest")
        stage_names = {stage.get("stage") for stage in manifest.get("stages", [])}
        expected_stages = {
            "weight_quantize",
            "im2col_quantize",
            "gemm_i8",
            "dequantize_bias",
        }
        if stage_names != expected_stages or len(manifest["stages"]) != len(
            expected_stages
        ):
            raise SimulationBlocked("composite build has an incomplete stage list")
        emitted_functions = set(
            re.findall(
                r"^\s*func\.func(?:\s+\w+)?\s+@([^ (]+)\(",
                module.read_text(),
                re.MULTILINE,
            )
        )
        expected_functions = {stage["symbol"] for stage in manifest["stages"]}
        if emitted_functions != expected_functions:
            raise SimulationBlocked(
                "BOSCAME module must contain exactly the four W8A8 stages; "
                f"found {sorted(emitted_functions)}"
            )
        if "triton_" + case["name"] + "(" in module.read_text():
            raise SimulationBlocked(
                "BOSCAME module contains a legacy scalar conv entry"
            )
        if composite.get("raw_entries") != [
            stage["symbol"] for stage in manifest["stages"]
        ] or len(composite.get("stages", [])) != len(expected_stages):
            raise SimulationBlocked("composite stage entries differ from frontend")
        for stage in composite["stages"]:
            for relative, expected_stage_hash in stage.get("artifacts", {}).items():
                artifact = output / relative
                if not artifact.is_file() or digest(artifact) != expected_stage_hash:
                    raise SimulationBlocked(
                        "stale composite stage artifact: " + relative
                    )
        for relative, expected_source_hash in manifest.get("source_sha256", {}).items():
            source = ROOT.parents[2] / relative
            if not source.is_file() or digest(source) != expected_source_hash:
                raise SimulationBlocked("stale composite pipeline source: " + relative)
    if is_quantized_linear(case):
        from quantized_linear_simcheck import check_pipeline

        check_pipeline(case, output)
    expected_hash = result.get("artifacts", {}).get(module.name)
    if expected_hash != digest(module):
        raise SimulationBlocked(
            "stale build: emitted module hash differs from successful build result"
        )
    ttir = output / "kernel.ttir"
    if not ttir.is_file() or manifest.get("ttir_sha256") != digest(ttir):
        raise SimulationBlocked("stale build: frontend TTIR hash differs from manifest")
    return module


def quantized_pipeline_program_counts(case):
    """Count the exact raw launches made by the generated C adapter."""
    from quantized_conv_pipeline import pipeline_layout

    layout = pipeline_layout(case)
    channels = layout["channels"]
    groups = layout["groups"]
    blocks = layout["blocks"]
    depthwise = layout["depthwise"]
    bm = 1 if depthwise else 16
    m_tiles = math.ceil(layout["matrix_rows"] / bm)
    return {
        "weight_quantize": channels,
        "im2col_quantize": blocks * groups,
        "gemm_i8": blocks * (groups if depthwise else m_tiles),
        "dequantize_bias": blocks * channels,
    }


def _composite_stage_machines(case, module, output_root, machines):
    """Create one persistent strict-simulator Machine for each raw entry."""
    output = Path(output_root) / case["name"]
    frontend = json.loads((output / "frontend.json").read_text())
    module_hash = digest(module)
    stage_machines = {}
    missing = []
    for stage in frontend["stages"]:
        key = (case["name"], module_hash, "composite-stage", stage["symbol"])
        machine = machines.get(key) if machines is not None else None
        if machine is None:
            missing.append((stage, key))
        else:
            stage_machines[stage["stage"]] = machine
    if missing:
        with tempfile.TemporaryDirectory(
            prefix="efficientnet-composite-sim-"
        ) as directory:
            loops = Path(directory) / "kernel.loops.mlir"
            try:
                process = subprocess.run(
                    [
                        tool("buddy-opt"),
                        str(module),
                        "--convert-linalg-to-loops",
                        "--lower-affine",
                        "--canonicalize",
                        "-o",
                        str(loops),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as error:
                raise SimulationBlocked(
                    "cannot invoke buddy-opt for composite simulation: " + str(error)
                ) from error
            if process.returncode:
                raise SimulationBlocked(
                    "cannot prepare composite module for strict simulation: "
                    + process.stdout.strip()
                )
            for stage, key in missing:
                machine = Machine(loops, stage["symbol"])
                for bank in machine.spm:
                    bank[:] = bytes([0xD3]) * len(bank)
                stage_machines[stage["stage"]] = machine
                if machines is not None:
                    machines[key] = machine
    return frontend, stage_machines


def _simulate_quantized_conv(
    case, arrays, output_root, progress=True, machines=None, selected_tiles=None
):
    """Run the composite im2col/W8A8 adapter in its exact stage order."""
    module = check_artifacts(case, output_root)
    output = Path(output_root) / case["name"]
    frontend = json.loads((output / "frontend.json").read_text())
    pipeline_manifest = json.loads((output / "contract-manifest.json").read_text())
    layout = frontend["pipeline_layout"]
    stage_machines_meta, stage_machines = _composite_stage_machines(
        case, module, output_root, machines
    )
    if stage_machines_meta != frontend:
        raise SimulationBlocked("composite frontend manifest changed during simulation")

    initial = {}
    for argument, shape in zip(case["arguments"], argument_shapes(case)):
        name = argument["name"]
        value = np.asarray(arrays[name])
        if value.shape != tuple(shape) or value.dtype != DTYPES[argument["dtype"]]:
            raise ValueError(
                f"{name}: expected {shape} {DTYPES[argument['dtype']]}, "
                f"got {value.shape} {value.dtype}"
            )
        initial[name] = np.ascontiguousarray(value).copy()

    expected = quantized_conv_reference(case, initial)["Out"]
    guard = bytes([0xC7]) * 1024
    public_buffers = {
        name: bytearray(value.tobytes() + guard) for name, value in initial.items()
    }
    public_lengths = {name: value.nbytes for name, value in initial.items()}
    scratch_guard = bytes([0x9B]) * 1024
    scratch = bytearray([0xA9]) * layout["total_bytes"] + bytearray(scratch_guard)
    scratch_view = memoryview(scratch)
    # Emulate the public adapter clear exactly once; keep every other byte dirty.
    init = frontend["scratch_initialization"]
    scratch_view[init["offset"] : init["offset"] + init["bytes"]] = bytes(init["bytes"])
    writable_scratch = bytearray(layout["total_bytes"])
    for item in layout.values():
        if not isinstance(item, dict) or "offset" not in item:
            continue
        writable_scratch[item["offset"] : item["offset"] + item["bytes"]] = (
            b"\x01" * item["bytes"]
        )
    stage_regions = {
        "QWeight": "qweight",
        "WeightScale": "weight_scale",
        "QPatch": "qpatch",
        "PatchScale": "patch_scale",
        "Acc": "acc",
    }
    stage_meta = {item["stage"]: item for item in frontend["stages"]}
    counts = dict.fromkeys(stage_meta, 0)
    accesses = dict.fromkeys(stage_meta, 0)
    dma_transfers = dict.fromkeys(stage_meta, 0)
    access_starts = {
        name: (len(stage_machines[name].accesses), len(stage_machines[name].dma_log))
        for name in stage_meta
    }

    def scratch_region(name, offset=0, size=None):
        field = layout[name]
        length = field["bytes"] if size is None else size
        start = field["offset"] + offset
        return scratch_view[start : start + length]

    def stage_view(stage, argument, pids, pointer_group=None):
        name = argument["name"]
        if name in public_buffers:
            return memoryview(public_buffers[name])[: public_lengths[name]]
        field = stage_regions[name]
        if name == "QWeight" and stage["stage"] == "gemm_i8" and layout["depthwise"]:
            if pointer_group is None:
                raise AssertionError("depthwise GEMM needs its host QWeight row")
            return scratch_region(field, pointer_group * 64, 64)
        return scratch_region(field)

    def launch(stage_name, pids, pointer_group=None):
        stage = stage_meta[stage_name]
        machine = stage_machines[stage_name]
        buffers = {
            f"%arg{index}": stage_view(stage, argument, pids, pointer_group)
            for index, argument in enumerate(stage["arguments"])
        }
        grid = stage["grid"]
        scalars = {
            f"%arg{len(buffers) + index}": value
            for index, value in enumerate([*grid, *pids])
        }
        try:
            machine.launch(buffers, scalars)
        except SimError as error:
            message = str(error)
            if "unsupported" in message or "only " in message:
                raise SimulationBlocked(
                    f"strict simulator in {stage_name} at {pids}: {message}"
                ) from error
            raise AssertionError(
                f"strict simulator in {stage_name} at {pids}: {message}"
            ) from error
        for name, start, end, action in machine.accesses[
            access_starts[stage_name][0] + accesses[stage_name] :
        ]:
            if name not in buffers or not 0 <= start < end <= len(buffers[name]):
                raise AssertionError(
                    f"composite stage access outside argument at {stage_name}/{pids}: "
                    f"{(name, start, end, action)}"
                )
        counts[stage_name] += 1
        accesses[stage_name] = len(machine.accesses) - access_starts[stage_name][0]
        dma_transfers[stage_name] = len(machine.dma_log) - access_starts[stage_name][1]

    weight_matrix = weight_rows(case, initial)
    expected_qweight, expected_weight_scale = quantize_rows(weight_matrix, 64)
    for row in range(layout["channels"]):
        launch("weight_quantize", (row, 0, 0))
    qweight = np.frombuffer(scratch_region("qweight"), dtype=np.int8).reshape(
        layout["channels"], 64
    )
    weight_scale = np.frombuffer(scratch_region("weight_scale"), dtype=np.float32)
    if not np.array_equal(qweight, expected_qweight) or not np.array_equal(
        weight_scale.view(np.uint32), expected_weight_scale.view(np.uint32)
    ):
        raise AssertionError("weight_quantize stage differs from quantize_rows")

    groups = layout["groups"]
    blocks = layout["blocks"]
    tile = layout["tile"]
    if selected_tiles is not None:
        selected_tiles = {tuple(point) for point in selected_tiles}
        if not selected_tiles or any(
            len(point) != 2
            or any(type(value) is not int for value in point)
            or not 0 <= point[0] < groups
            or not 0 <= point[1] < blocks
            for point in selected_tiles
        ):
            raise ValueError(
                "selected convolution tiles must be valid (group, block) pairs"
            )
    bm = stage_meta["gemm_i8"]["constexprs"]["BM"]
    m_tiles = stage_meta["gemm_i8"]["grid"][0]
    output_buffer = np.frombuffer(
        public_buffers["Out"], dtype=np.float32, count=initial["Out"].size
    ).reshape(initial["Out"].shape)
    for group in range(groups):
        for block in range(blocks):
            if selected_tiles is not None and (group, block) not in selected_tiles:
                continue
            start = block * tile
            valid = min(tile, layout["pixels"] - start)
            stage_expected = stage_reference(case, initial, group, start, tile)
            launch("im2col_quantize", (block, 0, group))
            patch_q = np.frombuffer(scratch_region("qpatch"), dtype=np.int8).reshape(
                tile, 64
            )
            patch_scale = np.frombuffer(scratch_region("patch_scale"), dtype=np.float32)
            if not np.array_equal(patch_q, stage_expected["patch_q"]):
                raise AssertionError(
                    f"im2col_quantize QPatch mismatch at group={group}, block={block}"
                )
            if not np.array_equal(
                patch_scale.view(np.uint32),
                stage_expected["patch_scale"].view(np.uint32),
            ):
                raise AssertionError(
                    f"im2col_quantize PatchScale mismatch at group={group}, block={block}"
                )

            if layout["depthwise"]:
                launch("gemm_i8", (0, 0, 0), pointer_group=group)
                accumulator = np.frombuffer(
                    scratch_region("acc"), dtype=np.int32
                ).reshape(1, tile)
                if not np.array_equal(accumulator, stage_expected["Acc"]):
                    raise AssertionError(
                        f"gemm_i8 Acc mismatch at group={group}, block={block}"
                    )
                launch("dequantize_bias", (group, block, 0))
                got = output_buffer[0, group].reshape(-1)[start : start + valid]
                want = stage_expected["Out"][0, :valid]
                if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                    raise AssertionError(
                        f"dequantize_bias Out mismatch at group={group}, block={block}"
                    )
            else:
                for m_tile in range(m_tiles):
                    launch("gemm_i8", (m_tile, 0, 0))
                    accumulator = np.frombuffer(
                        scratch_region("acc"), dtype=np.int32
                    ).reshape(bm, tile)
                    first = m_tile * bm
                    last = min(first + bm, layout["channels"])
                    if not np.array_equal(
                        accumulator[: last - first],
                        stage_expected["Acc"][first:last],
                    ):
                        raise AssertionError(
                            f"gemm_i8 Acc mismatch at M tile={m_tile}, block={block}"
                        )
                    for oc in range(first, last):
                        launch("dequantize_bias", (oc, block, m_tile))
                        got = output_buffer[0, oc].reshape(-1)[start : start + valid]
                        want = stage_expected["Out"][oc, :valid]
                        if not np.array_equal(
                            got.view(np.uint32), want.view(np.uint32)
                        ):
                            mismatch = int(
                                np.flatnonzero(
                                    got.view(np.uint32) != want.view(np.uint32)
                                )[0]
                            )
                            raise AssertionError(
                                "dequantize_bias Out mismatch at "
                                f"oc={oc}, block={block}, lane={mismatch}: "
                                f"got={got[mismatch]}, expected={want[mismatch]}"
                            )
            if progress and (
                block == 0
                or block + 1 == blocks
                or (block + 1) % max(1, blocks // 10) == 0
            ):
                print(
                    f"  {case['name']}: adapter group={group + 1}/{groups}, "
                    f"block={block + 1}/{blocks}",
                    flush=True,
                )

    actual = {}
    for name, value in initial.items():
        buffer = public_buffers[name]
        if buffer[-len(guard) :] != guard:
            raise AssertionError(name + ": DDR guard was modified")
        if name != "Out" and bytes(buffer) != value.tobytes() + guard:
            raise AssertionError(name + ": composite pipeline modified an input")
        actual[name] = (
            np.frombuffer(buffer, dtype=value.dtype, count=value.size)
            .copy()
            .reshape(value.shape)
        )
    if bytes(scratch[layout["total_bytes"] :]) != scratch_guard:
        raise AssertionError("composite workspace guard was modified")
    if any(
        value != 0xA9
        for value, allowed in zip(scratch[: layout["total_bytes"]], writable_scratch)
        if not allowed
    ):
        raise AssertionError("composite stage wrote into workspace alignment padding")
    output_mask = np.ones(expected.shape, dtype=bool)
    if selected_tiles is None:
        if not np.array_equal(actual["Out"].view(np.uint32), expected.view(np.uint32)):
            raise AssertionError("composite output differs from W8A8 stage reference")
        expected_counts = quantized_pipeline_program_counts(case)
    else:
        output_mask.fill(False)
        for group, block in selected_tiles:
            start = block * tile
            end = min(start + tile, layout["pixels"])
            if layout["depthwise"]:
                output_mask[0, group].reshape(-1)[start:end] = True
            else:
                output_mask[0].reshape(layout["channels"], -1)[:, start:end] = True
        if not np.array_equal(
            actual["Out"][output_mask].view(np.uint32),
            expected[output_mask].view(np.uint32),
        ):
            raise AssertionError("sampled composite output differs from W8A8 reference")
        if not np.array_equal(
            actual["Out"][~output_mask].view(np.uint32),
            initial["Out"][~output_mask].view(np.uint32),
        ):
            raise AssertionError("sampled composite modified an unselected output")
        tile_count = len(selected_tiles)
        expected_counts = {
            "weight_quantize": layout["channels"],
            "im2col_quantize": tile_count,
            "gemm_i8": tile_count * (1 if layout["depthwise"] else m_tiles),
            "dequantize_bias": tile_count
            * (1 if layout["depthwise"] else layout["channels"]),
        }
    if counts != expected_counts:
        raise AssertionError(
            f"composite adapter launch counts {counts} != {expected_counts}"
        )
    error_metrics = quantization_error_metrics(case, initial, actual, mask=output_mask)
    stage_evidence = {}
    for stage_name, stage in stage_meta.items():
        artifact = next(
            item for item in pipeline_manifest["stages"] if item["stage"] == stage_name
        )
        stage_evidence[stage_name] = {
            "programs": counts[stage_name],
            "ddr_accesses": accesses[stage_name],
            "dma_transfers": dma_transfers[stage_name],
            "symbol": stage["symbol"],
            "source_sha256": artifact["kernel_source_sha256"],
            "artifact_sha256": artifact["artifacts"],
        }
    total_programs = sum(counts.values())
    hashed_output = (
        actual["Out"] if selected_tiles is None else actual["Out"][output_mask]
    )
    hashed_reference = expected if selected_tiles is None else expected[output_mask]
    return actual, {
        "programs": total_programs,
        "total_programs": total_programs,
        "outputs": int(np.count_nonzero(output_mask)),
        "max_abs": 0.0,
        "module_sha256": digest(module),
        "kernel_module_sha256": digest(case["kernel_source"]),
        "contract_sha256": frontend["contract_sha256"],
        "contract_manifest_sha256": digest(output / "contract-manifest.json"),
        "order": "adapter",
        "coverage": "full" if selected_tiles is None else "sampled-tiles",
        "selected_tiles": (
            None
            if selected_tiles is None
            else [list(point) for point in sorted(selected_tiles)]
        ),
        "stage_programs": counts,
        "stages": stage_evidence,
        "quantization_error": error_metrics,
        "ddr_accesses": sum(accesses.values()),
        "dma_transfers": sum(dma_transfers.values()),
        "output_sha256": {"Out": hashlib.sha256(hashed_output.tobytes()).hexdigest()},
        "reference_sha256": {
            "Out": hashlib.sha256(hashed_reference.tobytes()).hexdigest()
        },
        "workspace_bytes": layout["total_bytes"],
        "scratch_initialization": frontend["scratch_initialization"],
    }


def simulate_case(
    case,
    arrays,
    output_root,
    order="reverse",
    progress=True,
    machines=None,
    selected_programs=None,
    selected_conv_tiles=None,
):
    """Return arrays and metrics after interpreting the requested real programs.

    ``arrays`` is a mapping from the case's argument names to NumPy arrays.
    Inputs and outputs are copied, so callers can safely retain their own data.
    The same Machine and poisoned SPM banks persist throughout this launch.
    Passing a machines dictionary also preserves them between invocations.
    Internal PV workers may select disjoint programs; unselected output bytes
    must then retain their initial values and only selected tiles are compared.
    These subsets require assemble_pv_round's independent full-grid acceptance.
    Without selected_programs, the complete real grid is executed.
    """
    if is_quantized_linear(case):
        from quantized_linear_simcheck import simulate as simulate_linear

        if selected_programs is not None or selected_conv_tiles is not None:
            raise ValueError("classifier requires its full adapter")
        return simulate_linear(case, arrays, output_root, progress, machines)
    if is_quantized_conv(case):
        if selected_programs is not None:
            raise ValueError(
                "composite convolution simulation executes its full adapter"
            )
        return _simulate_quantized_conv(
            case,
            arrays,
            output_root,
            progress=progress,
            machines=machines,
            selected_tiles=selected_conv_tiles,
        )
    module = check_artifacts(case, output_root)
    if case["family"] == "patch_embed":
        raise SimulationBlocked(
            "patch kernel materializes im2col only; projection, bias and NCHW composition have not been lowered and verified"
        )
    shapes = argument_shapes(case)
    initial = {}
    for argument, shape in zip(case["arguments"], shapes):
        name = argument["name"]
        value = np.asarray(arrays[name])
        if value.shape != tuple(shape) or value.dtype != DTYPES[argument["dtype"]]:
            raise ValueError(
                f"{name}: expected {shape} {DTYPES[argument['dtype']]}, got {value.shape} {value.dtype}"
            )
        initial[name] = np.ascontiguousarray(value).copy()
    expected = reference(case, initial)
    guard = bytes([0xC7]) * 1024
    buffers = {
        f"%arg{i}": bytearray(initial[arg["name"]].tobytes() + guard)
        for i, arg in enumerate(case["arguments"])
    }
    original = {key: bytes(data) for key, data in buffers.items()}
    limits = {key: len(data) - len(guard) for key, data in buffers.items()}
    machine_key = (case["name"], digest(module))
    machine = machines.get(machine_key) if machines is not None else None
    if machine is None:
        with tempfile.TemporaryDirectory(
            prefix="efficientnet-boscame-sim-"
        ) as directory:
            loops = Path(directory) / "kernel.loops.mlir"
            try:
                process = subprocess.run(
                    [
                        tool("buddy-opt"),
                        str(module),
                        "--convert-linalg-to-loops",
                        "--lower-affine",
                        "--canonicalize",
                        "-o",
                        str(loops),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as error:
                raise SimulationBlocked(
                    "cannot invoke buddy-opt: " + str(error)
                ) from error
            if process.returncode:
                raise SimulationBlocked(
                    "cannot prepare emitted module for strict simulation: "
                    + process.stdout.strip()
                )
            machine = Machine(loops, case["symbol"])
        for bank in machine.spm:
            bank[:] = bytes([0xD3]) * len(bank)
        if machines is not None:
            machines[machine_key] = machine
    access_start, dma_start = len(machine.accesses), len(machine.dma_log)
    grid = case["grid"]
    programs = list(itertools.product(*(range(extent) for extent in grid)))
    if selected_programs is not None:
        if case["family"] in ("gelu", "softmax"):
            nonlinear_program_mask(case, selected_programs)
        else:
            pv_program_mask(case, selected_programs)
        programs = list(selected_programs)
    if order == "reverse":
        programs.reverse()
    elif order == "tail-first":
        programs.sort(key=lambda point: (-point[0], point[1], point[2]))
    elif order != "forward":
        raise ValueError("unknown program order: " + order)
    count = len(case["arguments"])
    pv_audit = {
        "a_matrix_geometry": {},
        "b_dma_packets": 0,
        "ddr_bytes": dict.fromkeys(("%arg0", "%arg1", "%arg2"), 0),
    }
    nonlinear_bytes = {"%arg0": 0, "%arg1": 0}
    for launched, point in enumerate(programs, 1):
        scalars = {f"%arg{count + i}": value for i, value in enumerate([*grid, *point])}
        try:
            program_start = (
                len(machine.accesses),
                len(machine.dma_log),
                len(machine.matrix_log),
            )
            machine.launch(buffers, scalars)
        except SimError as error:
            message = str(error)
            if "unsupported" in message or "only " in message:
                raise SimulationBlocked(
                    f"strict simulator at program {point}: {message}"
                ) from error
            raise AssertionError(
                f"strict simulator at program {point}: {message}"
            ) from error
        if case["family"] == "attention_pv":
            access_index, dma_index, matrix_index = program_start
            audit = audit_pv_program(
                case,
                point,
                machine.accesses[access_index:],
                machine.dma_log[dma_index:],
                machine.matrix_log[matrix_index:],
            )
            pv_audit["b_dma_packets"] += audit["b_dma_packets"]
            for key in ("a_matrix_geometry", "ddr_bytes"):
                for label, value in audit[key].items():
                    pv_audit[key][label] = pv_audit[key].get(label, 0) + value
        elif case["family"] in ("gelu", "softmax"):
            audit = audit_nonlinear_program(
                case, point, machine.accesses[program_start[0] :]
            )
            for name, value in audit.items():
                nonlinear_bytes[name] += value
        if progress and (
            launched == 1
            or launched == len(programs)
            or launched % max(1, len(programs) // 10) == 0
        ):
            print(
                f"  {case['name']}: {launched}/{len(programs)} programs ({order})",
                flush=True,
            )
    writable = set(output_names(case))
    actual = {}
    for i, argument in enumerate(case["arguments"]):
        key, name = f"%arg{i}", argument["name"]
        if buffers[key][-len(guard) :] != guard:
            raise AssertionError(name + ": DDR guard was modified")
        if name not in writable and bytes(buffers[key]) != original[key]:
            raise AssertionError(name + ": input buffer was modified")
        actual[name] = (
            np.frombuffer(
                buffers[key],
                dtype=DTYPES[argument["dtype"]],
                count=initial[name].size,
            )
            .copy()
            .reshape(initial[name].shape)
        )
    for name, start, end, action in machine.accesses[access_start:]:
        if name not in limits or not 0 <= start < end <= limits[name]:
            raise AssertionError(
                f"DDR access outside logical argument: {(name, start, end, action)}"
            )
        index = int(name.removeprefix("%arg"))
        if action == "write" and case["arguments"][index]["name"] not in writable:
            raise AssertionError("write to read-only input: " + name)
    overwrite_output = {
        "matmul_i8_overwrite": "%arg2",
        "conv2d_i8": "%arg3",
    }.get(case["family"])
    if overwrite_output is not None and any(
        name == overwrite_output and action == "read"
        for name, _, _, action in machine.accesses[access_start:]
    ):
        raise AssertionError("overwrite matmul read its previous output")
    comparison_actual, comparison_expected = actual, expected
    nonlinear_metrics = None
    if case["family"] == "attention_pv":
        mask = check_pv_selected_output(
            case, programs, actual["C"], expected["C"], initial["C"]
        )
        comparison_actual = {"C": actual["C"][mask]}
        comparison_expected = {"C": expected["C"][mask]}
    elif case["family"] in ("gelu", "softmax"):
        comparison_actual, comparison_expected, nonlinear_metrics = (
            selected_nonlinear_comparison(case, programs, actual, expected, initial)
        )
    max_abs = compare(case, comparison_actual, comparison_expected)
    return actual, {
        "programs": len(programs),
        "total_programs": math.prod(grid),
        "outputs": sum(value.size for value in comparison_expected.values()),
        "max_abs": max_abs,
        "module_sha256": digest(module),
        "kernel_module_sha256": digest(case["kernel_source"]),
        "order": order,
        "ddr_accesses": len(machine.accesses) - access_start,
        "dma_transfers": len(machine.dma_log) - dma_start,
        "output_sha256": {
            name: hashlib.sha256(comparison_actual[name].tobytes()).hexdigest()
            for name in comparison_expected
        },
        **(
            {
                "pv_access_audit": pv_audit,
                "bit_exact_f32": True,
                "reference_sha256": {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in comparison_expected.items()
                },
            }
            if case["family"] == "attention_pv"
            else {}
        ),
        **(
            {
                "nonlinear_error": nonlinear_metrics,
                "nonlinear_ddr_bytes": nonlinear_bytes,
            }
            if nonlinear_metrics is not None
            else {}
        ),
        **({"selected_programs": programs} if selected_programs is not None else {}),
    }


def pv_shards(case):
    """Keep all M tiles of one (head, N tile) on the same dirty SPM."""
    return [
        [(x, y, head) for x in range(case["grid"][0])]
        for head in range(case["grid"][2])
        for y in range(case["grid"][1])
    ]


def pv_shard_worker(case, output_root, programs):
    """Interpret all three real rounds, retaining one Machine across them."""
    machines = {}
    mask = pv_program_mask(case, programs)
    runs = []
    for name, stress, order in scenarios_for(case):
        arrays = make_inputs(case, argument_shapes(case), stress=stress)
        actual, metrics = simulate_case(
            case,
            arrays,
            output_root,
            order=order,
            progress=False,
            machines=machines,
            selected_programs=programs,
        )
        if len(machines) != 1:
            raise AssertionError("PV shard did not retain one Machine/SPM")
        runs.append(
            {
                "name": name,
                "metrics": metrics,
                "values": actual["C"][mask].copy(),
            }
        )
    return {"programs": programs, "runs": runs}


def assemble_pv_round(case, pieces, expected):
    """Reconstruct every output exclusively from disjoint executed slices."""
    shape = argument_shapes(case)[2]
    assembled = np.empty(shape, dtype=DTYPES["f32"])
    covered = np.zeros(shape, dtype=bool)
    programs = set()
    for piece in pieces:
        points = piece["programs"]
        mask = pv_program_mask(case, points)
        point_set = {tuple(point) for point in points}
        if programs & point_set or np.any(covered & mask):
            raise AssertionError(
                "PV aggregate contains duplicate program/output coverage"
            )
        metrics, values = piece["metrics"], piece["values"]
        if (
            metrics["programs"] != len(points)
            or metrics["outputs"] != np.count_nonzero(mask)
            or {tuple(point) for point in metrics["selected_programs"]} != point_set
            or len(metrics["selected_programs"]) != len(points)
            or values.shape != (np.count_nonzero(mask),)
            or values.dtype != DTYPES["f32"]
            or hashlib.sha256(values.tobytes()).hexdigest()
            != metrics["output_sha256"]["C"]
            or metrics.get("bit_exact_f32") is not True
            or metrics.get("reference_sha256") != metrics["output_sha256"]
        ):
            raise AssertionError("PV aggregate lacks executed slice/hash evidence")
        assembled[mask] = values
        covered |= mask
        programs |= point_set
    complete = set(itertools.product(*(range(extent) for extent in case["grid"])))
    if programs != complete or not np.all(covered):
        raise AssertionError("PV aggregate is missing full-grid programs/outputs")
    if not np.array_equal(assembled.view(np.uint32), expected.view(np.uint32)):
        raise AssertionError("PV aggregate ordered f32 output is not bit-exact")
    return assembled


def check_pv_sharded(case, output_root, workers):
    if case["family"] != "attention_pv" or workers <= 1:
        raise ValueError("parallel shards require PV and at least two workers")
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]) * 3,
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "execution": "parallel-shards",
        "order_scope": "within each fixed (head, N tile) shard; no global inter-shard order",
        "spm_lifetime": "one Machine per shard, poisoned once and retained across all selected programs and three rounds",
        "comparison_rounds": 3,
        "full_grid_programs": math.prod(case["grid"]),
    }
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        record["kernel_module_sha256"] = digest(case["kernel_source"])
        shards = pv_shards(case)
        record.update(workers=min(workers, len(shards)), shard_count=len(shards))
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=record["workers"],
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(pv_shard_worker, case, output_root, points)
                for points in shards
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"  {case['name']}: {len(results)}/{len(shards)} shards, three rounds each",
                    flush=True,
                )
        results.sort(key=lambda result: result["programs"][0])
        runs = []
        for index, (name, stress, order) in enumerate(scenarios_for(case)):
            expected = reference(
                case, make_inputs(case, argument_shapes(case), stress=stress)
            )["C"]
            pieces = [
                {"programs": result["programs"], **result["runs"][index]}
                for result in results
            ]
            assembled = assemble_pv_round(case, pieces, expected)
            audit = {
                "a_matrix_geometry": {},
                "b_dma_packets": 0,
                "ddr_bytes": dict.fromkeys(("%arg0", "%arg1", "%arg2"), 0),
            }
            for piece in pieces:
                metrics = piece["metrics"]
                if (
                    metrics["module_sha256"] != digest(module)
                    or metrics["kernel_module_sha256"] != digest(case["kernel_source"])
                    or piece["name"] != name
                    or metrics["order"] != order
                ):
                    raise AssertionError("PV shard source/module/scenario differs")
                per_shard = metrics["pv_access_audit"]
                audit["b_dma_packets"] += per_shard["b_dma_packets"]
                for key in ("a_matrix_geometry", "ddr_bytes"):
                    for label, value in per_shard[key].items():
                        audit[key][label] = audit[key].get(label, 0) + value
            hashes = {"C": hashlib.sha256(assembled.tobytes()).hexdigest()}
            runs.append(
                {
                    "name": name,
                    "order": "shard-" + order,
                    "programs": math.prod(case["grid"]),
                    "total_programs": math.prod(case["grid"]),
                    "outputs": assembled.size,
                    "max_abs": 0.0,
                    "bit_exact_f32": True,
                    "module_sha256": digest(module),
                    "kernel_module_sha256": digest(case["kernel_source"]),
                    "output_sha256": hashes,
                    "reference_sha256": {
                        "C": hashlib.sha256(expected.tobytes()).hexdigest()
                    },
                    "ddr_accesses": sum(
                        piece["metrics"]["ddr_accesses"] for piece in pieces
                    ),
                    "dma_transfers": sum(
                        piece["metrics"]["dma_transfers"] for piece in pieces
                    ),
                    "pv_access_audit": audit,
                }
            )
        if runs[1]["output_sha256"] != runs[2]["output_sha256"]:
            raise AssertionError("PV shard launch order changed output")
        record.update(
            status="PASS",
            runs=runs,
            programs=sum(run["programs"] for run in runs),
            outputs=sum(run["outputs"] for run in runs),
            max_abs=0.0,
            shards=[
                {
                    "programs": result["programs"],
                    "runs": [
                        {"name": run["name"], **run["metrics"]}
                        for run in result["runs"]
                    ],
                }
                for result in results
            ],
        )
    except SimulationBlocked as error:
        record["reason"] = str(error)
    except Exception as error:  # noqa: BLE001 - record each case's simulator failure
        record.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    (output / "simcheck-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{record['status']}: {case['name']} ({record['programs']}/{record['total_programs']}; sharded)",
        flush=True,
    )
    return record


def validate_pv_shard_evidence(case, saved, module_hash):
    """Reject incomplete/overlapping stored shard metadata during collection."""
    shards = saved.get("shards", [])
    if len(shards) != len(pv_shards(case)) or saved.get("shard_count") != len(shards):
        raise SimulationBlocked("incomplete PV shard count")
    references = {
        stress: reference(
            case, make_inputs(case, argument_shapes(case), stress=stress)
        )["C"]
        for stress in (False, True)
    }
    fixed_shards = {frozenset(points) for points in pv_shards(case)}
    points_seen = set()
    for shard in shards:
        points = shard.get("programs", [])
        try:
            mask = pv_program_mask(case, points)
        except ValueError as error:
            raise SimulationBlocked("invalid PV shard program selection") from error
        point_set = {tuple(point) for point in points}
        if frozenset(point_set) not in fixed_shards:
            raise SimulationBlocked(
                "PV shard is not one complete fixed (head, N tile) M set"
            )
        if points_seen & point_set:
            raise SimulationBlocked("duplicate PV shard programs")
        points_seen |= point_set
        runs = shard.get("runs", [])
        if len(runs) != 3:
            raise SimulationBlocked("incomplete PV shard rounds")
        for run, (name, stress, order) in zip(runs, scenarios_for(case)):
            expected = references[stress][mask]
            expected_hash = {"C": hashlib.sha256(expected.tobytes()).hexdigest()}
            ordered_points = sorted(point_set)
            if order == "reverse":
                ordered_points.reverse()
            elif order == "tail-first":
                ordered_points.sort(key=lambda point: (-point[0], point[1], point[2]))
            if (
                run.get("name") != name
                or run.get("order") != order
                or run.get("module_sha256") != module_hash
                or run.get("kernel_module_sha256") != digest(case["kernel_source"])
                or run.get("programs") != len(points)
                or run.get("outputs") != np.count_nonzero(mask)
                or len(run.get("selected_programs", [])) != len(points)
                or {tuple(point) for point in run["selected_programs"]} != point_set
                or [tuple(point) for point in run["selected_programs"]]
                != ordered_points
                or run.get("bit_exact_f32") is not True
                or run.get("output_sha256") != expected_hash
                or run.get("reference_sha256") != expected_hash
                or "pv_access_audit" not in run
            ):
                raise SimulationBlocked(
                    "incomplete PV shard numerical/coverage evidence"
                )
    complete = set(itertools.product(*(range(extent) for extent in case["grid"])))
    if points_seen != complete:
        raise SimulationBlocked("PV shards do not cover the full grid")


def check_case(case, output_root):
    if case["family"] in ("gelu", "softmax"):
        return check_nonlinear_sharded(case, output_root, 1)
    full_grid = (
        sum(quantized_pipeline_program_counts(case).values())
        if is_quantized_conv(case)
        else sum(linear_program_counts(case).values())
        if is_quantized_linear(case)
        else math.prod(case["grid"])
    )
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": full_grid,
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "module_sha256": None,
        "full_grid_programs": full_grid,
    }
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        record["kernel_module_sha256"] = digest(case["kernel_source"])
        scenarios = scenarios_for(case)
        record["total_programs"] *= len(scenarios)
        record["comparison_rounds"] = len(scenarios)
        runs = []
        machines = (
            {}
            if case["family"] == "attention_pv"
            or is_quantized_conv(case)
            or is_quantized_linear(case)
            else None
        )
        for name, stress, order in scenarios:
            arrays = make_inputs(
                case,
                argument_shapes(case),
                stress=stress,
                pattern=quantize_pattern(case, name),
            )
            _, metrics = simulate_case(
                case, arrays, output_root, order=order, machines=machines
            )
            runs.append({"name": name, **metrics})
            record["programs"] += metrics["programs"]
            record["outputs"] += metrics["outputs"]
            record["max_abs"] = max(record["max_abs"] or 0, metrics["max_abs"])
        repeated = (
            len(scenarios) > 2
            and scenarios[1][1]
            and scenarios[1][1] == scenarios[2][1]
        )
        if repeated and runs[1]["output_sha256"] != runs[2]["output_sha256"]:
            raise AssertionError(
                "identical stress inputs depend on program launch order"
            )
        record.update(status="PASS", runs=runs)
    except SimulationBlocked as error:
        record["reason"] = str(error)
    except Exception as error:  # noqa: BLE001 - record each case's simulator failure
        record.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    (output / "simcheck-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{record['status']}: {case['name']} ({record['programs']}/{record['total_programs']})",
        flush=True,
    )
    return record


def collect_case(case, output_root):
    """Reuse only complete simulation evidence for the current emitted module."""
    if case["family"] in ("gelu", "softmax"):
        return collect_nonlinear_case(case, output_root)
    path = Path(output_root) / case["name"] / "simcheck-result.json"
    record = {
        "name": case["name"],
        "status": "BLOCKED",
        "programs": 0,
        "total_programs": math.prod(case["grid"]),
        "outputs": 0,
        "max_abs": None,
        "reason": "",
        "module_sha256": None,
    }
    try:
        module = check_artifacts(case, output_root)
        record["module_sha256"] = digest(module)
        if not path.is_file():
            raise SimulationBlocked("no simcheck-result.json for this successful build")
        saved = json.loads(path.read_text())
        if saved.get("module_sha256") != digest(module):
            raise SimulationBlocked("stale simulation: emitted module hash changed")
        if saved.get("status") != "PASS":
            return saved
        scenarios = scenarios_for(case)
        rounds = len(scenarios)
        full_grid = (
            sum(quantized_pipeline_program_counts(case).values())
            if is_quantized_conv(case)
            else sum(linear_program_counts(case).values())
            if is_quantized_linear(case)
            else math.prod(case["grid"])
        )
        runs = saved.get("runs", [])
        outputs = set(output_names(case))
        per_round = sum(
            math.prod(shape)
            for arg, shape in zip(case["arguments"], argument_shapes(case))
            if arg["name"] in outputs
        )
        if (
            saved.get("name") != case["name"]
            or len(runs) != rounds
            or saved.get("programs") != full_grid * rounds
            or saved.get("total_programs") != full_grid * rounds
            or saved.get("outputs") != per_round * rounds
        ):
            raise SimulationBlocked(
                "incomplete simulation: complete grid/round/output evidence is missing"
            )
        for run in runs:
            if (
                run.get("programs") != full_grid
                or run.get("total_programs") != full_grid
                or run.get("outputs") != per_round
                or run.get("module_sha256") != digest(module)
                or set(run.get("output_sha256", {})) != outputs
            ):
                raise SimulationBlocked(
                    "incomplete simulation: a round lacks complete current-module evidence"
                )
            if is_quantized_conv(case):
                counts = quantized_pipeline_program_counts(case)
                frontend = json.loads(
                    (Path(output_root) / case["name"] / "frontend.json").read_text()
                )
                stages = run.get("stages", {})
                if (
                    run.get("order") != "adapter"
                    or run.get("stage_programs") != counts
                    or run.get("contract_sha256") != frontend.get("contract_sha256")
                    or set(stages) != set(counts)
                ):
                    raise SimulationBlocked(
                        "incomplete composite stage/program evidence"
                    )
                expected_stage_hashes = {
                    stage["stage"]: stage["kernel_source_sha256"]
                    for stage in frontend["stages"]
                }
                for name, count in counts.items():
                    stage = stages[name]
                    if (
                        stage.get("programs") != count
                        or stage.get("source_sha256") != expected_stage_hashes[name]
                        or not stage.get("artifact_sha256")
                    ):
                        raise SimulationBlocked(
                            "incomplete composite evidence for stage " + name
                        )
            if is_quantized_linear(case):
                counts = linear_program_counts(case)
                frontend = json.loads(
                    (Path(output_root) / case["name"] / "frontend.json").read_text()
                )
                if (
                    run.get("order") != "adapter"
                    or run.get("stage_programs") != counts
                    or run.get("contract_sha256") != frontend["contract_sha256"]
                    or run.get("bit_exact_w8a8") is not True
                    or run.get("reference_sha256") != run.get("output_sha256")
                    or not run.get("fp32_error", {}).get("within_quantization_bound")
                ):
                    raise SimulationBlocked("incomplete W8A8 classifier evidence")
                expected_sources = {
                    s["stage"]: s["kernel_source_sha256"] for s in frontend["stages"]
                }
                if set(run.get("stages", {})) != set(counts):
                    raise SimulationBlocked("incomplete classifier stage list")
                for name, count in counts.items():
                    stage = run["stages"][name]
                    if (
                        stage.get("programs") != count
                        or stage.get("source_sha256") != expected_sources[name]
                        or stage.get("artifact_sha256") != digest(module)
                    ):
                        raise SimulationBlocked("incomplete classifier stage " + name)
            if case["family"] == "attention_pv" and (
                run.get("bit_exact_f32") is not True
                or run.get("reference_sha256") != run.get("output_sha256")
                or "pv_access_audit" not in run
            ):
                raise SimulationBlocked(
                    "incomplete PV bit-exact/transfer-geometry evidence"
                )
        order_prefix = ""
        if saved.get("execution") == "parallel-shards":
            validate_pv_shard_evidence(case, saved, digest(module))
            order_prefix = "shard-"
        repeated_stress = (
            rounds > 2 and scenarios[1][1] and scenarios[1][1] == scenarios[2][1]
        )
        if [(run.get("name"), run.get("order")) for run in runs] != [
            (name, order_prefix + order) for name, _, order in scenarios
        ] or (repeated_stress and runs[1]["output_sha256"] != runs[2]["output_sha256"]):
            raise SimulationBlocked("incomplete numerical/launch-order stress evidence")
        saved.update(
            kernel_module_sha256=digest(case["kernel_source"]),
            full_grid_programs=full_grid,
            comparison_rounds=rounds,
        )
        return saved
    except SimulationBlocked as error:
        record["reason"] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Combine existing results after verifying current source/module hashes and complete grid/round coverage; do not rerun simulations",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--pv-workers",
        type=int,
        default=1,
        help="explicitly interpret PV full-grid (head,N-tile) shards in separate processes; each retains dirty SPM across its three rounds",
    )
    parser.add_argument(
        "--nonlinear-workers",
        type=int,
        default=1,
        help="interpret GELU/Softmax program shards with one dirty-SPM Machine per shard across three rounds",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.all and args.cases:
        parser.error("use --all or --case, not both")
    if not args.all and not args.cases:
        parser.error("choose --all or at least one --case")
    if args.report is None:
        args.report = summary_report_path(args.output, args.cases, "simcheck")
    cases = (
        [describe(name) for name in dict.fromkeys(args.cases)]
        if args.cases
        else inventory()
    )
    if args.pv_workers < 1 or (
        args.pv_workers > 1 and any(case["family"] != "attention_pv" for case in cases)
    ):
        parser.error("--pv-workers requires positive workers and PV-only cases")
    if args.nonlinear_workers < 1 or (
        args.nonlinear_workers > 1
        and any(case["family"] not in ("gelu", "softmax") for case in cases)
    ):
        parser.error(
            "--nonlinear-workers requires positive workers and GELU/Softmax-only cases"
        )
    if args.collect:
        run = collect_case
    elif all(case["family"] in ("gelu", "softmax") for case in cases):

        def run(case, output):
            return check_nonlinear_sharded(case, output, args.nonlinear_workers)
    elif args.pv_workers > 1:

        def run(case, output):
            return check_pv_sharded(case, output, args.pv_workers)
    else:
        run = check_case
    results = [run(case, args.output) for case in cases]
    if args.collect:
        for result in results:
            directory = args.output / result["name"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "simcheck-result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                f"{result['status']}: {result['name']} ({result['programs']}/{result['total_programs']}; collected)"
            )
    status = (
        "FAIL"
        if any(result["status"] == "FAIL" for result in results)
        else "PARTIAL"
        if any(result["status"] == "BLOCKED" for result in results)
        else "PASS"
    )
    report = {"schema": 1, "status": status, "cases": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
