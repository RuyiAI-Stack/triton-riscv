"""Execute every real W8A8 classifier stage with guarded shared buffers."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from quantized_linear_oracle import fp32_error
from quantized_linear_oracle import stages as reference_stages
from quantized_linear_pipeline import pipeline_layout, stage_cases


def program_counts(case):
    return {s["stage"]: int(np.prod(s["grid"])) for s in stage_cases(case)}


def check_pipeline(case, output):
    from export import ROOT, digest

    output = Path(output)
    front = json.loads((output / "frontend.json").read_text())
    contract = json.loads((output / "contract-manifest.json").read_text())
    if (
        contract.get("kind") != "composite-w8a8-linear"
        or front.get("pipeline_layout") != pipeline_layout(case)
        or front.get("contract_manifest_sha256")
        != digest(output / "contract-manifest.json")
        or contract["outputs"]["contract_sha256"]
        != digest(output / "kernel.frontend.mlir")
        or contract["outputs"]["ttir_sha256"] != digest(output / "kernel.ttir")
    ):
        raise ValueError("stale W8A8 classifier manifest")
    expected = stage_cases(case)
    if [s["stage"] for s in front["stages"]] != [s["stage"] for s in expected]:
        raise ValueError("incomplete classifier stages")
    text = (output / "kernel.boscame.mlir").read_text()
    functions = set(
        re.findall(r"^\s*func\.func(?:\s+\w+)?\s+@([^ (]+)\(", text, re.MULTILINE)
    )
    if functions != {s["symbol"] for s in expected}:
        raise ValueError("unexpected or legacy classifier function")
    gemm = next(s for s in expected if s["stage"] == "gemm_i8")
    block = text[text.index("@" + gemm["symbol"]) :]
    block = block.split("  func.func", 1)[0]
    if "tile.mma" not in block or "f32" in block:
        raise ValueError("classifier GEMM is not integer MMA")
    for relative, sha in front["source_sha256"].items():
        if digest(ROOT.parents[2] / relative) != sha:
            raise ValueError("stale classifier source: " + relative)
    for s in contract["stages"]:
        for relative, sha in s["artifacts"].items():
            if digest(output / relative) != sha:
                raise ValueError("stale stage: " + relative)
    return front


def simulate(case, arrays, output_root, progress=True, machines=None):
    from export import argument_shapes, digest
    from simcheck import _composite_stage_machines, check_artifacts

    module = check_artifacts(case, output_root)
    frontend, stage_machines = _composite_stage_machines(
        case, module, output_root, machines
    )
    initial = {}
    for arg, shape in zip(case["arguments"], argument_shapes(case)):
        value = np.asarray(arrays[arg["name"]])
        if value.dtype != np.float32 or value.shape != tuple(shape):
            raise ValueError("classifier argument type or shape differs")
        initial[arg["name"]] = np.ascontiguousarray(value).copy()
    want = reference_stages(initial)
    guard = bytes([0xC7]) * 1024
    public = {k: bytearray(v.tobytes() + guard) for k, v in initial.items()}
    g = pipeline_layout(case)
    scratch = bytearray([0xA9]) * g["total_bytes"] + bytearray(guard)
    regions = {
        "QA": "qa",
        "QW": "qw",
        "RowScale": "rs",
        "WeightScale": "ws",
        "Acc": "acc",
    }

    def region(name):
        v = g[regions[name]]
        return memoryview(scratch)[v["offset"] : v["offset"] + v["bytes"]]

    counts = {}
    metrics = {}
    total_accesses = 0
    total_dma = 0
    for stage in frontend["stages"]:
        name = stage["stage"]
        machine = stage_machines[name]
        buffers = {
            f"%arg{i}": (
                memoryview(public[a["name"]])[: initial[a["name"]].nbytes]
                if a["name"] in public
                else region(a["name"])
            )
            for i, a in enumerate(stage["arguments"])
        }
        access_start = len(machine.accesses)
        dma_start = len(machine.dma_log)
        written = set()
        calls = 0
        for pids in itertools.product(*(range(v) for v in stage["grid"])):
            scalars = {
                f"%arg{len(buffers) + i}": v
                for i, v in enumerate([*stage["grid"], *pids])
            }
            machine.launch(buffers, scalars)
            calls += 1
        writable = {
            "activation_quantize": {"QA", "RowScale"},
            "weight_quantize": {"QW", "WeightScale"},
            "zero_acc": {"Acc"},
            "gemm_i8": {"Acc"},
            "dequantize_bias": {"Out"},
        }[name]
        for key, start, end, action in machine.accesses[access_start:]:
            if key not in buffers or not 0 <= start < end <= len(buffers[key]):
                raise AssertionError("classifier access outside logical argument")
            if action == "write":
                arg = stage["arguments"][int(key[4:])]["name"]
                if arg not in writable:
                    raise AssertionError("classifier wrote read-only input")
                written.add(arg)
        if written != writable:
            raise AssertionError("classifier missing output writes")
        for field in writable:
            if field == "Out":
                got = np.frombuffer(
                    public[field], np.float32, count=initial[field].size
                ).reshape(initial[field].shape)
                expected = want[field]
            else:
                expected = (
                    want[field] if name != "zero_acc" else np.zeros_like(want[field])
                )
                got = np.frombuffer(region(field), dtype=expected.dtype).reshape(
                    expected.shape
                )
            if got.tobytes() != expected.tobytes():
                raise AssertionError("classifier stage mismatch: " + name + "/" + field)
        counts[name] = calls
        metrics[name] = {
            "programs": calls,
            "source_sha256": stage["kernel_source_sha256"],
            "artifact_sha256": digest(module),
            "ddr_accesses": len(machine.accesses) - access_start,
        }
        total_accesses += len(machine.accesses) - access_start
        total_dma += len(machine.dma_log) - dma_start
        if progress:
            print(f"  {case['name']}: {name} {calls} programs PASS", flush=True)
    for name, value in initial.items():
        if public[name][-len(guard) :] != guard or (
            name != "Out" and public[name][: -len(guard)] != value.tobytes()
        ):
            raise AssertionError("classifier input or guard modified")
    if scratch[-len(guard) :] != guard:
        raise AssertionError("classifier scratch guard modified")
    actual = {
        **initial,
        "Out": np.frombuffer(public["Out"], np.float32, count=initial["Out"].size)
        .copy()
        .reshape(initial["Out"].shape),
    }
    sha = hashlib.sha256(actual["Out"].tobytes()).hexdigest()
    return actual, {
        "programs": sum(counts.values()),
        "total_programs": sum(counts.values()),
        "outputs": actual["Out"].size,
        "max_abs": 0.0,
        "module_sha256": digest(module),
        "kernel_module_sha256": digest(case["kernel_source"]),
        "contract_sha256": frontend["contract_sha256"],
        "order": "adapter",
        "stage_programs": counts,
        "stages": metrics,
        "ddr_accesses": total_accesses,
        "dma_transfers": total_dma,
        "output_sha256": {"Out": sha},
        "reference_sha256": {"Out": sha},
        "bit_exact_w8a8": True,
        "fp32_error": fp32_error(initial, actual["Out"], want),
    }
