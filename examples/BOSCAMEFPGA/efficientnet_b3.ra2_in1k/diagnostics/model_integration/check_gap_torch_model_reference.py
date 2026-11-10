#!/usr/bin/env python3
"""CPU-only explicit GAP arithmetic-order change exploration. Never changes golden."""

from pathlib import Path
from collections import Counter
import sys
import json
import ctypes
import hashlib
import time
import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "diagnostics/model_integration"))
import reference as primary  # noqa: E402
import argparse  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--output",
    type=Path,
    default=R / "build/model-integration/gap-torch-order-candidate/model-reference",
)
p.add_argument(
    "--gap-library",
    type=Path,
    default=R / "build/model-integration/gap-torch-order-candidate/host.so",
)
p.add_argument(
    "--unary-library",
    type=Path,
    default=R / "build/diagnostics/checked-source-target-order-reference/host-math.so",
)
p.add_argument("--dump-node", action="append", default=[])
a = p.parse_args()
B = R / "build/model-integration"
out_dir = a.output.resolve()
out_dir.mkdir(parents=True, exist_ok=True)
L = ctypes.CDLL(str(a.unary_library.resolve()))
for name in ["target_reference_silu", "target_reference_sigmoid"]:
    getattr(L, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
L.target_reference_fnv.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
L.target_reference_fnv.restype = ctypes.c_uint64
G = ctypes.CDLL(str(a.gap_library.resolve())).efficientnet_gap_torch_order_candidate
G.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
torch.set_num_threads(4)


def sha(x):
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


graph = json.loads((B / "shape-graph-manifest.json").read_text())
model, _ = primary.load_model(B / "assets")
modules = dict(model.named_modules())
archive = np.load(B / "reference/folded-and-quantized-parameters.npz")
params = {
    p["node"]: {
        f: torch.from_numpy(archive[p["node"] + "." + f].copy())
        for f in ["weight", "bias", "qweight", "weight_scale"]
    }
    for p in graph["parameters"]
}
image = torch.from_numpy(
    np.fromfile(B / "physical-plan/input.f32", np.float32)
    .reshape(graph["input"]["shape"])
    .copy()
)
golden = torch.from_numpy(
    np.fromfile(B / "reference/w8a8-logits.f32", np.float32)
    .reshape(graph["output"]["shape"])
    .copy()
)
pt = json.loads((B / "reference/w8a8-node-trace.json").read_text())
pt = pt["nodes"] if isinstance(pt, dict) and "nodes" in pt else pt
primary_hashes = {
    p["node"] if "node" in p else p["name"]: p["output_sha256"] for p in pt
}
physical = json.loads((B / "physical-plan/physical-plan.json").read_text())
calls_by_node = {}
for call in physical["calls"]:
    calls_by_node.setdefault(call["logical_node"], []).append(call["index"])
uses = Counter(k for n in graph["nodes"] for k in n["inputs"])
uses[graph["output"]["value"]] += 1
values = {"input": image}
trace = []
gaps = []
begin = time.monotonic()
for n in graph["nodes"]:
    x = [values[k] for k in n["inputs"]]
    name = n["name"]
    kind = n["kind"]
    if kind in ["conv", "depthwise"]:
        out = primary.w8a8_conv(x[0], modules[name], params[name])
    elif kind in ["silu", "sigmoid"]:
        out = torch.empty_like(x[0])
        getattr(L, "target_reference_" + kind)(
            x[0].data_ptr(), out.data_ptr(), out.numel()
        )
    elif kind == "gap":
        _, ch, h, w = x[0].shape
        out = torch.empty(n["shape"])
        G(x[0].data_ptr(), out.data_ptr(), h * w, ch, 0)
        original = x[0].mean((2, 3), keepdim=True)
        bad = int(
            np.count_nonzero(
                out.numpy().view(np.uint32) != original.numpy().view(np.uint32)
            )
        )
        gaps.append(
            {
                "node": name,
                "shape": list(x[0].shape),
                "elements": out.numel(),
                "bit_mismatches_vs_torch": bad,
                "maxabs": float((out - original).abs().max()),
            }
        )
    elif kind == "se_broadcast_multiply":
        out = x[0] * x[1]
    elif kind == "add":
        out = x[0] + x[1]
    elif kind == "reshape_view":
        out = x[0].reshape(n["shape"])
    elif kind == "linear_w8a8_chain":
        p = params[name]
        qx, xs = primary.quantize_rows(x[0])
        acc = primary.integer_mm(qx, p["qweight"].T)
        out = acc.float() * xs[:, None]
        out = out * p["weight_scale"][None, :]
        out = out + p["bias"][None, :]
    else:
        raise ValueError(kind)
    out = out.float().contiguous()
    trace.append(
        {
            "node": name,
            "kind": kind,
            "shape": list(out.shape),
            "output_sha256": sha(out),
            "output_fnv64": f"{L.target_reference_fnv(out.data_ptr(), out.numel() * 4):016x}",
            "physical_call_indices": calls_by_node.get(name, []),
            "final_physical_call": calls_by_node.get(name, [None])[-1],
            "matches_primary_node": sha(out) == primary_hashes[name],
        }
    )
    values[n["output"]] = out
    if name in a.dump_node:
        dump = out_dir / "node-values"
        dump.mkdir(exist_ok=True)
        arrays = {
            "output": out.numpy(),
            **{"input_" + str(i): v.numpy() for i, v in enumerate(x)},
        }
        if name in params:
            arrays.update(
                {field: value.numpy() for field, value in params[name].items()}
            )
        np.savez(dump / f"{len(trace) - 1:03d}.npz", **arrays)
        out.numpy().tofile(dump / f"{len(trace) - 1:03d}.f32")
        (dump / f"{len(trace) - 1:03d}.json").write_text(
            json.dumps(trace[-1], indent=2) + "\n"
        )
    for k in n["inputs"]:
        uses[k] -= 1
        if uses[k] == 0:
            del values[k]
    print(
        len(trace), name, "primarymatch", trace[-1]["matches_primary_node"], flush=True
    )
final = values[graph["output"]["value"]]
diff = (final - golden).abs()
report = {
    "scope": "CPU-only explicit change from original GAP order to pinned Torch AVX2 8-lane 4-ILP 4-level order; no production edits and no FPGA evidence",
    "torch": torch.__version__,
    "all_27_actual_gaps_bitexact_torch": len(gaps) == 27
    and all(p["bit_mismatches_vs_torch"] == 0 for p in gaps),
    "gap_evidence": gaps,
    "logical_nodes": len(trace),
    "primary_node_hash_matches": sum(n["matches_primary_node"] for n in trace),
    "output_sha256": sha(final),
    "primary_golden_sha256": sha(golden),
    "output_bit_mismatches": int(
        np.count_nonzero(
            final.numpy().view(np.uint32) != golden.numpy().view(np.uint32)
        )
    ),
    "maxabs": float(diff.max()),
    "meanabs": float(diff.mean()),
    "top5": torch.topk(final.flatten(), 5).indices.tolist(),
    "host_seconds": time.monotonic() - begin,
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__).with_name("gap_torch_order_candidate.c"),
            a.gap_library,
            a.unary_library,
            Path(__file__),
            B / "reference/w8a8-logits.f32",
        ]
    },
}
(out_dir / "whole-graph-node-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
(out_dir / "whole-graph-report.json").write_text(json.dumps(report, indent=2) + "\n")
final.numpy().tofile(out_dir / "explicit-torch-gap-order-logits.f32")
print(json.dumps(report, indent=2))
