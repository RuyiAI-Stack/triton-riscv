#!/usr/bin/env python3
"""Reproduce separately labeled CPU target-order semantics without changing goldens.

Example (paths relative to the EfficientNet example):
  python diagnostics/model_integration/target_semantic_reference.py \
    --plan-dir build/model-integration/physical-plan \
    --assets build/model-integration/assets \
    --graph build/model-integration/shape-graph-manifest.json \
    --primary-reference build/model-integration/reference \
    --kernel-root build/kernels \
    --silu-source diagnostics/model_integration/silu_rvv_candidate.c \
    --silu-symbol efficientnet_silu_rvv_candidate \
    --gap-source diagnostics/model_integration/gap_phase_candidate.c \
    --output build/model-integration/target-order-cpu-reference

This executes native host C arithmetic and exact integer W8A8 graph semantics.
It does not execute FPGA instructions or model cache/transaction behavior.
Every original emitted GAP LLVM IR is host compiled and checked on the actual
graph inputs. The original Torch reference is independently reproduced first.
"""

import argparse
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
import cases  # noqa: E402
import reference as primary  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args, log):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    log.write_text(
        json.dumps(list(map(str, args))) + "\n" + result.stdout + result.stderr
    )
    if result.returncode:
        raise RuntimeError("command failed: " + str(log))
    return result.stdout


class Memref(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


class Unranked(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int64), ("descriptor", ctypes.c_void_p)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in (
        "plan-dir",
        "assets",
        "graph",
        "primary-reference",
        "kernel-root",
        "silu-source",
        "gap-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--silu-symbol", required=True)
    parser.add_argument(
        "--llvm-clang", type=Path, default=REPO / "buddy-mlir/llvm/build/bin/clang"
    )
    parser.add_argument("--host-cc", default="cc")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--dump-node",
        action="append",
        default=[],
        help="also export this target-order logical node and its inputs/parameters as NPZ",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    for protected in (args.plan_dir, args.assets, args.primary_reference):
        if args.output == protected or protected in args.output.parents:
            raise ValueError(
                "--output must be separate from original plan/assets/reference directories"
            )
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", args.silu_symbol):
        raise ValueError("invalid SiLU symbol")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = args.output / "source-snapshots"
    snapshots.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    tracked = [
        args.graph,
        args.plan_dir / "physical-plan.json",
        args.plan_dir / "parameters.json",
        args.plan_dir / "parameters.bin",
        args.plan_dir / "input.f32",
        args.primary_reference / "w8a8-logits.f32",
        args.primary_reference / "w8a8-node-trace.json",
        args.primary_reference / "folded-and-quantized-parameters.npz",
        args.assets / "model.safetensors",
        args.assets / "config.json",
        args.silu_source,
        args.gap_source,
        HERE / "reference.py",
        Path(__file__),
    ]
    before_hashes = {str(path): sha(path) for path in tracked}
    for source, name in [
        (args.silu_source, "silu.c"),
        (args.gap_source, "gap.c"),
        (HERE / "reference.py", "primary-reference.py"),
        (Path(__file__), "target-semantic-reference.py"),
    ]:
        shutil.copy2(source, snapshots / name)
    math_dir = REPO / "buddy-mlir/midend/include/Conversion/LowerLinalgToBOSCAME"
    for name in ("BOSCAMEMathApprox.h", "BOSCAMEMathApproxCoefficients.h"):
        path = math_dir / name
        before_hashes[str(path)] = sha(path)
        shutil.copy2(path, snapshots / name)
    wrapper = snapshots / "host-math.c"
    wrapper.write_text(
        """/* CPU-only sigmoid recipe matching BOSCAMEMathApprox.h. */
#define """
        + args.silu_symbol
        + """ target_reference_silu
#include "silu.c"
void target_reference_sigmoid(const float *input,float *output,size_t count) {
  for(size_t i=0;i<count;++i) {
    float z=-input[i],s1,s2,r=reduced(z,&s1,&s2),p=coefficients[7];
    for(int n=6;n>=0;--n){p=p*r;p=p+coefficients[n];}
    p=p*s2;p=p*s1;
    if(z<=-103.97208404541015625f)p=0.0f;
    if(z>=88.72283935546875f)p=__builtin_inff();
    if(z!=z)p=z;
    p=p+1.0f;output[i]=1.0f/p;
  }
}
uint64_t target_reference_fnv(const unsigned char *input,size_t count) {
  uint64_t hash=UINT64_C(0xcbf29ce484222325);
  for(size_t i=0;i<count;++i){hash^=input[i];hash*=UINT64_C(0x100000001b3);}
  return hash;
}
"""
    )
    library = args.output / "host-math.so"
    command(
        [
            args.host_cc,
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-tree-vectorize",
            "-fno-tree-slp-vectorize",
            "-ffp-contract=off",
            wrapper,
            snapshots / "gap.c",
            "-o",
            library,
        ],
        args.output / "host-compile.log",
    )
    compiler_version = command(
        [args.host_cc, "--version"], args.output / "host-compiler-version.log"
    )
    lib = ctypes.CDLL(str(library))
    for name in ("target_reference_silu", "target_reference_sigmoid"):
        getattr(lib, name).argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
    lib.efficientnet_gap_phase_candidate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.target_reference_fnv.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.target_reference_fnv.restype = ctypes.c_uint64
    graph = json.loads(args.graph.read_text())
    if graph.get("topology_status") != "PASS":
        raise ValueError("validated graph required")
    plan = json.loads((args.plan_dir / "physical-plan.json").read_text())
    calls_by_node = {}
    for call in plan["calls"]:
        calls_by_node.setdefault(call["logical_node"], []).append(call["index"])
    model, _ = primary.load_model(args.assets)
    modules = dict(model.named_modules())
    archive = np.load(args.primary_reference / "folded-and-quantized-parameters.npz")
    parameters = {
        item["node"]: {
            field: torch.from_numpy(archive[item["node"] + "." + field].copy())
            for field in ("weight", "bias", "qweight", "weight_scale")
        }
        for item in graph["parameters"]
    }
    manifest = json.loads((args.plan_dir / "parameters.json").read_text())
    blob = (args.plan_dir / "parameters.bin").read_bytes()
    if sha(args.plan_dir / "parameters.bin") != manifest["sha256"]:
        raise ValueError("physical parameter blob hash mismatch")
    for entry in manifest["entries"]:
        if entry["name"].endswith(".zero_bias_i32"):
            if any(blob[entry["offset"] : entry["offset"] + entry["bytes"]]):
                raise ValueError("nonzero physical integer GEMM bias placeholder")
            continue
        raw = archive[entry["name"]].tobytes()
        if (
            hashlib.sha256(raw).hexdigest() != entry["sha256"]
            or raw != blob[entry["offset"] : entry["offset"] + entry["bytes"]]
        ):
            raise ValueError("physical/NPZ parameter mismatch: " + entry["name"])
    image = torch.from_numpy(
        np.fromfile(args.plan_dir / "input.f32", dtype="<f4")
        .reshape(graph["input"]["shape"])
        .copy()
    )
    golden = torch.from_numpy(
        np.fromfile(args.primary_reference / "w8a8-logits.f32", dtype="<f4")
        .reshape(graph["output"]["shape"])
        .copy()
    )
    primary_trace = json.loads(
        (args.primary_reference / "w8a8-node-trace.json").read_text()
    )
    original_gap_functions = {}
    gap_evidence = {}
    ir_dir = args.output / "original-gap-llvm-host"
    ir_dir.mkdir(exist_ok=True)
    for name in sorted(
        {node["case"] for node in graph["nodes"] if node["kind"] == "gap"}
    ):
        case = cases.describe(name)
        ir = args.kernel_root / name / "kernel.ll"
        boscame = args.kernel_root / name / "kernel.boscame.mlir"
        if "@triton_" + name not in ir.read_text():
            raise ValueError("original Triton GAP LLVM reference required: " + str(ir))
        snapshot_ir = ir_dir / (name + ".ll")
        shutil.copy2(ir, snapshot_ir)
        original_lib = ir_dir / (name + ".so")
        command(
            [
                args.llvm_clang,
                "-shared",
                "-fPIC",
                "-O2",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-ffp-contract=off",
                snapshot_ir,
                "-o",
                original_lib,
            ],
            ir_dir / (name + "-compile.log"),
        )
        fn = getattr(ctypes.CDLL(str(original_lib)), "_mlir_ciface_triton_" + name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int32] * 6
        original_gap_functions[name] = fn
        gap_evidence[name] = {
            "block": case["constexprs"]["BLOCK"],
            "case_source_path": case["kernel_source"],
            "case_source_sha256": sha(case["kernel_source"]),
            "original_emitted_llvm_path": str(ir),
            "original_emitted_llvm_sha256": sha(ir),
            "original_emitted_boscame_path": str(boscame),
            "original_emitted_boscame_sha256": sha(boscame),
            "host_compiled_original_llvm_sha256": sha(original_lib),
            "logical_inputs_checked": 0,
            "bit_mismatches": 0,
        }

    def fnv(value):
        return f"{lib.target_reference_fnv(value.data_ptr(), value.numel() * value.element_size()):016x}"

    def unary(name, value):
        out = torch.empty_like(value)
        getattr(lib, name)(value.data_ptr(), out.data_ptr(), value.numel())
        return out

    def run(target_order):
        uses = Counter(key for node in graph["nodes"] for key in node["inputs"])
        uses[graph["output"]["value"]] += 1
        values, trace = {"input": image}, []
        for node in graph["nodes"]:
            x = [values[key] for key in node["inputs"]]
            name, kind, begin = node["name"], node["kind"], time.monotonic()
            entry = {}
            if kind in ("conv", "depthwise"):
                out = primary.w8a8_conv(x[0], modules[name], parameters[name])
            elif kind == "silu":
                out = (
                    unary("target_reference_silu", x[0])
                    if target_order
                    else F.silu(x[0])
                )
            elif kind == "sigmoid":
                out = (
                    unary("target_reference_sigmoid", x[0])
                    if target_order
                    else torch.sigmoid(x[0])
                )
            elif kind == "gap":
                if target_order:
                    case = node["case"]
                    _, channels, height, width = x[0].shape
                    spatial = height * width
                    out = torch.empty(node["shape"], dtype=torch.float32)
                    lib.efficientnet_gap_phase_candidate(
                        x[0].data_ptr(),
                        out.data_ptr(),
                        spatial,
                        channels,
                        gap_evidence[case]["block"],
                    )
                    emitted = torch.empty_like(out)
                    xm = Memref(x[0].data_ptr(), x[0].data_ptr(), 0, x[0].numel(), 1)
                    ym = Memref(
                        emitted.data_ptr(), emitted.data_ptr(), 0, emitted.numel(), 1
                    )
                    xu, yu = (
                        Unranked(1, ctypes.addressof(xm)),
                        Unranked(1, ctypes.addressof(ym)),
                    )
                    for channel in range(channels):
                        original_gap_functions[case](
                            ctypes.byref(xu),
                            ctypes.byref(yu),
                            channels,
                            1,
                            1,
                            channel,
                            0,
                            0,
                        )
                    mismatches = int(
                        np.count_nonzero(
                            out.numpy().view(np.uint32)
                            != emitted.numpy().view(np.uint32)
                        )
                    )
                    gap_evidence[case]["logical_inputs_checked"] += 1
                    gap_evidence[case]["bit_mismatches"] += mismatches
                    entry["gap_original_emitted_llvm_bit_mismatches"] = mismatches
                    if mismatches:
                        raise ValueError(
                            "ordered C differs from original emitted GAP LLVM: " + name
                        )
                else:
                    out = x[0].mean((2, 3), keepdim=True)
            elif kind == "se_broadcast_multiply":
                out = x[0] * x[1]
            elif kind == "add":
                out = x[0] + x[1]
            elif kind == "reshape_view":
                out = x[0].reshape(node["shape"])
            elif kind == "linear_w8a8_chain":
                p = parameters[name]
                qx, xs = primary.quantize_rows(x[0])
                acc = primary.integer_mm(qx, p["qweight"].T)
                out = acc.float() * xs[:, None]
                out = out * p["weight_scale"][None, :]
                out = out + p["bias"][None, :]
            else:
                raise ValueError(kind)
            out = out.float().contiguous()
            if list(out.shape) != node["shape"] or not torch.isfinite(out).all():
                raise ValueError("shape/nonfinite failure: " + name)
            values[node["output"]] = out
            entry.update(
                index=node["index"],
                node=name,
                kind=kind,
                case=node["case"],
                output=node["output"],
                shape=list(out.shape),
                output_sha256=primary.array_sha(out),
                output_fnv64=fnv(out),
                primary_node_sha256_matches=primary.array_sha(out)
                == primary_trace[node["index"]]["output_sha256"],
                physical_call_indices=calls_by_node.get(name, []),
                final_physical_call=max(calls_by_node.get(name, [-1])),
                min=float(out.min()),
                max=float(out.max()),
                host_seconds=time.monotonic() - begin,
            )
            trace.append(entry)
            if target_order and name in args.dump_node:
                directory = args.output / "target-order-node-values"
                directory.mkdir(exist_ok=True)
                tensors = {"output": out.numpy()}
                tensors.update(
                    {"input_" + str(i): value.numpy() for i, value in enumerate(x)}
                )
                if name in parameters:
                    tensors.update(
                        {
                            field: value.numpy()
                            for field, value in parameters[name].items()
                        }
                    )
                np.savez(directory / (str(node["index"]).zfill(3) + ".npz"), **tensors)
                out.numpy().astype("<f4").tofile(
                    directory / (str(node["index"]).zfill(3) + ".f32")
                )
                save(directory / (str(node["index"]).zfill(3) + ".json"), entry)
            for key in node["inputs"]:
                uses[key] -= 1
                if not uses[key]:
                    del values[key]
            if len(trace) % 40 == 0:
                print(
                    ("target-order" if target_order else "primary-baseline"),
                    len(trace),
                    "/",
                    len(graph["nodes"]),
                    flush=True,
                )
        return values[graph["output"]["value"]], trace

    with torch.inference_mode():
        baseline, baseline_trace = run(False)
        if primary.array_sha(baseline) != sha(
            args.primary_reference / "w8a8-logits.f32"
        ) or not all(row["primary_node_sha256_matches"] for row in baseline_trace):
            raise ValueError(
                "CPU primary baseline did not reproduce all original node hashes/logits"
            )
        actual, actual_trace = run(True)
    logits_path = args.output / "target-order-logits.f32"
    actual.numpy().astype("<f4").tofile(logits_path)
    save(args.output / "target-order-node-trace.json", actual_trace)
    save(args.output / "primary-baseline-node-trace.json", baseline_trace)
    error = (actual - golden).abs()
    mismatches = torch.nonzero(error > 0.05 + 0.01 * golden.abs(), as_tuple=False)
    if any(sha(path) != value for path, value in before_hashes.items()):
        raise ValueError("an input/source changed during semantic replay")
    report = {
        "status": "CPU_DERIVED_TARGET_ORDER_REFERENCE",
        "full_model_on_fpga": False,
        "scope": "308-node real-weight CPU semantics with exact integer W8A8 products; not FPGA or image-classification accuracy evidence",
        "primary_files_preserved": True,
        "primary_baseline_all_node_hashes_match": True,
        "output_path": str(logits_path),
        "output_sha256": sha(logits_path),
        "output_fnv64": fnv(actual),
        "logical_nodes": len(actual_trace),
        "physical_plan_calls": len(plan["calls"]),
        "max_abs_error_vs_primary": float(error.max()),
        "mean_abs_error_vs_primary": float(error.mean()),
        "rmse_vs_primary": float(((actual - golden) ** 2).mean().sqrt()),
        "primary_top5": golden[0].topk(5).indices.tolist(),
        "target_order_top5": actual[0].topk(5).indices.tolist(),
        "allclose_atol_0_05_rtol_0_01": bool(
            torch.allclose(actual, golden, atol=0.05, rtol=0.01)
        ),
        "indices_outside_atol_0_05_rtol_0_01": mismatches.tolist(),
        "silu_source_symbol": args.silu_symbol,
        "gap_original_emitted_llvm_parity": gap_evidence,
        "torch_version": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": args.threads,
        "host_compiler_version": compiler_version,
        "host_library_sha256": sha(library),
        "llvm_clang_sha256": sha(args.llvm_clang),
        "source_and_input_hashes": before_hashes,
        "parameter_entries_verified_against_exact_physical_blob": len(
            manifest["entries"]
        ),
        "limitations": [
            "Host C scalar arithmetic predicts the declared RVV expressions, without FPGA memory/cache behavior.",
            "The sigmoid host recipe uses the private exp reduction/coefficient helpers in the explicitly supplied SiLU source; backend math headers are snapshotted and hashed.",
            "Ordered GAP retains original emitted LLVM addition/division order; Torch CPU mean uses a different reduction order.",
            "Primary Torch golden remains the comparison baseline; the separately named target-order logits are a derived reference.",
        ],
    }
    save(args.output / "target-order-reference-report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "output_sha256",
                    "max_abs_error_vs_primary",
                    "mean_abs_error_vs_primary",
                    "target_order_top5",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
