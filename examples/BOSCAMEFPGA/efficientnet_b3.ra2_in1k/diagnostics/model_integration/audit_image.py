#!/usr/bin/env python3
"""Independently audit exact image bytes, symbols and per-connector DDR policies."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image-dir", type=Path, default=ROOT / "build/model-integration/model-image"
    )
    p.add_argument(
        "--plan-dir", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    p.add_argument(
        "--connectors",
        type=Path,
        default=ROOT / "build/model-integration/connectors-ddr-policy",
    )
    a = p.parse_args()
    m = json.loads((a.image_dir / "image-manifest.json").read_text())
    plan = json.loads((a.plan_dir / "physical-plan.json").read_text())
    snapshots = {}
    snapshot_manifest = a.image_dir / "input-snapshot-manifest.json"
    if snapshot_manifest.exists():
        frozen = json.loads(snapshot_manifest.read_text())
        if frozen["image_sha256"] != m["image_sha256"] or frozen[
            "image_manifest_sha256"
        ] != sha(a.image_dir / "image-manifest.json"):
            raise ValueError("input snapshot manifest belongs to a different image")
        snapshots = {
            (item["kind"], item["case"], item["filename"]): Path(item["snapshot"])
            for item in frozen["records"]
        }
    runner = m.get("runner_inputs")
    if runner:
        for key in ("manifest", "source", "header", "linked_object"):
            if sha(Path(runner[key])) != runner[key + "_sha256"]:
                raise ValueError("runner input changed " + key)
        if runner["plan_sha256"] != sha(a.plan_dir / "physical-plan.json"):
            raise ValueError("runner was generated for another plan")

    def input_path(item, kind, filename):
        name = item.get("case", item.get("name"))
        explicit = item.get(
            "object_snapshot" if filename == "kernel.o" else "adapter_snapshot"
        )
        return (
            Path(explicit)
            if explicit
            else snapshots.get((kind, name, filename))
            or Path(
                item["object"]
                if filename == "kernel.o"
                else item.get(
                    "adapter_source", str(Path(item["object"]).parent / "adapter.c")
                )
            )
        )

    binary = a.image_dir / "efficientnet_b3_model.bin"
    elf = a.image_dir / "efficientnet_b3_model.elf"
    if sha(binary) != m["image_sha256"] or sha(elf) != m["elf_sha256"]:
        raise ValueError("image content changed")
    disassembly = subprocess.run(
        [str(REPO / "buddy-mlir/llvm/build/bin/llvm-objdump"), "-d", str(elf)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    forbidden_reductions = re.findall(r"\bvfred(?:max|osum|usum)\.vs\b", disassembly)
    if forbidden_reductions:
        raise ValueError(
            "unsafe FPGA vector reduction in actual model ELF: "
            + ", ".join(sorted(set(forbidden_reductions)))
        )
    nm = REPO / "buddy-mlir/llvm/build/bin/llvm-nm"
    output = subprocess.run(
        [str(nm), "-n", str(elf)], capture_output=True, text=True, check=True
    ).stdout
    symbols = {}
    for line in output.splitlines():
        f = line.split()
        if len(f) == 3:
            try:
                symbols[f[2]] = int(f[0], 16)
            except ValueError:
                pass
    raw = binary.read_bytes()
    blobs = []
    for symbol, filename in (
        ("model_params_blob", "parameters.bin"),
        ("model_input_blob", "input.f32"),
        ("model_reference_blob", "w8a8-logits.f32"),
    ):
        expected = (a.plan_dir / filename).read_bytes()
        offset = symbols[symbol] - 0x80000000
        actual = raw[offset : offset + len(expected)]
        if actual != expected:
            raise ValueError("embedded bytes mismatch " + symbol)
        blobs.append(
            {
                "symbol": symbol,
                "address": hex(symbols[symbol]),
                "bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            }
        )
    start = symbols["__nh_e6_ddr_window_start"]
    end = symbols["__nh_e6_ddr_window_end"]
    if symbols["__model_workspace_start"] != start:
        raise ValueError("workspace not at window start")
    if symbols["__model_workspace_end"] - start != m["workspace_bytes"]:
        raise ValueError("workspace extent mismatch")
    for symbol in ("__bss_start", "__e6_stack_bottom", "__nh_stack_bottom"):
        if symbols[symbol] < end:
            raise ValueError("control or CPU stack in switchable region " + symbol)
    if any(int(b["address"], 16) + b["bytes"] > start for b in blobs):
        raise ValueError("embedded blob start in operand window")
    expected = {c["target"] for c in plan["calls"]}
    for target in expected:
        if "_mlir_ciface_kernel_" + target not in symbols:
            raise ValueError("missing actual call binding " + target)
    policies = []
    bound_connectors = {item["name"]: item for item in m["connector_inputs"]}
    for spec in plan["connectors"]:
        bound = bound_connectors[spec["name"]]
        d = (
            input_path(bound, "connector", "kernel.o").parent
            if bound.get("object_snapshot")
            else Path(bound["object"]).parent
        )
        if sha(input_path(bound, "connector", "kernel.o")) != bound["object_sha256"]:
            raise ValueError("connector object changed " + spec["name"])
        if sha(input_path(bound, "connector", "adapter.c")) != bound["adapter_sha256"]:
            raise ValueError("connector adapter changed " + spec["name"])
        if (
            bound.get("adapter_object")
            and sha(Path(bound["adapter_object"])) != bound["adapter_object_sha256"]
        ):
            raise ValueError("actual linked connector adapter changed " + spec["name"])
        asm = (d / "kernel.nr.s").read_text()
        native = bound.get("implementation") == "hybrid_rvv_c"
        if native:
            if asm.count("csrw") < 2:
                raise ValueError(
                    "native connector missing read/write DDR policy " + spec["name"]
                )
        else:
            lower = (d / "kernel.llvm.mlir").read_text()
            if "bosc_ame.cpu_ddr_policy" not in lower or "csrw" not in asm:
                raise ValueError(
                    "connector missing actual per-access DDR policy " + spec["name"]
                )
        policies.append(
            {
                "name": spec["name"],
                "csr_writes": asm.count("csrw"),
                "implementation": "hybrid_rvv_c" if native else "boscame",
            }
        )
    cpu_policy_checks = []
    for item in m.get("model_local_cpu_policy_kernels", []):
        directory = a.image_dir / "cpu-kernels" / item["name"]
        if sha(directory / "kernel.o") != item["kernel_sha256"]:
            raise ValueError("actual linked CPU policy object changed " + item["name"])
        if (
            sha(directory / "kernel.cpu-ddr-policy.mlir")
            != item["policy_source_sha256"]
        ):
            raise ValueError("CPU policy source changed " + item["name"])
        if (
            "bosc_ame.cpu_ddr_policy"
            not in (directory / "kernel.cpu-ddr-policy.mlir").read_text()
        ):
            raise ValueError("missing CPU DDR policy " + item["name"])
        if "csrw" not in (directory / "kernel.nr.s").read_text():
            raise ValueError(
                "CPU policy not present in actual assembly " + item["name"]
            )
        cpu_policy_checks.append(
            {"case": item["name"], "object_sha256": item["kernel_sha256"]}
        )
    mma = []
    native_cases = []
    for item in m["existing_case_inputs"]:
        if sha(input_path(item, "existing", "kernel.o")) != item["object_sha256"]:
            raise ValueError("existing kernel object changed " + item["case"])
        if (
            sha(input_path(item, "existing", "adapter.c"))
            != item["adapter_source_sha256"]
        ):
            raise ValueError("existing kernel adapter changed " + item["case"])
        if (
            item.get("adapter_object")
            and sha(Path(item["adapter_object"])) != item["adapter_object_sha256"]
        ):
            raise ValueError("actual linked adapter object changed " + item["case"])
        if item.get("implementation") in ("hybrid_rvv_c", "hybrid_ordered_c"):
            asm = (
                input_path(item, "existing", "kernel.o").parent / "kernel.nr.s"
            ).read_text()
            if asm.count("csrw") < 2:
                raise ValueError(
                    "native case missing actual DDR read/write policy " + item["case"]
                )
            native_cases.append(
                {
                    "case": item["case"],
                    "implementation": item["implementation"],
                    "csr_writes": asm.count("csrw"),
                    "object_sha256": item["object_sha256"],
                }
            )
        if item["family"] not in (
            "conv2d_i8",
            "conv2d_f32",
            "dwconv2d_f32",
            "linear_f32",
        ):
            continue
        d = input_path(item, "existing", "kernel.o").parent
        sources = (
            list((d / "stages").rglob("*.boscame.mlir"))
            if (d / "stages").exists()
            else []
        )
        if not sources:
            sources = [d / "kernel.boscame.mlir"]
        dots = []
        for source in sources:
            for line in source.read_text().splitlines():
                if "bosc_ame.tile.mma" in line:
                    if "f32" in line:
                        raise ValueError("F32 MMA/matmul route " + item["case"])
                    dots.append(line.strip())
        if not dots:
            raise ValueError("no actual integer MMA binding " + item["case"])
        mma.append({"case": item["case"], "integer_mma_ops_in_ir": len(dots)})
    audit = json.loads((a.image_dir / "elf-audit.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("ELF instruction audit failed")
    report = {
        "status": "PASS",
        "scope": "independent image/static binding audit; no FPGA numeric evidence",
        "image_sha256": m["image_sha256"],
        "physical_calls": len(plan["calls"]),
        "distinct_actual_bound_adapters": len(expected),
        "exact_embedded_blobs": blobs,
        "pmp_window": {"start": hex(start), "end": hex(end)},
        "workspace_bytes": m["workspace_bytes"],
        "connector_policy_checks": policies,
        "integer_mma_case_checks": mma,
        "native_case_policy_checks": native_cases,
        "model_local_cpu_policy_checks": cpu_policy_checks,
        "actual_elf_unsafe_fp_vector_reductions": len(forbidden_reductions),
        "elf_instruction_audit_sha256": sha(a.image_dir / "elf-audit.json"),
    }
    (a.image_dir / "independent-image-audit.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "image_sha256",
                    "physical_calls",
                    "distinct_actual_bound_adapters",
                    "exact_embedded_blobs",
                    "pmp_window",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
