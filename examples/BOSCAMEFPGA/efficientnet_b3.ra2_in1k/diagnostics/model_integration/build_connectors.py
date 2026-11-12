#!/usr/bin/env python3
"""Compile physical-plan connectors with the repository's local BOSCAME chain.

No production cases, compiler sources or board jobs are modified/submitted.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BUDDY = REPO / "buddy-mlir"
sys.path[:0] = [str(ROOT / "tools"), str(REPO / "examples/BOSCAMEFPGA/tools")]
from export import descriptor_checks  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, log, stdin=None):
    p = subprocess.run(
        list(map(str, args)), input=stdin, capture_output=True, text=True
    )
    with log.open("a") as f:
        f.write("$ " + repr(list(map(str, args))) + chr(10) + p.stderr)
    if p.returncode:
        raise RuntimeError("command failed: " + repr(args) + "; see " + str(log))
    return p.stdout


def adapter(spec):
    args = spec["arguments"]
    shapes = spec["argument_shapes"]
    params = ", ".join("MemRef%d *a%d" % (a["rank"], i) for i, a in enumerate(args))
    lines = [
        "/* Generated physical-model connector ABI. */",
        '#include "support.h"',
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        "extern void _mlir_ciface_"
        + spec["symbol"]
        + "("
        + ", ".join(["Unranked *"] * len(args) + ["int32_t"] * 6)
        + ");",
        "void _mlir_ciface_kernel_" + spec["name"] + "(" + params + ") {",
        *descriptor_checks(spec, shapes),
    ]
    gx, gy, gz = spec["grid"]
    lines += [
        "  for (int32_t x=0;x<%d;++x)" % gx,
        "    for (int32_t y=0;y<%d;++y)" % gy,
        "      for (int32_t z=0;z<%d;++z)" % gz,
        "        _mlir_ciface_"
        + spec["symbol"]
        + "("
        + ", ".join("&u%d" % i for i in range(len(args)))
        + ", %d, %d, %d, x, y, z);" % (gx, gy, gz),
        "}",
        "",
    ]
    return chr(10).join(lines)


def worker(a, spec):
    from contract_frontend import export_kernel

    out = a.output / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "build.log"
    log.write_text("")
    os.environ["BUDDY_OPT"] = str(a.buddy_opt)
    modspec = importlib.util.spec_from_file_location(
        "integration_connector", HERE / "connector_kernels.py"
    )
    module = importlib.util.module_from_spec(modspec)
    sys.modules[modspec.name] = module
    modspec.loader.exec_module(module)
    start = time.monotonic()
    export_kernel(
        getattr(module, spec["kernel"]),
        spec["signature"],
        spec["constexprs"],
        spec["grid"],
        spec["symbol"],
        out,
        math_accuracy=spec["math_accuracy"],
    )
    (out / "adapter.c").write_text(adapter(spec))
    source = out / "kernel.frontend.mlir"
    for target, flags in [
        ("kernel.bufferized.mlir", ["--kernel-contract-verify"]),
        ("kernel.planned.mlir", ["--bosc-ame-plan", "--bosc-ame-plan-verify"]),
        ("kernel.boscame.mlir", ["--bosc-ame-emit"]),
    ]:
        dest = out / target
        run([a.buddy_opt, source, *flags, "-o", dest], log)
        source = dest
    if a.stage == "object":
        # Pure CPU connectors also consume DMA/AME outputs from earlier calls.
        # The backend auto-enables this policy only in mixed AME/CPU functions;
        # explicitly request its existing per-DDR-load/read and store/write
        # lowering here, with no changes to production kernels or the backend.
        policy_source = out / "kernel.cpu-ddr-policy.mlir"
        text = source.read_text()
        if "bosc_ame.cpu_ddr_policy" not in text.split("func.func", 1)[0]:
            text = text.replace(
                "module attributes {", "module attributes {bosc_ame.cpu_ddr_policy, ", 1
            )
        policy_source.write_text(text)
        run(
            [
                a.buddy_opt,
                policy_source,
                "--lower-bosc-ame=scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true",
                "-o",
                out / "kernel.llvm.mlir",
            ],
            log,
        )
        run(
            [
                a.buddy_translate,
                "--buddy-to-llvmir",
                out / "kernel.llvm.mlir",
                "-o",
                out / "kernel.ll",
            ],
            log,
        )
        run(
            [
                a.llc,
                out / "kernel.ll",
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                out / "kernel.s",
            ],
            log,
        )
        encoded = run(
            [
                sys.executable,
                BUDDY / "examples/FPGA-BOSCAME/tools/ame_to_word.py",
                "--gpr-mode=fixed",
                "--preserve-matrix-registers",
            ],
            log,
            (out / "kernel.s").read_text(),
        )
        (out / "kernel.encoded.s").write_text(encoded)
        restricted = run(
            [
                sys.executable,
                BUDDY / "examples/FPGA-BOSCAME/tools/restrict_fpga_assembly.py",
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            log,
            encoded,
        )
        (out / "kernel.nr.s").write_text(restricted)
        flags = [
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
        ]
        run([a.clang, *flags, "-c", out / "kernel.nr.s", "-o", out / "kernel.o"], log)
        run(
            [
                a.clang,
                *flags,
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-I",
                ROOT,
                "-c",
                out / "adapter.c",
                "-o",
                out / "adapter.o",
            ],
            log,
        )
    files = ["kernel.boscame.mlir", "adapter.c"] + (
        ["kernel.o", "adapter.o"] if a.stage == "object" else []
    )
    result = {
        "name": spec["name"],
        "status": "PASS",
        "stage": a.stage,
        "seconds": time.monotonic() - start,
        "artifacts": {
            name: {"bytes": (out / name).stat().st_size, "sha256": sha(out / name)}
            for name in files
        },
    }
    (out / "build-result.json").write_text(json.dumps(result, indent=2) + chr(10))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "build/model-integration/physical-plan/physical-plan.json",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/model-integration/connectors"
    )
    p.add_argument("--buddy-opt", type=Path, default=BUDDY / "build/bin/buddy-opt")
    p.add_argument(
        "--buddy-translate", type=Path, default=BUDDY / "build/bin/buddy-translate"
    )
    p.add_argument("--llc", type=Path, default=BUDDY / "llvm/build/bin/llc")
    p.add_argument("--clang", type=Path, default=BUDDY / "llvm/build/bin/clang")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--stage", choices=["boscame", "object"], default="object")
    p.add_argument("--worker")
    p.add_argument("--kernel")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())
    specs = plan["connectors"]
    if plan["connector_source_sha256"] != sha(HERE / "connector_kernels.py"):
        raise ValueError("connector source changed; regenerate physical plan")
    if a.worker:
        spec = next(s for s in specs if s["name"] == a.worker)
        print(json.dumps(worker(a, spec)))
        return
    if a.kernel:
        specs = [s for s in specs if s["kernel"] == a.kernel]
    if a.limit:
        specs = specs[: a.limit]
    results = []

    def dispatch(spec):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            spec["name"],
            "--plan",
            str(a.plan),
            "--output",
            str(a.output),
            "--stage",
            a.stage,
            "--buddy-opt",
            str(a.buddy_opt),
            "--buddy-translate",
            str(a.buddy_translate),
            "--llc",
            str(a.llc),
            "--clang",
            str(a.clang),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            (a.output / spec["name"]).mkdir(parents=True, exist_ok=True)
            (a.output / spec["name"] / "failure.log").write_text(
                proc.stdout + proc.stderr
            )
            return {
                "name": spec["name"],
                "status": "FAIL",
                "error": proc.stderr[-2000:],
            }
        return json.loads(proc.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(dispatch, s) for s in specs]
        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            print(result["status"], result["name"], flush=True)
    tools = {
        name: {"path": str(getattr(a, name)), "sha256": sha(getattr(a, name))}
        for name in ("buddy_opt", "buddy_translate", "llc", "clang")
    }
    report = {
        "status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
        "scope": "full-shape connector compilation only, no FPGA execution or full-shape numeric validation",
        "stage": a.stage,
        "expected": len(specs),
        "completed": len(results),
        "tools": tools,
        "plan_sha256": sha(a.plan),
        "connector_source_sha256": sha(HERE / "connector_kernels.py"),
        "results": sorted(results, key=lambda r: r["name"]),
    }
    (a.output / "build-summary.json").write_text(json.dumps(report, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "status": report["status"],
                "expected": len(specs),
                "passed": sum(r["status"] == "PASS" for r in results),
            }
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
