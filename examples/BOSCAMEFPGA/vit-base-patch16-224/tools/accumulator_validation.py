#!/usr/bin/env python3
"""Reproduce and validate plan06 with fresh artifacts and explicit provenance.

This driver invokes the existing Makefiles, strict grid simulator and queued
board runner. It never treats old objects, a zero error count without complete
coverage, or host simulation as board execution evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

SUITE = Path(__file__).resolve().parents[1]
REPO = SUITE.parents[2]
BUDDY = REPO / "buddy-mlir"
sys.path.insert(0, str(SUITE))
from cases import describe, inventory  # noqa: E402

FAILED_CASES = (
    "gemm_i8_196x768x768",
    "gemm_i8_197x3072x768",
    "gemm_i8_197x768x3072",
    "gemm_i8_197x768x768",
    "gemm_i8_1x1000x768",
    "matmul_i8_1x1000x768",
)
REGRESSION_CASES = (
    "matmul_i8_196x768x768",
    "matmul_i8_197x3072x768",
    "matmul_i8_197x768x3072",
    "matmul_i8_197x768x768",
)
CASES = FAILED_CASES + REGRESSION_CASES
SOURCE_IDENTITY_FIELDS = (
    "repository_head",
    "backend_head",
    "llvm_head",
    "repository_tracked_diff_sha256",
    "backend_tracked_diff_sha256",
    "llvm_tracked_diff_sha256",
    "source_sha256",
    "frontend",
    "cases",
)
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".inc",
    ".py",
    ".td",
    ".mlir",
    ".ll",
    ".s",
    ".S",
    ".mk",
    ".ld",
    ".cmake",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".sh",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL
    )


def frontend_identity():
    path = REPO / "examples/BOSCAMEFPGA/tools/contract_frontend.py"
    spec = importlib.util.spec_from_file_location("validation_frontend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.frontend_identity()


def source_files(names):
    """Include build inputs even when a new implementation file is untracked."""
    sources = set()

    def add(directory, recursive=False):
        iterator = directory.rglob("*") if recursive else directory.glob("*")
        sources.update(
            path
            for path in iterator
            if path.is_file()
            and (
                path.suffix in SOURCE_SUFFIXES
                or path.name in ("makefile", "Makefile", "CMakeLists.txt")
            )
            and not {"__pycache__", "build", ".tmp"}.intersection(
                path.relative_to(directory).parts
            )
        )

    add(SUITE)
    add(SUITE / "tools")
    for name in names:
        add(SUITE / name)
    for directory in (
        REPO / "include",
        REPO / "lib",
        REPO / "examples/BOSCAMEFPGA/tools",
        BUDDY / "midend/include",
        BUDDY / "midend/lib",
        BUDDY / "runtime/boscame/runtime",
        BUDDY / "runtime/boscame/nh_e6",
        BUDDY / "examples/FPGA-BOSCAME/tools",
        BUDDY / "llvm/llvm/lib/Target/RISCV",
    ):
        add(directory, recursive=True)
    for path in (
        REPO / "CMakeLists.txt",
        BUDDY / "CMakeLists.txt",
        BUDDY / "llvm/llvm/include/llvm/IR/IntrinsicsRISCV.td",
    ):
        if path.is_file():
            sources.add(path)
    return sorted(sources)


def identity(names=CASES):
    """Hash compiler/runtime/harness sources and all three dirty tracked trees."""
    tools = {
        name: BUDDY / "build/bin" / name for name in ("buddy-opt", "buddy-translate")
    }
    tools.update({name: BUDDY / "llvm/build/bin" / name for name in ("llc", "clang")})
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": sys.version,
        "repository_head": git(REPO, "rev-parse", "HEAD").decode().strip(),
        "backend_head": git(BUDDY, "rev-parse", "HEAD").decode().strip(),
        "llvm_head": git(BUDDY / "llvm", "rev-parse", "HEAD").decode().strip(),
        "repository_tracked_diff_sha256": hashlib.sha256(
            git(REPO, "diff", "--no-ext-diff", "--binary", "HEAD")
        ).hexdigest(),
        "backend_tracked_diff_sha256": hashlib.sha256(
            git(BUDDY, "diff", "--no-ext-diff", "--binary", "HEAD")
        ).hexdigest(),
        "llvm_tracked_diff_sha256": hashlib.sha256(
            git(BUDDY / "llvm", "diff", "--no-ext-diff", "--binary", "HEAD")
        ).hexdigest(),
        "source_sha256": {
            str(path.relative_to(REPO)): digest(path) for path in source_files(names)
        },
        "frontend": frontend_identity(),
        "tools": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in tools.items()
            if path.is_file()
        },
        "cases": {name: describe(name) for name in names},
    }


def source_identity_changes(before, after):
    fields = [key for key in SOURCE_IDENTITY_FIELDS if before[key] != after[key]]
    old, new = before["source_sha256"], after["source_sha256"]
    return {
        "fields": fields,
        "sources": sorted(
            name for name in old.keys() | new.keys() if old.get(name) != new.get(name)
        ),
    }


def snapshot_sources(destination, current):
    """Save exact source bytes plus HEAD/patches, detecting concurrent edits."""
    directory = destination / "source-snapshot"
    directory.mkdir()
    archive_path = directory / "sources.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, expected in current["source_sha256"].items():
            path = REPO / name
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("source changed while freezing: " + name)
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = path.stat().st_mode & 0o777
            archive.addfile(member, io.BytesIO(data))
    files = {str(archive_path): digest(archive_path)}
    for label, root in (
        ("repository", REPO),
        ("backend", BUDDY),
        ("llvm", BUDDY / "llvm"),
    ):
        patch = git(root, "diff", "--no-ext-diff", "--binary", "HEAD")
        if hashlib.sha256(patch).hexdigest() != current[label + "_tracked_diff_sha256"]:
            raise RuntimeError(label + " tracked sources changed while freezing")
        path = directory / (label + ".patch")
        path.write_bytes(patch)
        files[str(path)] = digest(path)
    result = {
        "files": files,
        "source_sha256": current["source_sha256"],
        "heads": {
            key: current[key] for key in SOURCE_IDENTITY_FIELDS if key.endswith("_head")
        },
    }
    write_json(directory / "manifest.json", result)
    files[str(directory / "manifest.json")] = digest(directory / "manifest.json")
    return result


def snapshot_unchanged(snapshot):
    return all(
        Path(path).is_file() and digest(path) == expected
        for path, expected in snapshot["files"].items()
    )


def frozen_tools_unchanged(tools):
    for name, value in tools.items():
        path = Path(value["path"])
        if not path.is_file() or digest(path) != value["sha256"]:
            return False
        if name == "clang" and "resource_sha256" in value:
            resource = Path(
                subprocess.check_output(
                    [str(path), "-print-resource-dir"], text=True
                ).strip()
            )
            actual = {
                str(item.relative_to(resource)): digest(item)
                for item in resource.rglob("*")
                if item.is_file()
            }
            if actual != value["resource_sha256"]:
                return False
    return True


def run(command, log, cwd=None):
    print("$ " + " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    with Path(log).open("w") as output:
        output.write("$ " + " ".join(map(str, command)) + "\n")
        output.flush()
        process = subprocess.run(
            command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT
        )
    return {
        "command": list(map(str, command)),
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "log": str(log),
        "log_sha256": digest(log),
    }


def reproduce(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    before = identity(FAILED_CASES)
    write_json(destination / "identity.json", before)
    frozen_opt = destination / "buddy-opt"
    shutil.copyfile(before["tools"]["buddy-opt"]["path"], frozen_opt)
    frozen_opt.chmod(0o755)
    if digest(frozen_opt) != before["tools"]["buddy-opt"]["sha256"]:
        raise RuntimeError("buddy-opt changed while freezing baseline")
    records = []
    for name in FAILED_CASES:
        case_dir = destination / name
        case_dir.mkdir()
        source = args.kernels / name / "kernel.bufferized.mlir"
        frozen_source = case_dir / source.name
        shutil.copyfile(source, frozen_source)
        planned = case_dir / "kernel.planned.mlir"
        emitted = case_dir / "kernel.boscame.mlir"
        record = {
            "case": name,
            "bufferized_sha256": digest(frozen_source),
            "stages": [],
        }
        plan = run(
            [
                str(frozen_opt),
                str(frozen_source),
                "--bosc-ame-plan",
                "--bosc-ame-plan-verify",
                "-o",
                str(planned),
            ],
            case_dir / "planned.log",
        )
        record["stages"].append({"stage": "plan_verify_default", **plan})
        if plan["exit_code"] == 0:
            if "plan.version = 2 : i32" not in planned.read_text():
                raise RuntimeError("default planner did not select V2")
            emission = run(
                [str(frozen_opt), str(planned), "--bosc-ame-emit", "-o", str(emitted)],
                case_dir / "emitted.log",
            )
            record["stages"].append({"stage": "emit", **emission})
        record["artifacts"] = {
            path.name: digest(path) for path in (planned, emitted) if path.exists()
        }
        records.append(record)
        write_json(destination / "reproduction.json", {"schema": 1, "results": records})
    return 0


def build(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    names = selected(args)
    before = identity(names)
    write_json(destination / "identity-before.json", before)
    source_snapshot = snapshot_sources(destination, before)
    tool_flags = []
    frozen_tools = {}
    if args.freeze_tools:
        tool_dir = destination / "tools"
        tool_dir.mkdir()
        make_variables = {
            "buddy-opt": "BUDDY_OPT",
            "buddy-translate": "BUDDY_TRANSLATE",
            "llc": "LLC",
            "clang": "CLANG",
        }
        for name, original in before["tools"].items():
            target = tool_dir / name
            shutil.copyfile(original["path"], target)
            target.chmod(0o755)
            if digest(target) != original["sha256"]:
                raise RuntimeError(name + " changed while freezing build tools")
            frozen_tools[name] = {"path": str(target), "sha256": original["sha256"]}
            tool_flags.append(make_variables[name] + "=" + str(target))
            if name == "clang":
                resource = Path(
                    subprocess.check_output(
                        [original["path"], "-print-resource-dir"], text=True
                    ).strip()
                )
                target_resource = Path(
                    subprocess.check_output(
                        [str(target), "-print-resource-dir"], text=True
                    ).strip()
                )
                shutil.copytree(resource, target_resource, symlinks=True)
                frozen_tools[name]["resource_sha256"] = {
                    str(path.relative_to(resource)): digest(path)
                    for path in sorted(resource.rglob("*"))
                    if path.is_file()
                }
        write_json(destination / "frozen-tools.json", frozen_tools)
    kernels = destination / "kernels"
    result = run(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(SUITE),
            "-j" + str(args.jobs),
            "asm",
            "CASES=" + " ".join(names),
            "KERNELS=" + str(kernels),
            "PYTHON=" + sys.executable,
            *tool_flags,
        ],
        destination / "build.log",
    )
    results = {"asm": result, "cases": {}}
    for name in names:
        directory = kernels / name
        summary = directory / "build-result.json"
        record = (
            json.loads(summary.read_text())
            if summary.exists()
            else {"status": "MISSING"}
        )
        plan = directory / "kernel.planned.mlir"
        record["default_v2"] = (
            plan.exists() and "plan.version = 2 : i32" in plan.read_text()
        )
        if args.bin and record.get("status") == "PASS":
            record["bin_build"] = run(
                [
                    "make",
                    "--no-print-directory",
                    "-C",
                    str(SUITE / name),
                    "bin",
                    "GOLDEN=1",
                    "KERNELS=" + str(kernels),
                    "PYTHON=" + sys.executable,
                    *(["ACCUMULATOR_VALIDATION=1"] if args.accumulator_board else []),
                    *(
                        ["ACCUMULATOR_ROUND=" + str(args.accumulator_round)]
                        if args.accumulator_round is not None
                        else []
                    ),
                    *tool_flags,
                ],
                destination / (name + ".bin.log"),
            )
            image = directory / "board" / (name + ".bin")
            if image.is_file() and record["bin_build"]["exit_code"] == 0:
                record["image_sha256"] = digest(image)
        results["cases"][name] = record
        write_json(destination / "build-validation.json", results)
    after = identity(names)
    write_json(destination / "identity-after.json", after)
    results["effective_tools"] = frozen_tools or before["tools"]
    results["stable_tool_identity"] = (
        frozen_tools_unchanged(frozen_tools)
        if frozen_tools
        else before["tools"] == after["tools"]
    )
    results["source_identity_changes"] = source_identity_changes(before, after)
    results["stable_source_identity"] = not results["source_identity_changes"]["fields"]
    results["source_snapshot"] = source_snapshot
    results["stable_source_snapshot"] = snapshot_unchanged(source_snapshot)
    results["status"] = (
        "PASS"
        if (
            result["exit_code"] == 0
            and results["stable_tool_identity"]
            and results["stable_source_identity"]
            and results["stable_source_snapshot"]
            and all(
                r["default_v2"]
                and r["status"] == "PASS"
                and (not args.bin or r.get("image_sha256"))
                for r in results["cases"].values()
            )
        )
        else "INCOMPLETE"
    )
    write_json(destination / "build-validation.json", results)
    return 0 if results["status"] == "PASS" else 1


def selected(args):
    names = args.case or ([case["name"] for case in inventory()] if args.all else CASES)
    for name in names:
        describe(name)
    return list(names)


def simulate(args):
    names = selected(args)
    frozen = args.kernels.resolve().parent / "frozen-tools.json"
    opt = (
        json.loads(frozen.read_text())["buddy-opt"]["path"]
        if frozen.is_file()
        else str(BUDDY / "build/bin/buddy-opt")
    )
    command = [
        "env",
        "BUDDY_ROOT=" + str(BUDDY),
        "BUDDY_OPT=" + opt,
        sys.executable,
        str(SUITE / "tools/simcheck.py"),
        "--output",
        str(args.kernels.resolve()),
    ]
    for name in names:
        command += ["--case", name]
    args.output.mkdir(parents=True, exist_ok=True)
    return run(command, args.output / "simulation.log")["exit_code"]


def images(args):
    """Build a separate image flavor from verified, immutable kernel objects."""
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    names = selected(args)
    original_root = args.kernels.resolve()
    tool_manifest = original_root.parent / "frozen-tools.json"
    tools = json.loads(tool_manifest.read_text())
    if not frozen_tools_unchanged(tools):
        raise ValueError("frozen compiler or Clang resource changed")
    make_variables = {
        "buddy-opt": "BUDDY_OPT",
        "buddy-translate": "BUDDY_TRANSLATE",
        "llc": "LLC",
        "clang": "CLANG",
    }
    tool_flags = []
    for name, value in tools.items():
        tool_flags.append(make_variables[name] + "=" + value["path"])
    write_json(destination / "frozen-tools.json", tools)
    before = identity(names)
    write_json(destination / "identity-before.json", before)
    source_snapshot = snapshot_sources(destination, before)
    kernels = destination / "kernels"
    kernels.mkdir()
    shutil.copy2(original_root / ".frontend-inputs.json", kernels)
    records = []
    for name in names:
        original = original_root / name
        result = json.loads((original / "build-result.json").read_text())
        if result.get("status") != "PASS" or result.get("stage") != "asm":
            raise ValueError("image source requires a passing asm build: " + name)
        target = kernels / name
        target.mkdir()
        for filename, expected in result["artifacts"].items():
            if digest(original / filename) != expected:
                raise ValueError(
                    "image source artifact changed: " + name + "/" + filename
                )
            shutil.copy2(original / filename, target / filename)
        for filename in (
            "frontend.json",
            "adapter.c",
            "contract-manifest.json",
            ".compiler-inputs.json",
            "build-result.json",
        ):
            shutil.copy2(original / filename, target / filename)
        command = [
            "make",
            "--no-print-directory",
            "-C",
            str(SUITE / name),
            "bin",
            "GOLDEN=1",
            "KERNELS=" + str(kernels),
            "PYTHON=" + sys.executable,
            *(["ACCUMULATOR_VALIDATION=1"] if args.accumulator_board else []),
            *(
                ["ACCUMULATOR_ROUND=" + str(args.accumulator_round)]
                if args.accumulator_round is not None
                else []
            ),
            *tool_flags,
        ]
        record = {"case": name, **run(command, destination / (name + ".bin.log"))}
        image = target / "board" / (name + ".bin")
        record["same_kernel_object"] = (
            digest(target / "kernel.o") == result["artifacts"]["kernel.o"]
        )
        if (
            image.is_file()
            and record["exit_code"] == 0
            and record["same_kernel_object"]
        ):
            record["image_sha256"] = digest(image)
            record["status"] = "PASS"
        else:
            record["status"] = "FAIL"
        records.append(record)
        write_json(
            destination / "images.json",
            {"source_kernels": str(original_root), "cases": records},
        )
    after = identity(names)
    write_json(destination / "identity-after.json", after)
    changes = source_identity_changes(before, after)
    stable_snapshot = snapshot_unchanged(source_snapshot)
    stable_tools = frozen_tools_unchanged(tools)
    passed = (
        all(record["status"] == "PASS" for record in records)
        and not changes["fields"]
        and stable_snapshot
        and stable_tools
    )
    write_json(
        destination / "images.json",
        {
            "source_kernels": str(original_root),
            "cases": records,
            "source_identity_changes": changes,
            "stable_source_identity": not changes["fields"],
            "source_snapshot": source_snapshot,
            "stable_source_snapshot": stable_snapshot,
            "stable_tool_identity": stable_tools,
            "status": "PASS" if passed else "INCOMPLETE",
        },
    )
    return 0 if passed else 1


def performance(args):
    """Summarize already collected queued runs; never submit board jobs."""
    manifest = json.loads(args.manifest.read_text())
    records = []
    grouped = {}
    comparisons = {}
    for run_spec in manifest["runs"]:
        result_path = (args.manifest.parent / run_spec["result"]).resolve()
        result = json.loads(result_path.read_text())
        cycles = result.get("kernel_cycles", [])
        if result.get("status") != "PASS" or not cycles:
            raise ValueError(
                "performance requires complete passing board evidence: "
                + str(result_path)
            )
        task_path = result_path.parent / "task.json"
        image_path = result_path.parent / (result["case"] + ".bin")
        if (
            not task_path.is_file()
            or not image_path.is_file()
            or digest(image_path) != result["image_sha256"]
        ):
            raise ValueError("missing/mismatched image and scheduler evidence")
        task = json.loads(task_path.read_text())
        details = task.get("result", {}).get("details", {})
        if details.get("image_sha256") != result["image_sha256"]:
            raise ValueError("scheduler executed another image")
        route = run_spec["route"]
        if route not in ("ddr", "spm", "acc"):
            raise ValueError("unknown measured route: " + route)
        code = (args.manifest.parent / run_spec["route_evidence"]).resolve()
        if not code.is_file() or digest(code) != run_spec["route_evidence_sha256"]:
            raise ValueError("missing/stale generated route evidence")
        text = code.read_text()
        required = run_spec.get("required_route_fragments", [])
        if not required or any(fragment not in text for fragment in required):
            raise ValueError(
                "generated route evidence lacks required accumulator markers"
            )
        if any(
            fragment in text
            for fragment in run_spec.get("forbidden_route_fragments", [])
        ):
            raise ValueError(
                "generated route evidence contains a forbidden accumulator path"
            )
        reference_path = result_path.parent / "accumulator-manifest.json"
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text())
            input_identity = {
                "shape": reference["shape"],
                "rounds": [
                    {
                        "inputs_sha256": entry["inputs_sha256"],
                        "output_sha256": entry["output_sha256"],
                    }
                    for entry in reference["rounds"]
                ],
                "orders": reference["orders"],
            }
        else:
            reference_path = result_path.parent / "golden_manifest.json"
            if not reference_path.is_file():
                raise ValueError(
                    "missing independently captured input/reference identity"
                )
            reference = json.loads(reference_path.read_text())
            input_identity = {"rounds": reference["rounds"]}
        if not run_spec.get("cache_policy"):
            raise ValueError(
                "performance manifest must identify the compiled cache policy"
            )
        if result.get("process_returncode") != 0:
            raise ValueError("performance run client did not finish successfully")
        started = datetime.fromisoformat(result["started_utc"])
        finished = datetime.fromisoformat(result["finished_utc"])
        if finished < started:
            raise ValueError("invalid board execution timestamps")
        record = {
            **run_spec,
            "case": result["case"],
            "task_id": result["task_id"],
            "image_sha256": result["image_sha256"],
            "cycles": cycles,
            "node": result.get("node"),
            "fpga": result.get("fpga"),
            "board_config": result.get("board_config"),
            "input_identity": input_identity,
            "started_utc": result["started_utc"],
            "finished_utc": result["finished_utc"],
        }
        records.append(record)
        comparison = run_spec["comparison"]
        if comparison not in ("ddr-spm", "ddr-acc"):
            raise ValueError("comparison must be ddr-spm or ddr-acc")
        comparisons.setdefault((result["case"], comparison), []).append(record)
        for index, value in enumerate(cycles):
            grouped.setdefault((result["case"], route, index), []).append(value)
    deltas = []
    for (name, comparison), runs in sorted(comparisons.items()):
        baseline, candidate = comparison.split("-")
        if (
            len(runs) < 4
            or len(runs) % 2
            or [r["route"] for r in runs] != [baseline, candidate] * (len(runs) // 2)
        ):
            raise ValueError(
                "comparison requires actual A/B/A/B alternating runs: " + comparison
            )
        if any(
            datetime.fromisoformat(first["finished_utc"])
            > datetime.fromisoformat(second["started_utc"])
            for first, second in zip(runs, runs[1:])
        ):
            raise ValueError("performance runs are out of execution order or overlap")
        equivalence = ("input_identity", "board_config", "cache_policy", "node", "fpga")
        if any(
            any(run[key] != runs[0][key] for key in equivalence) for run in runs[1:]
        ):
            raise ValueError(
                "performance variants differ in input/grid/cache/window/hardware"
            )
        if len({r["task_id"] for r in runs}) != len(runs):
            raise ValueError("a board task was counted more than once")
        for route in (baseline, candidate):
            if len({r["image_sha256"] for r in runs if r["route"] == route}) != 1:
                raise ValueError(
                    "one route's image changed between benchmark repetitions"
                )
        for invocation in range(len(runs[0]["cycles"])):
            values = {
                route: [r["cycles"][invocation] for r in runs if r["route"] == route]
                for route in (baseline, candidate)
            }
            delta = statistics.median(values[baseline]) - statistics.median(
                values[candidate]
            )
            spread = max(max(values[route]) - min(values[route]) for route in values)
            deltas.append(
                {
                    "case": name,
                    "comparison": comparison,
                    "invocation": invocation,
                    "median_saved_cycles": delta,
                    "median_speedup": statistics.median(values[baseline])
                    / statistics.median(values[candidate]),
                    "largest_observed_spread": spread,
                    "improvement_exceeds_spread": delta > spread,
                }
            )
    summaries = []
    for (name, route, invocation), values in sorted(grouped.items()):
        if len(values) < 2:
            raise ValueError("at least two samples per route/invocation are required")
        summaries.append(
            {
                "case": name,
                "route": route,
                "invocation": invocation,
                "samples": values,
                "minimum": min(values),
                "maximum": max(values),
                "median": statistics.median(values),
                "spread": max(values) - min(values),
            }
        )
    report = {
        "schema": 1,
        "measurement_window": "complete typed adapter including grid/DMA/final fence",
        "runs_in_execution_order": records,
        "summaries": summaries,
        "comparisons": deltas,
        "note": "A delta within observed run-to-run spread is not evidence of speedup; input/cache/route equivalence must be reviewed.",
    }
    write_json(args.output, report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, handler in (
        ("reproduce", reproduce),
        ("build", build),
        ("simulate", simulate),
    ):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--output", type=Path, required=True)
        if name != "build":
            command.add_argument(
                "--kernels", type=Path, default=SUITE / "build/kernels"
            )
        if name != "reproduce":
            command.add_argument("--case", action="append")
            command.add_argument("--all", action="store_true")
        if name == "build":
            command.add_argument("--bin", action="store_true")
            command.add_argument(
                "--accumulator-board",
                action="store_true",
                help="Use three-round bit-exact guard/input diagnostic for bin builds",
            )
            command.add_argument("--jobs", type=int, default=4)
            command.add_argument("--accumulator-round", type=int, choices=(0, 1, 2))
            command.add_argument(
                "--freeze-tools",
                action="store_true",
                help="Copy compiler executables before any build stage",
            )
    command = commands.add_parser(
        "images",
        help="Build production/diagnostic images from the same frozen kernel objects",
    )
    command.set_defaults(handler=images)
    command.add_argument("--kernels", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--case", action="append")
    command.add_argument("--all", action="store_true")
    command.add_argument("--accumulator-board", action="store_true")
    command.add_argument("--accumulator-round", type=int, choices=(0, 1, 2))
    command = commands.add_parser(
        "performance", help="Collect already completed board runs"
    )
    command.set_defaults(handler=performance)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        getattr(args, "accumulator_round", None) is not None
        and not args.accumulator_board
    ):
        parser.error("--accumulator-round requires --accumulator-board")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
