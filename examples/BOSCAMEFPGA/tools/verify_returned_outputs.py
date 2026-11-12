#!/usr/bin/env python3
"""Replay downloaded operator outputs through the unchanged native C oracle.

The typed entry only loads exact output snapshots. It computes no replacement
kernel results. Two calls of the complete original launch() run its input
initialization, every invocation/trial, and every original numerical check.
LOCAL_ORACLE_PASS describes the supplied files; scheduler provenance and board
execution status must be established separately from the job/artifact record.
Single-job manifests use {id, artifacts}. Schema version 2 binds selected
invocations from separate jobs to preserved task/config/image files before
the same complete original oracle runs over all merged output snapshots.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


DTYPES = {"f32": ("<f4", 4), "i32": ("<i4", 4), "i8": ("|i1", 1)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_spec(model, case_name, config_path):
    case = load_module(model / "cases.py", "returned_output_cases").describe(case_name)
    output_spec = load_module(model / "tools/output_spec.py", "returned_output_spec")
    output_arguments = tuple(output_spec.output_indices(case_name))
    config = json.loads(config_path.read_text())
    if config.get("case") != case_name:
        raise ValueError("board config case does not match --case")
    invocations = config.get("expected_invocations")
    if type(invocations) is not int or invocations < 1:
        raise ValueError("expected_invocations must be a positive integer")
    outputs = config.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("board config has no exported outputs")
    observed = set()
    names = set()
    for item in outputs:
        invocation, argument = item.get("invocation"), item.get("argument")
        if type(invocation) is not int or not 0 <= invocation < invocations:
            raise ValueError("output invocation is out of range")
        if type(argument) is not int or argument not in output_arguments:
            raise ValueError("output argument is not declared by output_spec.py")
        arg = case["arguments"][argument]
        expected_name = f"r{invocation}_{arg['name']}"
        if item.get("name") != expected_name or not re.fullmatch(
            r"[A-Za-z0-9_]+", expected_name
        ):
            raise ValueError(f"output snapshot must be named {expected_name!r}")
        dtype, itemsize = DTYPES[arg["dtype"]]
        if item.get("dtype") not in ({"<i1", "|i1"} if dtype == "|i1" else {dtype}):
            raise ValueError(f"{expected_name}: dtype does not match typed entry")
        shape = item.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != arg["rank"]
            or any(type(extent) is not int or extent <= 0 for extent in shape)
        ):
            raise ValueError(f"{expected_name}: invalid output shape")
        if "argument_shapes" in case and shape != case["argument_shapes"][argument]:
            raise ValueError(
                f"{expected_name}: output shape differs from kernel config"
            )
        nbytes = math.prod(shape) * itemsize
        if type(item.get("nbytes")) is not int or item["nbytes"] != nbytes:
            raise ValueError(f"{expected_name}: nbytes does not match shape/dtype")
        if (invocation, argument) in observed or expected_name in names:
            raise ValueError("duplicate output snapshot")
        observed.add((invocation, argument))
        names.add(expected_name)
    expected = {
        (r, argument) for r in range(invocations) for argument in output_arguments
    }
    if observed != expected:
        raise ValueError("config must export every output for every invocation")
    return case, config


def verify_split_jobs(artifacts, manifest, config, metadata):
    """Validate each selected invocation against its preserved scheduler task."""
    if config is None or manifest.get("case") != config["case"]:
        raise ValueError("split manifest case must match the complete oracle config")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != config["expected_invocations"]:
        raise ValueError("split manifest must identify one real job per invocation")
    ids, invocations, sources = set(), set(), {}
    verified = []
    for job in jobs:
        ident, invocation = job.get("id"), job.get("invocation")
        if (
            not isinstance(ident, str)
            or not re.fullmatch(r"[0-9a-f]{32}", ident)
            or ident in ids
            or type(invocation) is not int
            or not 0 <= invocation < config["expected_invocations"]
            or invocation in invocations
        ):
            raise ValueError(
                "split jobs must have unique scheduler IDs and invocation indices"
            )
        ids.add(ident)
        invocations.add(invocation)
        directory = artifacts / "provenance" / ident
        task_path, config_path, image_path = (
            directory / name for name in ("task.json", "board_config.json", "image.bin")
        )
        hashes = {
            str(path): digest(path) for path in (task_path, config_path, image_path)
        }
        sources.update(hashes)
        for key, path in (
            ("task_json_sha256", task_path),
            ("config_sha256", config_path),
            ("image_sha256", image_path),
        ):
            if job.get(key) != hashes[str(path)]:
                raise ValueError(
                    f"{ident}: preserved task/config/image hash differs from manifest"
                )
        task, selected = (
            json.loads(task_path.read_text()),
            json.loads(config_path.read_text()),
        )
        result = task.get("result") or {}
        details = result.get("details") or {}
        if (
            task.get("id") != ident
            or task.get("state") != "completed"
            or result.get("execution_status") != "exited"
            or result.get("exit_code") != 0
            or result.get("app_exit_code") != 0
            or details.get("reset_confirmed") is not True
            or details.get("application_started") is not True
            or details.get("ddr_readback_matches") is not True
            or details.get("image_sha256") != hashes[str(image_path)]
            or details.get("artifact_error")
            or details.get("artifacts_skipped")
        ):
            raise ValueError(
                f"{ident}: scheduler task did not complete with verified image and artifacts"
            )
        for key in (
            "case",
            "expected_invocations",
            "launch_source_sha256",
            "host_support_sha256",
        ):
            if key not in config or selected.get(key) != config[key]:
                raise ValueError(
                    f"{ident}: split config differs from complete oracle config ({key})"
                )
        expected = {
            item["name"]: item
            for item in config["outputs"]
            if item["invocation"] == invocation
        }
        actual = selected.get("outputs")
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(
                f"{ident}: split config must capture every selected output"
            )
        selected_names = set()
        for output in actual:
            name = output.get("name")
            if name not in expected or name in selected_names:
                raise ValueError(f"{ident}: unexpected or duplicate selected output")
            selected_names.add(name)
            for key in ("argument", "invocation", "dtype", "shape", "nbytes"):
                if output.get(key) != expected[name][key]:
                    raise ValueError(
                        f"{ident}: selected output {name} differs from complete config"
                    )
        declared = result.get("artifacts")
        if not isinstance(declared, list) or len(declared) != len(expected):
            raise ValueError(
                f"{ident}: task artifacts do not cover selected invocation"
            )
        found = set()
        for item in declared:
            name = item.get("name")
            if (
                name not in expected
                or name in found
                or metadata[name].get("task_id") != ident
            ):
                raise ValueError(
                    f"{ident}: artifact is not bound to its selected real task"
                )
            found.add(name)
            if metadata[name] != {**item, "task_id": ident}:
                raise ValueError(
                    f"{ident}: manifest artifact differs from actual task metadata"
                )
        uart = result.get("uart_output") or result.get("output") or ""
        cycles = [
            int(v, 16) for v in re.findall(r"kernel_cycles=0x([0-9a-fA-F]+)", uart)
        ]
        ordinals = [
            int(v) for v in re.findall(r"invocation=(\d+)\s+kernel_cycles=", uart)
        ]
        if len(cycles) != config["expected_invocations"] or ordinals != list(
            range(config["expected_invocations"])
        ):
            raise ValueError(
                f"{ident}: split task did not run every original kernel invocation"
            )
        outputs = [
            json.loads(v) for v in re.findall(r"@@FPGA_OUTPUT\s+(\{[^\r\n]*\})", uart)
        ]
        if len(outputs) != len(expected) or {o.get("name") for o in outputs} != set(
            expected
        ):
            raise ValueError(
                f"{ident}: UART does not declare the complete selected output set"
            )
        for output in outputs:
            name = output["name"]
            if output.get("version") != 1 or any(
                output.get(key) != metadata[name].get(key)
                for key in ("address", "dtype", "shape", "nbytes")
            ):
                raise ValueError(
                    f"{ident}: UART output differs from scheduler artifact"
                )
        finishes = [
            json.loads(v) for v in re.findall(r"@@FPGA_RESULT\s+(\{[^\r\n]*\})", uart)
        ]
        if finishes != [{"version": 1, "status": "exited", "app_exit_code": 0}]:
            raise ValueError(f"{ident}: missing valid UART result")
        verified.append(
            {
                **job,
                "fpga": task.get("fpga"),
                "kernel_cycles": cycles,
                "provenance_directory": str(directory),
                "state": task["state"],
            }
        )
    if {item.get("task_id") for item in metadata.values()} != ids:
        raise ValueError(
            "split artifact references an unknown or unused scheduler task"
        )
    return {
        "scheduler_job_id": None,
        "scheduler_job_ids": [j["id"] for j in jobs],
        "verified_job_provenance": verified,
        "provenance_source_sha256": sources,
    }


def verify_manifest(artifacts, outputs, inputs, config=None):
    """Bind locally read bytes to the scheduler artifact metadata supplied."""
    path = artifacts / "manifest.json"
    manifest = json.loads(path.read_text())
    split = manifest.get("schema_version") == 2
    job = manifest.get("id")
    if not split and (not isinstance(job, str) or not job):
        raise ValueError("artifact manifest has no scheduler job id")
    items = manifest.get("artifacts")
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ValueError("artifact manifest must contain a list of artifacts")
    metadata = {item.get("name"): item for item in items}
    if len(metadata) != len(items) or set(metadata) != {o["name"] for o in outputs}:
        raise ValueError(
            "artifact manifest names differ from the complete output config"
        )
    for output, local in zip(outputs, inputs):
        declared = metadata[output["name"]]
        for key in ("shape", "nbytes"):
            if declared.get(key) != output[key]:
                raise ValueError(f"{output['name']}: scheduler artifact {key} mismatch")
        expected_dtype = "|i1" if output["dtype"] == "<i1" else output["dtype"]
        declared_dtype = (
            "|i1" if declared.get("dtype") == "<i1" else declared.get("dtype")
        )
        if expected_dtype != declared_dtype:
            raise ValueError(f"{output['name']}: scheduler artifact dtype mismatch")
        if declared.get("sha256") != local["sha256"]:
            raise ValueError(
                f"{output['name']}: local BIN SHA differs from scheduler artifact"
            )
    provenance = (
        verify_split_jobs(artifacts, manifest, config, metadata)
        if split
        else {"scheduler_job_id": job}
    )
    return {
        **provenance,
        "artifact_manifest_verified": True,
        "artifact_manifest": str(path),
        "artifact_manifest_sha256": digest(path),
    }


def wrapper_source(case, config):
    arguments = case["arguments"]
    params = ", ".join(f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(arguments))
    lines = [
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "static const char *artifact_directory;",
        "static unsigned invocations;",
        "static size_t loaded_bytes, loaded_tensors;",
        'static void require(int ok, const char *what) { if (!ok) { fprintf(stderr, "RETURNED_OUTPUT_FAIL %s\\n", what); exit(2); } }',
        "static void read_output(const char *name, void *destination, size_t bytes) {",
        "  char path[8192];",
        '  int length = snprintf(path, sizeof(path), "%s/%s.bin", artifact_directory, name);',
        '  require(length >= 0 && (size_t)length < sizeof(path), "artifact path too long");',
        '  FILE *file = fopen(path, "rb");',
        '  if (!file) { fprintf(stderr, "RETURNED_OUTPUT_FAIL missing artifact %s\\n", path); exit(2); }',
        '  require(fseek(file, 0, SEEK_END) == 0, "artifact seek failed");',
        "  long actual = ftell(file);",
        '  if (actual < 0 || (unsigned long long)actual != (unsigned long long)bytes) { fprintf(stderr, "RETURNED_OUTPUT_FAIL artifact length %s expected=%zu actual=%ld\\n", name, bytes, actual); fclose(file); exit(2); }',
        '  require(fseek(file, 0, SEEK_SET) == 0, "artifact rewind failed");',
        '  require(fread(destination, 1, bytes, file) == bytes, "artifact short read");',
        '  require(fgetc(file) == EOF && !ferror(file), "artifact grew or read failed");',
        '  require(fclose(file) == 0, "artifact close failed");',
        "  loaded_bytes += bytes; ++loaded_tensors;",
        "}",
        f"void _mlir_ciface_kernel_{case['name']}({params}) {{",
        f'  require(invocations < {config["expected_invocations"]}u, "excess kernel invocations");',
        "  switch (invocations) {",
    ]
    for invocation in range(config["expected_invocations"]):
        lines.append(f"  case {invocation}: {{")
        for output in config["outputs"]:
            if output["invocation"] != invocation:
                continue
            i, name, shape = output["argument"], output["name"], output["shape"]
            checks = [
                f"a{i}",
                f"a{i}->allocated",
                f"a{i}->allocated == a{i}->aligned",
                f"a{i}->offset == 0",
            ]
            stride = 1
            for axis in reversed(range(len(shape))):
                checks.extend(
                    (
                        f"a{i}->sizes[{axis}] == {shape[axis]}",
                        f"a{i}->strides[{axis}] == {stride}",
                    )
                )
                stride *= shape[axis]
            lines.append(
                f'    require({" && ".join(checks)}, "{name}: descriptor shape/stride/offset");'
            )
            lines += [
                f"    uintptr_t p_{i} = (uintptr_t)a{i}->aligned, base_{i} = (uintptr_t)workspace(0);",
                f'    require({output["nbytes"]}ull <= VIT_WORKSPACE_BYTES && p_{i} >= base_{i} && p_{i} - base_{i} <= VIT_WORKSPACE_BYTES - {output["nbytes"]}ull, "{name}: output outside workspace");',
                f'    read_output("{name}", a{i}->aligned, {output["nbytes"]}u);',
            ]
        lines.append("    break; }")
    lines += [
        '  default: require(0, "invalid invocation");',
        "  }",
        "  ++invocations;",
        "}",
        "extern int launch(void);",
        "int main(int argc, char **argv) {",
        '  require(argc == 2, "expected artifact directory argument");',
        "  artifact_directory = argv[1];",
        "  for (unsigned trial = 0; trial < 2; ++trial) {",
        "    invocations = 0; loaded_bytes = 0; loaded_tensors = 0;",
        "    int result = launch();",
        '    if (result) { fprintf(stderr, "RETURNED_OUTPUT_ORACLE_FAIL trial=%u launch_result=%d\\n", trial, result); return 1; }',
        f'    require(invocations == {config["expected_invocations"]}u, "missing kernel invocations");',
        f'    require(loaded_tensors == {len(config["outputs"])}u, "missing output snapshots");',
        '    printf("ORACLE_REPLAY_TRIAL_PASS trial=%u invocations=%u tensors=%zu bytes=%zu\\n", trial, invocations, loaded_tensors, loaded_bytes);',
        "  }",
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def run_native(executable, artifacts, output, label, timeout):
    command = [str(executable), str(artifacts)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (output / f"{label}.stdout").write_text(completed.stdout)
    (output / f"{label}.stderr").write_text(completed.stderr)
    trials = [
        {
            "trial": int(trial),
            "invocations": int(calls),
            "tensors": int(tensors),
            "nbytes": int(nbytes),
        }
        for trial, calls, tensors, nbytes in re.findall(
            r"^ORACLE_REPLAY_TRIAL_PASS trial=(\d+) invocations=(\d+) tensors=(\d+) bytes=(\d+)$",
            completed.stdout,
            re.M,
        )
    ]
    return {
        "command": command,
        "exit_code": completed.returncode,
        "trials": trials,
        "stdout": str(output / f"{label}.stdout"),
        "stderr": str(output / f"{label}.stderr"),
    }


def regression_checks(executable, artifacts, outputs, destination, timeout):
    """Exercise the native reader and original oracle without editing inputs."""
    records = []
    first = outputs[0]
    for kind in ("missing_file", "short_file", "corrupt_data"):
        directory = destination / "regressions" / kind
        directory.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            target = directory / (output["name"] + ".bin")
            if target.is_symlink() or target.is_file():
                target.unlink()
            if output is not first:
                target.symlink_to((artifacts / target.name).resolve())
        target = directory / (first["name"] + ".bin")
        original = artifacts / target.name
        if kind == "short_file":
            target.write_bytes(original.read_bytes()[:-1])
        elif kind == "corrupt_data":
            contents = bytearray(original.read_bytes())
            if first["dtype"] == "<f4":
                contents[:4] = b"\x00\x00\xc0\x7f"  # quiet NaN must fail FP oracle
            else:
                contents[0] ^= 1
            target.write_bytes(contents)
        result = run_native(executable, directory, destination, kind, timeout)
        result.update(
            {
                "kind": kind,
                "expected_failure": True,
                "status": "PASS"
                if result["exit_code"] == (1 if kind == "corrupt_data" else 2)
                else "FAIL",
            }
        )
        records.append(result)
    return records


def verify(
    model,
    case_name,
    artifacts,
    config_path,
    output,
    cc="cc",
    timeout=120,
    evidence_kind="returned_artifacts",
    regressions=True,
):
    model, artifacts, config_path, output = map(
        lambda p: Path(p).resolve(), (model, artifacts, config_path, output)
    )
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "model": str(model),
        "case": case_name,
        "status": "FAIL",
        "evidence_kind": evidence_kind,
        "board_execution_verified": False,
        "artifact_manifest_verified": False,
        "scheduler_job_id": None,
        "validation": "unchanged native launch oracle over supplied binary output snapshots",
        "native_replay_trials": 2,
        "oracle_executed": False,
        "returned_artifacts_compared": False,
        "artifacts": str(artifacts),
        "config": str(config_path),
    }
    try:
        if evidence_kind not in ("returned_artifacts", "simulated_selftest"):
            raise ValueError("unknown evidence kind")
        if sys.byteorder != "little":
            raise ValueError("native replay requires a little-endian host")
        case, config = load_spec(model, case_name, config_path)
        record["expected_invocations"] = config["expected_invocations"]
        record["config_sha256"] = digest(config_path)
        record["sources"] = {
            str(p): digest(p)
            for p in (
                Path(case["launch_source"]),
                model / "support.c",
                model / "support.h",
            )
        }
        for key, path in (
            ("launch_source_sha256", Path(case["launch_source"])),
            ("host_support_sha256", model / "support.c"),
        ):
            if key in config and config[key] != record["sources"][str(path)]:
                raise ValueError(f"{path}: source differs from frozen board config")
        inputs = []
        for item in config["outputs"]:
            path = artifacts / (item["name"] + ".bin")
            if path.stat().st_size != item["nbytes"]:
                raise ValueError(
                    f"{path}: exact artifact length must be {item['nbytes']}"
                )
            inputs.append({**item, "path": str(path), "sha256": digest(path)})
        record["outputs"] = inputs
        if evidence_kind == "returned_artifacts":
            record.update(verify_manifest(artifacts, config["outputs"], inputs, config))
        source = output / "returned_output_wrapper.c"
        source.write_text(wrapper_source(case, config))
        executable = output / "verify-returned-outputs"
        compiler = shutil.which(cc)
        if not compiler:
            raise ValueError("native C compiler not found: " + cc)
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-DHOST_TEST",
            "-I",
            str(model),
            "-I",
            str(model / "tools"),
            str(case["launch_source"]),
            str(model / "support.c"),
            str(source),
            "-lm",
            "-o",
            str(executable),
        ]
        compiled = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        (output / "compile.stdout").write_text(compiled.stdout)
        (output / "compile.stderr").write_text(compiled.stderr)
        record["compile"] = {"command": command, "exit_code": compiled.returncode}
        if compiled.returncode:
            raise RuntimeError("native oracle build failed; see compile.stderr")
        record["wrapper_sha256"] = digest(source)
        record["executable_sha256"] = digest(executable)
        replay = run_native(executable, artifacts, output, "replay", timeout)
        record["replay"] = replay
        if replay["exit_code"] or [t["trial"] for t in replay["trials"]] != [0, 1]:
            raise RuntimeError(
                "original C oracle or output loading failed; see replay logs"
            )
        if any(
            t["invocations"] != config["expected_invocations"]
            or t["tensors"] != len(config["outputs"])
            or t["nbytes"] != sum(o["nbytes"] for o in config["outputs"])
            for t in replay["trials"]
        ):
            raise RuntimeError("native replay coverage differs from exported outputs")
        record["oracle_executed"] = True
        record["returned_artifacts_compared"] = evidence_kind == "returned_artifacts"
        if regressions:
            record["regressions"] = regression_checks(
                executable, artifacts, config["outputs"], output, timeout
            )
            if any(r["status"] != "PASS" for r in record["regressions"]):
                raise RuntimeError(
                    "negative regression failed to reject invalid output"
                )
        # Input changes during replay invalidate the recorded evidence.
        if any(digest(item["path"]) != item["sha256"] for item in inputs):
            raise RuntimeError("artifact changed during native replay")
        if any(digest(path) != sha for path, sha in record["sources"].items()):
            raise RuntimeError("original oracle source changed during native replay")
        if (
            record["artifact_manifest_verified"]
            and digest(record["artifact_manifest"])
            != record["artifact_manifest_sha256"]
        ):
            raise RuntimeError("artifact manifest changed during native replay")
        if any(
            digest(path) != sha
            for path, sha in record.get("provenance_source_sha256", {}).items()
        ):
            raise RuntimeError(
                "split-job scheduler provenance changed during native replay"
            )
        record["status"] = (
            "LOCAL_ORACLE_PASS"
            if evidence_kind == "returned_artifacts"
            else "SIMULATED_SELFTEST_PASS"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        record["reason"] = str(exc)
    (output / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--evidence-kind",
        choices=("returned_artifacts", "simulated_selftest"),
        default="returned_artifacts",
    )
    parser.add_argument("--skip-regressions", action="store_true")
    args = parser.parse_args()
    config = args.config
    if config is None:
        candidates = (
            args.artifacts / "board_config.json",
            args.model / "build/kernels" / args.case / "board/board_config.json",
        )
        config = next((p for p in candidates if p.is_file()), candidates[0])
    record = verify(
        args.model,
        args.case,
        args.artifacts,
        config,
        args.output,
        args.cc,
        args.timeout,
        args.evidence_kind,
        not args.skip_regressions,
    )
    print(record["status"], args.case, record.get("reason", ""))
    return (
        0 if record["status"] in ("LOCAL_ORACLE_PASS", "SIMULATED_SELFTEST_PASS") else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
