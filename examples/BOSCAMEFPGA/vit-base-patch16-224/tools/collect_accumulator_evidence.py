#!/usr/bin/env python3
"""Collect existing diagnostic or calibration evidence without board access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct

from accumulator_validation import digest, write_json
from board_accumulator import fnv64

ROUND_PATTERN = re.compile(
    r"validation_round=(\d+) checked=(\d+) errors=(\d+) guard_errors=(\d+) "
    r"input_errors=(\d+) output_fnv64=0x([0-9a-fA-F]+)"
)
CALL_PATTERN = re.compile(
    r"\[acc-path\] invocation=(\d+) variant=(ddr|spm|acc) stress=([01]) "
    r"kernel_cycles=0x([0-9a-fA-F]+) errors=(\d+) input_errors=(\d+) "
    r"guard_errors=(\d+) frame_errors=(\d+) output_fnv64=0x([0-9a-fA-F]+)"
)


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calibration_mode(record, label):
    post_add = record.get("post_add")
    require(type(post_add) is bool, f"{label} must declare boolean post_add")
    mode = record.get("mode")
    require(
        mode == ("normal-post-add" if post_add else "overwrite"),
        f"{label} mode and post_add are missing or inconsistent",
    )
    return mode, post_add


def verified_task(path, image, expected_sha256):
    task = read_json(path)
    result = task["result"]
    require(digest(image) == expected_sha256, "image identity changed")
    require(task["state"] == "completed", "scheduler task is not complete")
    require(
        result["execution_status"] == "exited"
        and result["app_exit_code"] == 0
        and result.get("exit_code") == 0,
        "scheduler did not record successful application exit",
    )
    require(
        result["details"]["image_sha256"] == expected_sha256,
        "scheduler executed another image",
    )
    require(
        task["started"] > 0 and task["finished"] >= task["started"],
        "scheduler timestamps are incomplete",
    )
    uart = result.get("uart_output")
    if uart is None:
        uart = result.get("output")
    require(bool(uart), "scheduler has no application UART evidence")
    return task, uart


def parsed_rounds(uart):
    return [
        dict(
            round=int(r[0]),
            checked=int(r[1]),
            errors=int(r[2]),
            guard_errors=int(r[3]),
            input_errors=int(r[4]),
            output_fnv64=r[5].lower().zfill(16),
        )
        for r in ROUND_PATTERN.findall(uart)
    ]


def diagnostic_result(path):
    path = path.resolve()
    record = read_json(path)
    if record["status"] != "PASS":
        return record, None
    require(
        record.get("validation_mode") == "accumulator_three_round_bit_exact",
        "diagnostic result used another validation mode",
    )
    directory = path.parent
    image = directory / (record["case"] + ".bin")
    task, uart = verified_task(directory / "task.json", image, record["image_sha256"])
    require(task["id"] == record["task_id"], "result references another scheduler task")
    require(
        parsed_rounds(uart) == record["validation_rounds"],
        "stored round records differ from scheduler UART",
    )
    kernel = directory / "build-evidence/kernel.o"
    require(
        digest(kernel) == record["build_evidence_sha256"]["kernel.o"],
        "diagnostic kernel object changed",
    )
    manifest = read_json(directory / "accumulator-manifest.json")
    config = record["board_config"]
    round_ids = manifest.get("round_ids", [0, 1, 2])
    require(
        round_ids == config.get("validation_round_ids", [0, 1, 2]),
        "reference round IDs differ from board configuration",
    )
    require(
        [r["round"] for r in record["validation_rounds"]] == round_ids,
        "selected diagnostic rounds are missing or duplicated",
    )
    require(
        round_ids in ([0], [1], [2], [0, 1, 2]), "invalid selected diagnostic rounds"
    )
    require(
        record["kernel_returned"] == len(round_ids) == len(record["kernel_cycles"]),
        "selected diagnostic invocation coverage is incomplete",
    )
    count = manifest["shape"][0] * manifest["shape"][1]
    require(
        record["checked"] == count * len(round_ids) and record["errors"] == 0,
        "diagnostic total output coverage is incomplete",
    )
    references = {entry["name"]: entry for entry in manifest["rounds"]}
    # Each reference is physically embedded at the linker-defined offset;
    # manifests alone cannot turn a production image into a diagnostic image.
    offset = 0x100000
    with image.open("rb") as binary:
        for entry in manifest["rounds"]:
            binary.seek(offset)
            data = binary.read(entry["bytes"])
            require(
                hashlib.sha256(data).hexdigest() == entry["output_sha256"]
                and fnv64(data) == entry["output_fnv64"],
                "embedded independent output reference changed",
            )
            offset = (offset + len(data) + 63) & ~63
    for validation in record["validation_rounds"]:
        reference = references["normal" if validation["round"] == 0 else "stress"]
        require(
            validation["checked"] == count
            and validation["errors"]
            == validation["guard_errors"]
            == validation["input_errors"]
            == 0
            and validation["output_fnv64"] == reference["output_fnv64"],
            "round lacks complete bit-exact output/input/guard evidence",
        )
    return record, {
        "manifest": manifest,
        "kernel_sha256": digest(kernel),
        "task": task,
        "result": str(path),
    }


def diagnostics(args):
    grouped = {}
    for path in args.result:
        record, evidence = diagnostic_result(path)
        grouped.setdefault(record["case"], []).append((record, evidence))
    cases = []
    for name, records in sorted(grouped.items()):
        rounds, runs, excluded, identities = {}, [], [], []
        for record, evidence in records:
            if evidence is None:
                excluded.append(
                    {"status": record["status"], "task_id": record.get("task_id")}
                )
                continue
            manifest = evidence["manifest"]
            identities.append(
                (
                    evidence["kernel_sha256"],
                    manifest["kernel_source_sha256"],
                    manifest["shape"],
                )
            )
            runs.append(
                {
                    "result": evidence["result"],
                    "task_id": record["task_id"],
                    "image_sha256": record["image_sha256"],
                    "kernel_sha256": evidence["kernel_sha256"],
                }
            )
            for validation, cycles in zip(
                record["validation_rounds"], record["kernel_cycles"]
            ):
                round_id = validation["round"]
                require(
                    round_id not in rounds,
                    "a diagnostic round was counted twice: " + name,
                )
                rounds[round_id] = dict(
                    validation, kernel_cycles=cycles, task_id=record["task_id"]
                )
        require(
            not identities or all(identity == identities[0] for identity in identities),
            "split diagnostic images use different kernel objects/source/shape",
        )
        complete = sorted(rounds) == [0, 1, 2]
        if complete:
            require(
                rounds[1]["output_fnv64"] == rounds[2]["output_fnv64"],
                "repeated stress rounds produced different output bits",
            )
        cases.append(
            {
                "case": name,
                "status": "PASS" if complete else "INCOMPLETE",
                "rounds": [rounds[index] for index in sorted(rounds)],
                "runs": runs,
                "excluded_failures": excluded,
            }
        )
    complete = bool(cases) and all(case["status"] == "PASS" for case in cases)
    write_json(
        args.output,
        {
            "schema": 1,
            "status": "PASS" if complete else "INCOMPLETE",
            "scope": "three complete board diagnostic rounds per listed case",
            "cases": cases,
        },
    )
    return 0 if complete else 1


def calibration_schedule(record, label, three_routes=True):
    """Only an explicit split may shorten the canonical full schedule."""
    fields = {"calls", "stress", "stress_schedule"}
    present = fields.intersection(record)
    width = 8 if three_routes else 4
    if not present:
        # Historical full images retain their fixed two-stress contract.
        return None, [0] * width + [1] * width
    require(present == fields, f"{label} has incomplete stress selection")
    stress = record["stress"]
    require(
        stress is None or (type(stress) is int and stress in (0, 1)),
        f"{label} stress must be null, integer 0 or integer 1",
    )
    require(three_routes or stress is None, "two-route calibration cannot split stress")
    selected = [0, 1] if stress is None else [stress]
    schedule = [value for value in selected for _ in range(width)]
    require(
        type(record["calls"]) is int and record["calls"] == len(schedule),
        f"{label} calls differ from its complete stress selection",
    )
    require(
        isinstance(record["stress_schedule"], list)
        and all(type(value) is int for value in record["stress_schedule"])
        and record["stress_schedule"] == schedule,
        f"{label} stress_schedule differs from its complete stress selection",
    )
    return stress, schedule


def calibration_calls(uart, expected_hashes, three_routes=False, stress_schedule=None):
    calls = []
    for entry in CALL_PATTERN.findall(uart):
        calls.append(
            dict(
                invocation=int(entry[0]),
                route=entry[1],
                stress=int(entry[2]),
                cycles=int(entry[3], 16),
                errors=int(entry[4]),
                input_errors=int(entry[5]),
                guard_errors=int(entry[6]),
                frame_errors=int(entry[7]),
                output_fnv64=entry[8].lower().zfill(16),
            )
        )
    order = ["ddr", "spm", "ddr", "spm"]
    if three_routes:
        order += ["ddr", "acc", "ddr", "acc"]
    if stress_schedule is None:
        stress_schedule = [0] * len(order) + [1] * len(order)
    expected_count = len(stress_schedule)
    require(
        len(calls) == expected_count, "calibration invocation coverage is incomplete"
    )
    for index, call in enumerate(calls):
        stress = stress_schedule[index]
        require(
            call["invocation"] == index
            and call["route"] == order[index % len(order)]
            and call["stress"] == stress,
            "calibration schedule is not the declared complete stress ABAB",
        )
        require(
            call["cycles"] > 0
            and call["errors"]
            == call["input_errors"]
            == call["guard_errors"]
            == call["frame_errors"]
            == 0,
            "calibration invocation has invalid cycles or validation errors",
        )
        require(
            call["output_fnv64"] == expected_hashes[stress],
            "calibration output hash differs from independent golden",
        )
    summary = re.findall(
        r"\[acc-path\] returned=(\d+) validated=(\d+) errors=(\d+) "
        r"frame_acquires=0x([0-9a-fA-F]+) frame_releases=0x([0-9a-fA-F]+) "
        r"frame_peak_bytes=0x([0-9a-fA-F]+)",
        uart,
    )
    require(len(summary) == 1, "calibration lacks one final lifecycle summary")
    final = summary[0]
    require(
        list(map(int, final[:3])) == [expected_count, expected_count, 0]
        and int(final[3], 16) == int(final[4], 16) == expected_count // 2
        and int(final[5], 16) >= 8192,
        "calibration frame lifecycle or validation coverage is incomplete",
    )
    return calls


def normalized_calibration_source(text, *, physical=False, observer=False):
    """Allow only disclosed placement and terminal-observer differences."""
    if physical:
        private_type = "memref<32x64xi32, #bosc_ame.spm<0>>"
        physical_allocation = (
            "%c = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : "
            + private_type
        )
        require(
            text.count(physical_allocation) == 1,
            "unrecognized fixed-SPM source constraint",
        )
        require(
            "memref.dealloc %c" not in text, "fixed-SPM source unexpectedly frees P"
        )
        text = text.replace(
            physical_allocation,
            "%c = memref.alloc() alignment = 64 : memref<32x64xi32>",
        ).replace(private_type, "memref<32x64xi32>")
        text = text.replace(
            "    return", "    memref.dealloc %c : memref<32x64xi32>\n    return"
        )
    if observer:
        terminal_copy = (
            "    %snapshot = memref.reinterpret_cast %arg3 to offset: [0], "
            "sizes: [32, 64], strides: [64, 1]\n"
            "      : memref<*xi32> to memref<32x64xi32>\n"
            "    memref.copy %c, %snapshot : memref<32x64xi32> to memref<32x64xi32>"
        )
        require(
            text.count(terminal_copy) == 1,
            "calibration lacks the disclosed terminal observer",
        )
        text = text.replace(terminal_copy, "")
    return text.split()


def calibration_outputs(manifest, task, uart, calls, directory, artifacts_path):
    """Verify immutable readmem bytes against every independently generated grid."""
    expected_outputs = manifest.get("ddr_outputs", [])
    if not expected_outputs:
        require(
            artifacts_path is None,
            "legacy calibration does not declare readmem outputs",
        )
        return {
            "status": "NOT_DECLARED",
            "scope": "UART validation only for this historical image",
        }
    require(
        artifacts_path is not None,
        "calibration requires downloaded DDR artifacts for host validation",
    )
    artifacts_path = Path(artifacts_path)
    saved = read_json(artifacts_path)
    require(
        saved == task["result"].get("artifacts"),
        "downloaded manifest differs from the recorded task artifacts",
    )
    by_name = {entry["name"]: entry for entry in saved}
    require(
        len(by_name) == len(saved) == len(expected_outputs)
        and set(by_name) == {entry["name"] for entry in expected_outputs},
        "calibration output artifacts are missing, duplicated or unexpected",
    )
    declarations = []
    for line in uart.splitlines():
        if line.startswith("@@FPGA_OUTPUT "):
            declarations.append(json.loads(line[len("@@FPGA_OUTPUT ") :]))
    require(
        len(declarations) == len(expected_outputs),
        "UART output declarations are incomplete",
    )
    declared = {entry["name"]: entry for entry in declarations}
    gold = (directory / "golden.bin").read_bytes()
    reports = []
    for specification in expected_outputs:
        name = specification["name"]
        metadata = by_name[name]
        declaration = declared.get(name, {})
        for key in ("name", "address", "nbytes", "dtype", "shape"):
            require(
                metadata.get(key) == specification[key] == declaration.get(key),
                "DDR artifact metadata differs from image declaration: " + key,
            )
        require(
            declaration.get("version") == 1
            and specification["dtype"] == "<i4"
            and specification["shape"] == [len(calls), 4096]
            and specification["nbytes"] == len(calls) * 4096 * 4,
            "calibration ledger layout is not complete little-endian i32 grids",
        )
        data_path = artifacts_path.parent / (name + ".bin")
        actual = data_path.read_bytes()
        require(
            len(actual) == metadata["nbytes"]
            and digest(data_path) == metadata["sha256"],
            "DDR artifact failed transport length or SHA256 verification",
        )
        expected = b"".join(
            gold[call["stress"] * 8192 : (call["stress"] + 1) * 8192] * 2
            for call in calls
        )
        expected_path = directory / specification["golden_file"]
        require(
            expected_path.read_bytes() == expected
            and digest(expected_path) == specification["golden_sha256"],
            "ledger oracle differs from independent normal/stress output grids",
        )
        errors, maximum = 0, 0
        for actual_word, expected_word in zip(
            struct.iter_unpack("<i", actual), struct.iter_unpack("<i", expected)
        ):
            errors += actual_word != expected_word
            maximum = max(maximum, abs(actual_word[0] - expected_word[0]))
        reports.append(
            {
                "name": name,
                "dtype": metadata["dtype"],
                "shape": metadata["shape"],
                "nbytes": len(actual),
                "sha256": metadata["sha256"],
                "golden_sha256": specification["golden_sha256"],
                "checked": len(actual) // 4,
                "errors": errors,
                "max_abs": maximum,
                "status": "PASS" if errors == 0 else "FAIL",
            }
        )
    return {
        "status": "PASS"
        if all(report["status"] == "PASS" for report in reports)
        else "FAIL",
        "comparison": "bit-exact i32 for actual C and snapshot of every call",
        "artifacts_manifest_sha256": digest(artifacts_path),
        "outputs": reports,
    }


def calibration(args):
    manifest = read_json(args.manifest)
    mode, post_add = calibration_mode(manifest, "calibration manifest")
    directory = args.manifest.resolve().parent
    names = [variant["name"] for variant in manifest["variants"]]
    require(
        names in (["ddr", "spm"], ["ddr", "spm", "acc"]),
        "unexpected calibration routes",
    )
    three_routes = names == ["ddr", "spm", "acc"]
    stress_selection, stress_schedule = calibration_schedule(
        manifest, "calibration manifest", three_routes
    )
    selected_stresses = list(dict.fromkeys(stress_schedule))
    reference = None
    if three_routes:
        reference = read_json(directory / "accumulator-manifest.json")
        require(
            calibration_mode(reference, "calibration input/reference manifest")
            == (mode, post_add),
            "calibration mode differs from input/reference manifest",
        )
        require(
            all(
                (key in manifest) == (key in reference)
                for key in ("calls", "stress", "stress_schedule")
            ),
            "calibration and reference must both declare stress selection",
        )
        require(
            calibration_schedule(reference, "calibration input/reference manifest")
            == (stress_selection, stress_schedule),
            "calibration stress selection differs from input/reference manifest",
        )
        require(
            reference.get("orders")
            == [["ddr", "spm", "ddr", "spm", "ddr", "acc", "ddr", "acc"]]
            * len(selected_stresses),
            "calibration reference orders differ from its complete stress selection",
        )
    if stress_selection is not None:
        outputs = manifest.get("ddr_outputs", [])
        require(
            len(outputs) == 1
            and outputs[0].get("name") == "accumulator_calls"
            and outputs[0].get("dtype") == "<i4"
            and outputs[0].get("shape") == [8, 4096]
            and outputs[0].get("nbytes") == 8 * 4096 * 4,
            "split calibration requires one complete eight-call DDR ledger",
        )
    image = args.image or directory / "acc-path-calibration.bin"
    task, uart = verified_task(args.task, image, manifest["image_sha256"])
    golden = directory / "golden.bin"
    require(
        digest(golden) == manifest["golden_sha256"],
        "calibration independent golden changed",
    )
    data = golden.read_bytes()
    require(len(data) == 2 * 32 * 64 * 4, "calibration golden has incomplete grids")
    calls = calibration_calls(
        uart,
        [fnv64(data[:8192]), fnv64(data[8192:])],
        three_routes,
        stress_schedule,
    )
    host_outputs = calibration_outputs(
        manifest, task, uart, calls, directory, args.artifacts
    )
    if host_outputs["status"] == "FAIL":
        write_json(
            args.output,
            {
                "schema": 1,
                "status": "FAIL",
                "mode": mode,
                "post_add": post_add,
                "stress": stress_selection,
                "stress_schedule": stress_schedule,
                "call_count": len(calls),
                "task_id": task["id"],
                "image_sha256": manifest["image_sha256"],
                "host_output_validation": host_outputs,
                "note": "DDR output validation failed; performance comparison is excluded.",
            },
        )
        return 1
    if three_routes:
        require(
            all(
                manifest.get(key)
                for key in ("observable_contract", "acc_adapter", "spm_constraint")
            ),
            "three-route calibration must disclose its source/adapter constraints",
        )
        require(
            len(reference["rounds"]) == len(calls),
            "input/reference identities are incomplete",
        )
        for stress in selected_stresses:
            identities = []
            for call, entry in zip(calls, reference["rounds"]):
                require(
                    entry["route"] == call["route"]
                    and type(entry["stress"]) in (bool, int)
                    and entry["stress"] in (0, 1)
                    and int(entry["stress"]) == call["stress"],
                    "input/reference identities do not match execution order",
                )
                if call["stress"] == stress:
                    identities.append((entry["inputs_sha256"], entry["output_sha256"]))
            require(
                len(set(identities)) == 1,
                "calibration routes use different inputs or outputs",
            )
    sources = []
    for variant in manifest["variants"]:
        name = variant["name"]
        stem = "hot_" + name
        for suffix, key in (
            (".mlir", "source_sha256"),
            (".planned.mlir", "planned_sha256"),
            (".emitted.mlir", "emitted_sha256"),
        ):
            if variant[key] is None:
                require(
                    name == "spm"
                    and three_routes
                    and suffix == ".planned.mlir"
                    and "physical" in manifest["spm_constraint"],
                    "calibration omitted required route evidence",
                )
            else:
                require(
                    digest(directory / (stem + suffix)) == variant[key],
                    "calibration generated route evidence changed",
                )
        sources.append(
            (directory / (stem + ".mlir"))
            .read_text()
            .replace("@" + stem + "(", "@matrix(")
        )
        emitted = (directory / (stem + ".emitted.mlir")).read_text()
        require(
            ("bosc_ame.register_schedule" in emitted) == (name == "acc"),
            "calibration register schedule differs from named route",
        )
        require(
            ("bosc_ame.ddr.frame" in emitted) == (name == "ddr")
            and ("bosc_ame.spm.view" in emitted) == (name == "spm"),
            "calibration placement differs from the named route",
        )
    if not three_routes:
        require(
            sources[0] == sources[1],
            "two-route calibration variants were not compiled from the same source",
        )
    else:
        common_source = normalized_calibration_source(sources[2])
        require(
            normalized_calibration_source(sources[0], observer=True) == common_source
            and normalized_calibration_source(
                sources[1],
                physical="physical" in manifest["spm_constraint"],
                observer=True,
            )
            == common_source,
            "three-route sources differ beyond disclosed physical P and terminal observer",
        )
    summaries, comparisons = [], []
    for stress in selected_stresses:
        values = {
            route: [
                call["cycles"]
                for call in calls
                if call["route"] == route and call["stress"] == stress
            ]
            for route in names
        }
        for route, samples in values.items():
            summaries.append(
                dict(
                    stress=stress,
                    route=route,
                    samples=samples,
                    median=statistics.median(samples),
                    spread=max(samples) - min(samples),
                )
            )
        selected = [call for call in calls if call["stress"] == stress]
        for offset, candidate in (
            ((0, "spm"), (4, "acc")) if three_routes else ((0, "spm"),)
        ):
            block = selected[offset : offset + 4]
            baseline = [block[index]["cycles"] for index in (0, 2)]
            candidate_values = [block[index]["cycles"] for index in (1, 3)]
            saved = statistics.median(baseline) - statistics.median(candidate_values)
            spread = max(
                max(samples) - min(samples) for samples in (baseline, candidate_values)
            )
            comparisons.append(
                dict(
                    stress=stress,
                    comparison="ddr-" + candidate,
                    baseline_samples=baseline,
                    candidate_samples=candidate_values,
                    median_saved_cycles=saved,
                    median_speedup=statistics.median(baseline)
                    / statistics.median(candidate_values),
                    largest_observed_spread=spread,
                    improvement_exceeds_spread=saved > spread,
                )
            )
    write_json(
        args.output,
        dict(
            schema=1,
            status="PASS",
            mode=mode,
            post_add=post_add,
            stress=stress_selection,
            stress_schedule=stress_schedule,
            call_count=len(calls),
            comparison=manifest["comparison"],
            measurement=manifest["measurement"],
            task_id=task["id"],
            image_sha256=manifest["image_sha256"],
            node=task["node"],
            fpga=task["fpga"],
            scheduler_started=task["started"],
            scheduler_finished=task["finished"],
            scheduler_details=task["result"].get("details", {}),
            variants=manifest["variants"],
            calls=calls,
            host_output_validation=host_outputs,
            observable_contract=manifest.get("observable_contract"),
            acc_adapter=manifest.get("acc_adapter"),
            spm_constraint=manifest.get("spm_constraint"),
            input_reference_identity=reference,
            summaries=summaries,
            comparisons=comparisons,
            note="Plan versions and physical source constraints remain explicit; calibration does not establish automatic V2 SPM profitability.",
        ),
    )
    return 0


def artifacts(args):
    """Download immutable outputs from one existing successful scheduler task."""
    from fpga_scheduler import Client

    task = read_json(args.task)
    result = task.get("result")
    require(
        task.get("state") == "completed" and result is not None,
        "artifact download requires a completed task",
    )
    require(
        result.get("execution_status") == "exited"
        and result.get("app_exit_code") == 0
        and result.get("exit_code") == 0,
        "artifact download requires successful application and scheduler exit",
    )
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with Client(
        "http://127.0.0.1:18765",
        token_file=str(Path(__file__).resolve().parents[2] / "client.token"),
        request_timeout=360,
    ) as client:
        manifest = client.artifacts(task["id"])
        require(bool(manifest), "successful task has no saved DDR outputs")
        require(
            manifest == result.get("artifacts"),
            "server output manifest differs from the supplied task snapshot",
        )
        write_json(destination / "artifacts.json", manifest)
        for item in manifest:
            require(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", item["name"]),
                "artifact has an invalid output name",
            )
            path = client.download_artifact(
                task["id"], item["name"], destination / (item["name"] + ".bin")
            )
            require(
                Path(path).stat().st_size == item["nbytes"]
                and digest(path) == item["sha256"],
                "downloaded artifact differs from scheduler manifest",
            )
    write_json(
        destination / "download-evidence.json",
        {
            "schema": 1,
            "scope": "transport integrity only; requires numerical oracle validation",
            "task_id": task["id"],
            "task_sha256": digest(args.task),
            "image_sha256": result["details"]["image_sha256"],
            "artifacts": manifest,
        },
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("diagnostics")
    command.set_defaults(handler=diagnostics)
    command.add_argument("--result", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser(
        "artifacts", help="Download outputs of an existing completed task"
    )
    command.set_defaults(handler=artifacts)
    command.add_argument("--task", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("calibration")
    command.set_defaults(handler=calibration)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--task", type=Path, required=True)
    command.add_argument("--image", type=Path)
    command.add_argument(
        "--artifacts",
        type=Path,
        help="Downloaded artifacts.json; required when the image declares DDR outputs",
    )
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
