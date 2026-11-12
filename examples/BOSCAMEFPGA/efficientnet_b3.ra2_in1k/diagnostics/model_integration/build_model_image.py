#!/usr/bin/env python3
"""Link and audit a genuine-weight EfficientNet-B3 NH/E6 whole-model image.

Uses final optimized convolution objects through the explicit overlay. Never
submits a board job; phase gates and scheduler invocation remain separate.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[2]
BUDDY = REPO / "buddy-mlir"
RUNTIME = BUDDY / "runtime/boscame"
TOOLS = BUDDY / "examples/FPGA-BOSCAME/tools"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from cases import inventory  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, log):
    proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    log.write_text(
        "$ " + repr(list(map(str, cmd))) + chr(10) + proc.stdout + proc.stderr
    )
    if proc.returncode:
        raise RuntimeError(
            "command failed; see " + str(log) + chr(10) + proc.stderr[-2500:]
        )
    return proc.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-dir", type=Path, default=ROOT / "build/model-integration/physical-plan"
    )
    p.add_argument(
        "--runner", type=Path, default=ROOT / "build/model-integration/runner"
    )
    p.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    p.add_argument(
        "--conv-overlay",
        type=Path,
        help="explicit convolution objects; defaults to the current --kernels directory",
    )
    p.add_argument(
        "--connectors",
        type=Path,
        default=ROOT / "build/model-integration/connectors-ddr-policy",
    )
    p.add_argument(
        "--silu-overlay",
        type=Path,
        help="explicit independently validated full-array C/RVV SiLU objects",
    )
    p.add_argument(
        "--silu-source",
        type=Path,
        help="explicit source override for an isolated SiLU overlay; must match its object proof",
    )
    p.add_argument(
        "--dequant-overlay",
        type=Path,
        help="explicit independently validated full-array C/RVV dequant connectors",
    )
    p.add_argument(
        "--gap-overlay",
        type=Path,
        help="explicit independently validated ordered C GAP objects",
    )
    p.add_argument(
        "--gap-source",
        type=Path,
        help="explicit source override for an isolated GAP overlay; must match its object proof",
    )
    p.add_argument(
        "--gap-reference",
        action="store_true",
        help="bounded probe only: lower the original ordered BOSCAME GAP reference",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "build/model-integration/model-image"
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--prefix-probe", action="store_true")
    p.add_argument(
        "--bounded-probe",
        action="store_true",
        help="explicit small plan without prefix snapshots",
    )
    p.add_argument(
        "--strict-probe",
        action="store_true",
        help="2e-6 absolute/relative board numeric checks",
    )
    p.add_argument(
        "--host-numerical-validation",
        action="store_true",
        help="export completed outputs even when inline comparison fails; require independent host audit",
    )
    p.add_argument("--llvm-bin", type=Path, default=BUDDY / "llvm/build/bin")
    p.add_argument("--buddy-opt", type=Path, default=BUDDY / "build/bin/buddy-opt")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.conv_overlay is None:
        a.conv_overlay = a.kernels
    if a.gap_reference and (not a.bounded_probe or a.gap_overlay is not None):
        raise ValueError(
            "--gap-reference requires --bounded-probe and no --gap-overlay"
        )
    if a.silu_source is not None and a.silu_overlay is None:
        raise ValueError("--silu-source requires an explicit --silu-overlay")
    if a.gap_source is not None and a.gap_overlay is None:
        raise ValueError("--gap-source requires an explicit --gap-overlay")
    objdir = a.output / "obj"
    objdir.mkdir(exist_ok=True)
    plan = json.loads((a.plan_dir / "physical-plan.json").read_text())
    params = json.loads((a.plan_dir / "parameters.json").read_text())
    runner_manifest_path = a.runner / "runner-manifest.json"
    runner_manifest = json.loads(runner_manifest_path.read_text())
    if (
        runner_manifest.get("status") != "PASS"
        or runner_manifest.get("plan_sha256") != sha(a.plan_dir / "physical-plan.json")
        or runner_manifest.get("physical_calls") != len(plan["calls"])
    ):
        raise ValueError(
            "runner manifest does not match the physical plan; regenerate runner"
        )
    for filename in ("graph_runner.c", "graph_runner.h"):
        if sha(a.runner / filename) != runner_manifest["artifacts"].get(filename):
            raise ValueError(
                "runner source/header changed; regenerate runner " + filename
            )
    runner_snapshot = (a.output / "runner-inputs").resolve()
    runner_snapshot.mkdir(exist_ok=True)
    for filename in ("runner-manifest.json", "graph_runner.c", "graph_runner.h"):
        shutil.copy2(a.runner / filename, runner_snapshot / filename)
    conbuild = json.loads((a.connectors / "build-summary.json").read_text())
    if (
        conbuild["status"] != "PASS"
        or conbuild["expected"] != 84
        or conbuild["completed"] != 84
    ):
        raise ValueError("all 84 connector objects must compile before model linking")
    wbytes = plan["workspace"]["provisional_total_bytes"]
    if plan["workspace"]["provisional"]:
        raise ValueError("provisional workspace plan")
    if sha(a.plan_dir / "parameters.bin") != params["sha256"]:
        raise ValueError("parameter blob hash mismatch")
    clang = a.llvm_bin / "clang"
    ld = a.llvm_bin / "ld.lld"
    objcopy = a.llvm_bin / "llvm-objcopy"
    if not ld.is_file():
        ld = Path(shutil.which("ld.lld") or "/missing-local-ld.lld")
    opt = a.buddy_opt
    translate = BUDDY / "build/bin/buddy-translate"
    llc = a.llvm_bin / "llc"
    objdump = a.llvm_bin / "llvm-objdump"
    nm = a.llvm_bin / "llvm-nm"
    readelf = a.llvm_bin / "llvm-readelf"
    for t in (clang, ld, objcopy, objdump, nm, readelf):
        if not t.is_file():
            raise ValueError("missing local tool " + str(t))
    flags = [
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdc_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mno-relax",
        "-O2",
        "-nostdlib",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-pie",
        "-fno-stack-protector",
        "-ffunction-sections",
        "-fdata-sections",
        "-ffp-contract=off",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-DUART_FREQ_HZ=5529600",
        "-DBOSC_AME_DMA_ASSERT(x)=",
        "-DVIT_WORKSPACE_BYTES=%du" % wbytes,
    ]
    if a.prefix_probe:
        flags += ["-DMODEL_PREFIX_PROBE=1"]
    if a.strict_probe:
        flags += ["-DMODEL_COMPARE_ATOL=2e-6f", "-DMODEL_COMPARE_RTOL=2e-6f"]
    if a.host_numerical_validation:
        flags += ["-DMODEL_HOST_NUMERICAL_VALIDATION=1"]
    for include in (
        ROOT,
        HERE,
        a.output,
        runner_snapshot,
        RUNTIME / "runtime",
        RUNTIME / "uart",
        RUNTIME / "nh_e6",
        RUNTIME / "nr",
    ):
        flags += ["-I", str(include)]
    # Immutable message table is safe to share between NH and E6; progress
    # handshake never exposes a mutable DDR string pointer across the cores.
    messages = []
    for c in plan["calls"]:
        for phase in ("BEGIN", "END"):
            messages.append(
                "[efficientnet-op] %d/%d %s node=%s case=%s\r\n"
                % (
                    c["index"] + 1,
                    len(plan["calls"]),
                    phase,
                    c["logical_node"],
                    c["target"],
                )
            )
    table = [
        "#define MODEL_CYCLE_SHAPE " + json.dumps("[" + str(len(plan["calls"])) + "]"),
        "static const char *const efficientnet_trace_messages[] = {",
    ]
    table += [
        "  "
        + json.dumps(m.replace(chr(92) + "r", chr(13)).replace(chr(92) + "n", chr(10)))
        + ","
        for m in messages
    ]
    table += ["};", ""]
    (a.output / "model_trace_messages.h").write_text(chr(10).join(table))
    assembly = []
    for section, symbol, blob in (
        (".model_params", "model_params_blob", a.plan_dir / "parameters.bin"),
        (".model_input", "model_input_blob", a.plan_dir / "input.f32"),
        (".model_reference", "model_reference_blob", a.plan_dir / "w8a8-logits.f32"),
    ):
        assembly += [
            ".section " + section + ',"a",@progbits',
            ".balign 64",
            ".globl " + symbol,
            symbol + ":",
            ".incbin " + json.dumps(str(blob.resolve())),
        ]
    (a.output / "model_blobs.S").write_text(chr(10).join(assembly) + chr(10))
    tasks = []
    objects = []
    inputs = []
    cpu_tasks = []

    def add(source, name, extra=()):
        output = objdir / (name + ".o")
        objects.append(output)
        tasks.append((source, output, list(extra)))
        return output

    # Runtime is linked first because boot trap stubs use JAL range.
    add(RUNTIME / "nh_e6/nh_e6_boot.S", "boot")
    add(
        RUNTIME / "nh_e6/nh_e6_platform.c",
        "platform",
        [
            "-Dinit_uart=efficientnet_init_uart",
            "-DNH_E6_WAIT_ITERATIONS=0xffffffffu",
            "-DNH_E6_MMIO_READ32=efficientnet_mmio_read32",
            "-include",
            str(HERE / "model_board_support.h"),
        ],
    )
    add(RUNTIME / "uart/uart.c", "uart")
    add(
        RUNTIME / "runtime/bosc_ame_dma.c",
        "dma",
        ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=1"],
    )
    add(RUNTIME / "nr/nr_math.c", "math")
    add(HERE / "model_board_support.c", "model_support")
    add(HERE / "model_launch.c", "model_launch")
    add(HERE / "model_trap.S", "model_trap")
    add(runner_snapshot / "graph_runner.c", "graph_runner")
    add(a.output / "model_blobs.S", "model_blobs")
    cases = {c["name"]: c for c in inventory()}
    used = sorted(
        {c["target"] for c in plan["calls"] if c["target_kind"] == "existing_case"}
    )
    optimized = []
    for name in used:
        case = cases[name]
        isconv = case["family"] in ("conv2d_f32", "dwconv2d_f32")
        gap_reference = case["family"] == "gap" and a.gap_reference
        native_overlay = (
            a.silu_overlay
            if case["family"] == "silu"
            else a.gap_overlay
            if case["family"] == "gap" and not gap_reference
            else None
        )
        native_case = (
            case["family"] in ("silu", "gap")
            and not gap_reference
            and (
                native_overlay is not None
                or case.get("implementation") in ("hybrid_rvv_c", "hybrid_ordered_c")
            )
        )
        directory = (native_overlay or (a.conv_overlay if isconv else a.kernels)) / name
        if isconv:
            optimized.append(name)
        for file in ("kernel.o", "adapter.c"):
            if not (directory / file).is_file():
                raise ValueError("missing exact artifact " + str(directory / file))
        if native_case:
            proof_path = directory / "native-implementation.json"
            if proof_path.exists():
                proof = json.loads(proof_path.read_text())
                valid = proof["object_sha256"] == sha(directory / "kernel.o") and proof[
                    "adapter_sha256"
                ] == sha(directory / "adapter.c")
            elif (
                native_overlay is not None
                and (directory / "build-result.json").exists()
            ):
                proof = json.loads((directory / "build-result.json").read_text())
                valid = (
                    proof.get("status") == "PASS"
                    and proof.get("kernel_sha256") == sha(directory / "kernel.o")
                    and proof.get("adapter_sha256") == sha(directory / "adapter.c")
                )
            else:
                valid = False
            source_file = ROOT / case.get(
                "implementation_source",
                "diagnostics/model_integration/"
                + (
                    "gap_phase_candidate.c"
                    if case["family"] == "gap"
                    else "silu_rvv_direct_output_candidate.c"
                ),
            )
            if case["family"] == "silu" and a.silu_source is not None:
                source_file = a.silu_source
            if case["family"] == "gap" and a.gap_source is not None:
                source_file = a.gap_source
            source_hash = (
                proof.get("candidate_source_sha256", proof.get("source_sha256"))
                if valid
                else None
            )
            valid = valid and source_hash == sha(source_file)
            if not valid:
                raise ValueError(
                    "native case has missing/stale object provenance; rebuild " + name
                )
        if isconv:
            source = (directory / "adapter.c").read_text()
            if "__im2col_quantize(&u0, &uqp, &ups" not in source:
                raise ValueError(
                    "optimized convolution adapter does not bind tiled im2col " + name
                )
        adapter_path = directory / "adapter.c"
        if gap_reference:
            from export import adapter_source

            reference_case = dict(case)
            reference_case.pop("implementation", None)
            reference_case.pop("implementation_symbol", None)
            adapter_path = a.output / ("reference-adapter-" + name + ".c")
            adapter_path.write_text(adapter_source(reference_case))
        if case["family"] in ("silu", "gap", "add") and not native_case:
            # Existing CPU-only cases do not automatically request DDR phase
            # lowering, but model workspace re-use can follow an AME producer.
            # Keep production sources/objects intact and lower a model-local
            # copy with the existing per-access policy used by mixed kernels.
            cpu_dir = a.output / "cpu-kernels" / name
            cpu_dir.mkdir(parents=True, exist_ok=True)
            source = (directory / "kernel.boscame.mlir").read_text()
            if "bosc_ame.cpu_ddr_policy" not in source.split("func.func", 1)[0]:
                source = source.replace(
                    "module attributes {",
                    "module attributes {bosc_ame.cpu_ddr_policy, ",
                    1,
                )
            (cpu_dir / "kernel.cpu-ddr-policy.mlir").write_text(source)
            cpu_tasks.append((name, cpu_dir))
            objects.append(cpu_dir / "kernel.o")
        else:
            objects.append(directory / "kernel.o")
        add(adapter_path, "adapter_" + name)
        inputs += [
            {
                "case": name,
                "family": case["family"],
                "overlay": isconv or native_overlay is not None,
                "implementation": (
                    "hybrid_ordered_c" if case["family"] == "gap" else "hybrid_rvv_c"
                )
                if native_case
                else "boscame",
                "object": str(directory / "kernel.o"),
                "object_sha256": sha(directory / "kernel.o"),
                "adapter_source": str(adapter_path),
                "adapter_source_sha256": sha(adapter_path),
            }
        ]
    if not (a.prefix_probe or a.bounded_probe) and len(optimized) != 18:
        raise ValueError("expected all 18 optimized convolution cases")
    if a.prefix_probe and plan.get("mode") != "real_weight_nonzero_prefix_probe":
        raise ValueError("--prefix-probe requires an explicitly prepared prefix plan")
    if a.bounded_probe and (
        plan.get("mode") not in ("bounded_silu_probe", "bounded_connector_probe")
        or plan.get("full_model") is not False
    ):
        raise ValueError(
            "--bounded-probe requires an explicitly prepared bounded_silu_probe/bounded_connector_probe plan"
        )
    connector_inputs = []
    for spec in plan["connectors"]:
        overlay = (
            a.silu_overlay
            if spec["kernel"] == "silu"
            else a.dequant_overlay
            if spec["kernel"] == "dequantize_unpack_bias"
            else None
        )
        directory = (overlay or a.connectors) / spec["name"]
        # Fail closed: pure scalar DDR bridges need the explicit existing
        # backend policy, not merely phase fences surrounding the whole call.
        if overlay:
            policy = (directory / "kernel.nr.s").read_text()
            if policy.count("csrw") < 2:
                raise ValueError(
                    "native overlay lacks actual DDR policy " + spec["name"]
                )
            validation = directory / "build-result.json"
            if validation.exists():
                result = json.loads(validation.read_text())
                valid = result.get("status") == "PASS"
            else:
                result = json.loads((directory / "candidate.json").read_text())
                valid = (
                    result.get("bit_exact_host")
                    and result.get("guard_pass_host")
                    and result.get("isa_restriction_pass")
                )
            valid = (
                valid
                and result.get("kernel_sha256", result.get("object_sha256"))
                == sha(directory / "kernel.o")
                and result.get("adapter_sha256") == sha(directory / "adapter.c")
                and result.get("source_sha256")
                == sha(
                    (a.silu_source or HERE / "silu_rvv_direct_output_candidate.c")
                    if spec["kernel"] == "silu"
                    else HERE / "dequant_rvv_candidate.c"
                )
            )
            if not valid:
                raise ValueError("native overlay lacks validation " + spec["name"])
        else:
            policy = (directory / "kernel.cpu-ddr-policy.mlir").read_text()
            if "bosc_ame.cpu_ddr_policy" not in policy.split("func.func", 1)[0]:
                raise ValueError("missing explicit scalar DDR policy " + spec["name"])
            lowered = (directory / "kernel.llvm.mlir").read_text()
            if "csrw 0xbc0" not in lowered and "bosc_ame_ddr_" not in lowered:
                raise ValueError("DDR policy did not survive lowering " + spec["name"])
        objects.append(directory / "kernel.o")
        add(directory / "adapter.c", "connector_" + spec["name"])
        connector_inputs.append(
            {
                "name": spec["name"],
                "object": str(directory / "kernel.o"),
                "object_sha256": sha(directory / "kernel.o"),
                "adapter_sha256": sha(directory / "adapter.c"),
                "implementation": "hybrid_rvv_c" if overlay else "boscame",
                "overlay": str(overlay) if overlay else None,
            }
        )

    def compile_one(task):
        source, output, extra = task
        run(
            [clang, *flags, *extra, "-c", source, "-o", output],
            a.output / ("compile-" + output.stem + ".log"),
        )
        return output

    def compile_cpu(task):
        name, directory = task
        run(
            [
                opt,
                directory / "kernel.cpu-ddr-policy.mlir",
                "--lower-bosc-ame=scalar-f32-ordered-reductions=true scalar-f32-max-reductions=true",
                "-o",
                directory / "kernel.llvm.mlir",
            ],
            directory / "lower.log",
        )
        run(
            [
                translate,
                "--buddy-to-llvmir",
                directory / "kernel.llvm.mlir",
                "-o",
                directory / "kernel.ll",
            ],
            directory / "translate.log",
        )
        run(
            [
                llc,
                directory / "kernel.ll",
                "-O2",
                "-filetype=asm",
                "-mtriple=riscv64",
                "-target-abi=lp64d",
                "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
                "-code-model=medium",
                "-o",
                directory / "kernel.s",
            ],
            directory / "llc.log",
        )
        encoded = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "ame_to_word.py"),
                "--gpr-mode=fixed",
                "--preserve-matrix-registers",
            ],
            input=(directory / "kernel.s").read_text(),
            capture_output=True,
            text=True,
        )
        (directory / "encode.log").write_text(encoded.stderr)
        if encoded.returncode:
            raise RuntimeError(encoded.stderr)
        (directory / "kernel.encoded.s").write_text(encoded.stdout)
        restricted = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "restrict_fpga_assembly.py"),
                "--expand-f32-sub",
                "--expand-vector-merges",
                "--order-vector-copies",
                "--reject-vector-ordered-reductions",
                "--reject-vector-max-reductions",
            ],
            input=encoded.stdout,
            capture_output=True,
            text=True,
        )
        (directory / "restrict.log").write_text(restricted.stderr)
        if restricted.returncode:
            raise RuntimeError(restricted.stderr)
        (directory / "kernel.nr.s").write_text(restricted.stdout)
        run(
            [
                clang,
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdcv_zicsr_zifencei",
                "-mabi=lp64d",
                "-mno-relax",
                "-c",
                directory / "kernel.nr.s",
                "-o",
                directory / "kernel.o",
            ],
            directory / "object.log",
        )
        return {
            "name": name,
            "kernel_sha256": sha(directory / "kernel.o"),
            "policy_source_sha256": sha(directory / "kernel.cpu-ddr-policy.mlir"),
        }

    start = time.monotonic()
    cpu_results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        fs = [pool.submit(compile_cpu, t) for t in cpu_tasks]
        for f in as_completed(fs):
            cpu_results.append(f.result())
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        fs = [pool.submit(compile_one, t) for t in tasks]
        for f in as_completed(fs):
            f.result()
    cpu_by_name = {item["name"]: item for item in cpu_results}
    for item in inputs:
        if item["case"] in cpu_by_name:
            # The policy-lowered copy, rather than the reference object, is
            # actually linked. Record both so the image audit follows it.
            item["reference_object"] = item["object"]
            item["reference_object_sha256"] = item["object_sha256"]
            item["object"] = str(a.output / "cpu-kernels" / item["case"] / "kernel.o")
            item["object_sha256"] = cpu_by_name[item["case"]]["kernel_sha256"]
            item["implementation"] = "boscame_model_cpu_ddr_policy"
    for kind, entries in (("existing", inputs), ("connector", connector_inputs)):
        for item in entries:
            name = item.get("case", item.get("name"))
            directory = Path(item["object"]).parent
            frozen = (a.output / "input-snapshots" / kind / name).resolve()
            frozen.mkdir(parents=True, exist_ok=True)
            adapter = Path(item.get("adapter_source", str(directory / "adapter.c")))
            expected_adapter = item.get(
                "adapter_source_sha256", item.get("adapter_sha256")
            )
            if (
                sha(directory / "kernel.o") != item["object_sha256"]
                or sha(adapter) != expected_adapter
            ):
                raise ValueError("input changed while building image " + name)
            shutil.copy2(directory / "kernel.o", frozen / "kernel.o")
            shutil.copy2(adapter, frozen / "adapter.c")
            for filename in (
                "kernel.nr.s",
                "kernel.boscame.mlir",
                "kernel.cpu-ddr-policy.mlir",
                "kernel.llvm.mlir",
            ):
                if (directory / filename).exists():
                    shutil.copy2(directory / filename, frozen / filename)
            item["object_snapshot"] = str(frozen / "kernel.o")
            item["adapter_snapshot"] = str(frozen / "adapter.c")
            item["adapter_object"] = str(
                objdir
                / (("adapter_" if kind == "existing" else "connector_") + name + ".o")
            )
            item["adapter_object_sha256"] = sha(Path(item["adapter_object"]))
    elf = a.output / "efficientnet_b3_model.elf"
    binary = a.output / "efficientnet_b3_model.bin"
    run(
        [
            ld,
            "-m",
            "elf64lriscv",
            "--gc-sections",
            "-T",
            HERE / "model_link.ld",
            "-Map=" + str(a.output / "efficientnet_b3_model.map"),
            "-o",
            elf,
            *objects,
        ],
        a.output / "link.log",
    )
    run(
        [
            sys.executable,
            TOOLS / "check_nr_elf.py",
            elf,
            "--objdump",
            objdump,
            "--output",
            a.output / "elf-audit.json",
        ],
        a.output / "audit.log",
    )
    undefined = run([nm, "-u", elf], a.output / "undefined.log").strip()
    if undefined:
        raise ValueError("unresolved symbols " + undefined)
    run([objcopy, "-O", "binary", elf, binary], a.output / "objcopy.log")
    run([readelf, "-S", "-l", elf], a.output / "elf-layout.log")
    symbols = run([nm, "-n", elf], a.output / "symbols.log")
    wanted = {
        "__nh_e6_ddr_window_start",
        "__nh_e6_ddr_window_end",
        "__model_workspace_start",
        "__model_workspace_end",
        "__bss_start",
        "__bss_end",
        "__e6_stack_bottom",
        "__e6_stack_top",
        "__nh_stack_bottom",
        "__nh_stack_top",
        "model_params_blob",
        "model_input_blob",
        "model_reference_blob",
        "_end",
    }
    wanted.update(
        {
            "__model_export_start",
            "__model_export_end",
            "model_export_logits",
            "model_export_cycles",
        }
    )
    layout = {}
    for line in symbols.splitlines():
        f = line.split()
        if len(f) == 3 and f[2] in wanted:
            layout[f[2]] = "0x" + f[0]
    report = {
        "status": "PASS",
        "scope": (
            "bounded operator probe image link/audit; not whole-model evidence"
            if a.bounded_probe
            else "real-weight prefix probe image link/audit; not whole-model evidence"
            if a.prefix_probe
            else "optimized-18 genuine-weight whole-model image link/audit; no FPGA evidence"
        ),
        "full_model": not (a.prefix_probe or a.bounded_probe),
        "board_submitted": False,
        "physical_calls": len(plan["calls"]),
        "existing_cases": len(used),
        "optimized_conv_cases": optimized,
        "connector_specializations": len(plan["connectors"]),
        "compiled_objects": len(objects),
        "workspace_bytes": wbytes,
        "parameter_bytes": params["bytes"],
        "parameter_sha256": params["sha256"],
        "plan_sha256": sha(a.plan_dir / "physical-plan.json"),
        "connector_build_summary_sha256": sha(a.connectors / "build-summary.json"),
        "tools": {
            name: {"path": str(tool), "sha256": sha(tool)}
            for name, tool in (
                ("buddy_opt", opt),
                ("buddy_translate", translate),
                ("llc", llc),
                ("clang", clang),
                ("linker", ld),
                ("objcopy", objcopy),
                ("objdump", objdump),
            )
        },
        "image_bytes": binary.stat().st_size,
        "image_sha256": sha(binary),
        "elf_sha256": sha(elf),
        "elapsed_seconds": time.monotonic() - start,
        "elf_layout_symbols": layout,
        "existing_case_inputs": inputs,
        "connector_inputs": connector_inputs,
        "runner_inputs": {
            "manifest": str(runner_snapshot / "runner-manifest.json"),
            "manifest_sha256": sha(runner_snapshot / "runner-manifest.json"),
            "plan_sha256": runner_manifest["plan_sha256"],
            "source": str(runner_snapshot / "graph_runner.c"),
            "source_sha256": sha(runner_snapshot / "graph_runner.c"),
            "header": str(runner_snapshot / "graph_runner.h"),
            "header_sha256": sha(runner_snapshot / "graph_runner.h"),
            "original_manifest": str(runner_manifest_path),
            "linked_object": str(objdir / "graph_runner.o"),
            "linked_object_sha256": sha(objdir / "graph_runner.o"),
        },
        "model_local_cpu_policy_kernels": sorted(cpu_results, key=lambda r: r["name"]),
        "build_sources": {
            str(f.relative_to(REPO)): sha(f)
            for f in (
                Path(__file__),
                HERE / "model_board_support.c",
                HERE / "model_board_support.h",
                HERE / "model_launch.c",
                HERE / "model_link.ld",
                HERE / "model_trap.S",
            )
        },
        "diagnostic_validation_threshold": {
            "reference": "W8A8 CPU semantics",
            "atol": 2e-6 if a.strict_probe else 0.05,
            "rtol": 2e-6 if a.strict_probe else 0.01,
            "meaning": "declared diagnostic threshold, not measured acceptance or ImageNet accuracy",
        },
        "numerical_validation_location": "host audit of all downloaded outputs; process exit zero means completed execution"
        if a.host_numerical_validation
        else "inline comparison and independent host comparison",
        "reference_logits_sha256": sha(a.plan_dir / "w8a8-logits.f32"),
        "reference_provenance": json.loads(
            (a.plan_dir / "reference-provenance.json").read_text()
        )
        if (a.plan_dir / "reference-provenance.json").exists()
        else {"kind": "original Torch W8A8 semantic CPU reference"},
        "silu_source_override": {
            "path": str(a.silu_source),
            "sha256": sha(a.silu_source),
        }
        if a.silu_source
        else None,
        "gap_source_override": {"path": str(a.gap_source), "sha256": sha(a.gap_source)}
        if a.gap_source
        else None,
        "output_artifacts": ["logits_f32", "call_cycles_u64"],
        "progress": "503 immutable BEGIN/END message pairs with mailbox ACK; per-call cycles and progressive output FNV64 hashes",
        "limitations": [
            "Image has not executed on FPGA.",
            "Connectors have only small-shape actualIR numeric evidence.",
            "Host must download all logits and compare against real-weight W8A8 golden after execution.",
            "Whole-model board submission waits for parent phase 1/2 completion.",
        ],
    }
    (a.output / "image-manifest.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "physical_calls",
                    "existing_cases",
                    "connector_specializations",
                    "workspace_bytes",
                    "image_bytes",
                    "image_sha256",
                    "elf_layout_symbols",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
