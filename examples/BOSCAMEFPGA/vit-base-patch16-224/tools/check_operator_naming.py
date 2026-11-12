#!/usr/bin/env python3
"""Audit the live primitive inventory without compiling or executing kernels.

Historical rename/source/IR equivalence reports retain their own source tool and
name maps. This audit checks the current names, source structure, typed export,
build inventory, dependencies, deleted cases and source snapshot independently.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
EVIDENCE_ROOT = ROOT / "build/attention-primitive-model-20261029"
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402

EXPECTED_FAMILIES = {
    "add": 3,
    "bias_add": 5,
    "dequantization": 6,
    "gelu": 1,
    "gemm_i8": 5,
    "head_layout": 2,
    "layernorm": 2,
    "matmul_i8": 7,
    "patch_embed": 1,
    "per_token_quantization": 8,
    "quantize_transpose": 2,
    "softmax": 1,
}
RETIRED = tuple(
    "linear_" + shape
    for shape in (
        "196x768x768",
        "197x768x768",
        "197x3072x768",
        "197x768x3072",
        "1x1000x768",
    )
) + ("attention_" + "qk_12x197x197x64", "attention_" + "pv_12x197x64x197")
SOURCE_NAMES = {"kernel.py", "launch.c", "makefile", "board.c", "link.ld"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def evidence(path):
    return {"path": str(path), "sha256": sha(path)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + "." + node.attr
    return ""


def integer_structure(case, tree):
    """Trace pointer aliases to distinguish old-C reads; this is not execution."""
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == case["kernel"]
    )
    dependencies = {arg.arg: {arg.arg} for arg in fn.args.args}

    def roots(node):
        return set().union(
            *(
                dependencies.get(item.id, {item.id})
                for item in ast.walk(node)
                if isinstance(item, ast.Name)
            )
        )

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    dependencies[target.id] = roots(node.value)
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
    old_c_reads = sum(
        call_name(node.func) == "tl.load" and "C" in roots(node.args[0])
        for node in calls
    )
    output_stores = sum(
        call_name(node.func) == "tl.store" and "C" in roots(node.args[0])
        for node in calls
    )
    dots = sum(call_name(node.func) == "tl.dot" for node in calls)
    zero_i32_init = any(
        call_name(node.func) == "tl.full"
        and len(node.args) >= 3
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 0
        and call_name(node.args[2]) == "tl.int32"
        for node in calls
    )
    accumulate = case["family"] == "gemm_i8"
    passed = (
        dots > 0
        and output_stores > 0
        and zero_i32_init
        and (old_c_reads > 0 if accumulate else old_c_reads == 0)
        and ([a["dtype"] for a in case["arguments"]] == ["i8", "i8", "i32"])
        and (
            accumulate
            or (
                case.get("accumulation") == "overwrite"
                and "model_pure_product" in case["roles"]
            )
        )
    )
    return {
        "name": case["name"],
        "status": "PASS" if passed else "FAIL",
        "contract": "C += A @ B.T" if accumulate else "C = A @ B.T",
        "layout": "A[M,K], physical B[N,K], C[M,N]",
        "dot_calls": dots,
        "old_C_load_sites": old_c_reads,
        "C_store_sites": output_stores,
        "zero_i32_accumulator": zero_i32_init,
        "evidence_scope": "static source/descriptor structure; no numerical execution",
    }


def live_files():
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if name not in ("build", "__pycache__", ".git", "assets")
        ]
        for name in names:
            yield Path(directory) / name


def negative_reference(source, line):
    """Recognize explicit retirement registries and rejection test fixtures."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not (getattr(node, "lineno", 0) <= line <= getattr(node, "end_lineno", 0)):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("RETIRED", "REMOVED", "OBSOLETE")
            for t in node.targets
        ):
            return "explicit retired-operator registry used by deletion audit"
        if isinstance(node, ast.For):
            body = "\n".join(ast.unparse(child) for child in node.body)
            if "must_reject(" in body or "invalid.append(" in body:
                return "explicit invalid/rejection test fixture"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, default=EVIDENCE_ROOT / "inventory-naming"
    )
    parser.add_argument("--snapshot-root", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args()
    output, snapshot = args.report_dir.resolve(), args.snapshot_root.resolve()
    if output == (ROOT / "build/operator-naming-20261029").resolve():
        parser.error(
            "historical rename evidence is immutable; choose a new report directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    cases = inventory()
    current = {case["name"]: case for case in cases}
    exporter = load_module("vit_naming_export", ROOT / "tools/export.py")
    board = load_module("vit_naming_board", ROOT / "tools/board_config.py")
    counts = dict(sorted(Counter(case["family"] for case in cases).items()))
    descriptors, semantics, dangling = [], [], []
    prefixes = {
        "per_token_quantization": ("quantize_",),
        "dequantization": ("dequantize_",),
        "head_layout": ("pack_heads_", "unpack_heads_"),
    }
    for case in cases:
        name = case["name"]
        tree = ast.parse(Path(case["kernel_source"]).read_text())
        fn = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == case["kernel"]
            ),
            None,
        )
        shapes = exporter.argument_shapes(case)
        adapter = exporter.adapter_source(case)
        deps = list(board.dependencies(case))
        declared_calls = sorted(
            set(re.findall(r"_mlir_ciface_kernel_(\w+)\(", adapter)) - {name}
        )
        dangling += [
            {"case": name, "dependency": dep} for dep in deps if dep not in current
        ]
        checks = {
            "family_name": name.startswith(
                prefixes.get(case["family"], (case["family"] + "_",))
            ),
            "shape_name": name.endswith("x".join(map(str, case["shape"])))
            or case["family"] in ("patch_embed", "head_layout"),
            "source_entry": fn is not None
            and len(fn.args.args) == len(case["arguments"]) + len(case["constexprs"]),
            "literal_specialization": fn is not None
            and set(case["constexprs"])
            == {a.arg for a in fn.args.args[len(case["arguments"]) :]},
            "required_sources": all(
                (ROOT / name / file).is_file()
                for file in ("kernel.py", "launch.c", "makefile")
            ),
            "makefile_common": "include ../common.mk"
            in (ROOT / name / "makefile").read_text(),
            "export_shapes": len(shapes) == len(case["arguments"])
            and all(
                len(s) == a["rank"] and all(v > 0 for v in s)
                for s, a in zip(shapes, case["arguments"])
            ),
            "export_symbol": case["symbol"] == "triton_" + name
            and "_mlir_ciface_" + case["symbol"] + "(" in adapter,
            "adapter_dependencies": declared_calls == sorted(deps),
            "board_dependencies_closed": all(dep in current for dep in deps),
        }
        descriptors.append(
            {
                "name": name,
                "family": case["family"],
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "argument_shapes": shapes,
                "dependencies": deps,
                "adapter_source_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
                "kernel_source": evidence(Path(case["kernel_source"])),
            }
        )
        if case["family"] in ("gemm_i8", "matmul_i8"):
            semantics.append(integer_structure(case, tree))
    # This added target expands ALL_CASES only; no compilation target is invoked.
    cmd = [
        "make",
        "--no-print-directory",
        "-s",
        "-f",
        str(ROOT / "makefile"),
        "--eval=vit-audit-inventory:;@printf '%s\\n' $(sort $(ALL_CASES))",
        "vit-audit-inventory",
    ]
    made = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    build_names = made.stdout.split()
    build_ok = (
        made.returncode == 0
        and sorted(current) == sorted(build_names)
        and len(build_names) == 43
    )
    (output / "make-inventory.log").write_text(made.stdout + made.stderr)
    links = [p for p in live_files() if p.is_symlink()]
    for directory, dirs, _ in os.walk(ROOT, followlinks=False):
        dirs[:] = [
            n for n in dirs if n not in ("build", "__pycache__", ".git", "assets")
        ]
        links += [
            Path(directory) / n for n in dirs if (Path(directory) / n).is_symlink()
        ]
    broken_links = [str(p) for p in links if not p.exists()]
    case_links = [str(p) for p in links if p.relative_to(ROOT).parts[0] in current]
    deleted = [
        {
            "name": name,
            "directory_absent": not (ROOT / name).exists(),
            "not_in_inventory": name not in current,
        }
        for name in RETIRED
    ]
    snapshot_manifest = snapshot / "before-change-sha256.json"
    hashes = json.loads(snapshot_manifest.read_text())
    snapshot_archive = snapshot / "before-change.tar.gz"
    sources, archive_mismatches = [], []
    with tarfile.open(snapshot_archive) as archive:
        for case in cases:
            for path in sorted((ROOT / case["name"]).iterdir()):
                if path.name not in SOURCE_NAMES:
                    continue
                key = path.relative_to(REPO).as_posix()
                expected, actual = hashes.get(key), sha(path)
                try:
                    archived = archive.extractfile(key).read()
                    archived_hash = hashlib.sha256(archived).hexdigest()
                except (KeyError, AttributeError):
                    archived_hash = None
                if not expected or archived_hash != expected:
                    archive_mismatches.append(
                        {"path": key, "manifest": expected, "archive": archived_hash}
                    )
                sources.append(
                    {
                        "path": key,
                        "expected_sha256": expected,
                        "actual_sha256": actual,
                        "status": "PASS" if expected and actual == expected else "FAIL",
                    }
                )
    metadata = ROOT / "OPERATOR_NAMING_MAP.json"
    mapping_unchanged = sha(metadata) == hashes.get(
        metadata.relative_to(REPO).as_posix()
    )
    old_metadata = json.loads(metadata.read_text())
    mapping_snapshot = ROOT / old_metadata["historical_source_snapshot"]
    mapping_history_ok = (
        sha(mapping_snapshot) == old_metadata["historical_source_snapshot_sha256"]
    )
    references, historical_refs, negative_refs = [], [], []
    tokens = RETIRED + ("matmul_i8_" + "overwrite",)
    paths = list(live_files()) + list(
        (REPO / "test/Integration/BOSCAMEFPGA").glob("vit-*.py")
    )
    for path in paths:
        if (
            path == Path(__file__).resolve()
            or path == metadata
            or path.is_symlink()
            or path.suffix == ".pyc"
        ):
            continue
        if (
            path.suffix not in (".py", ".c", ".h", ".mk", ".json", ".md")
            and path.name != "makefile"
        ):
            continue
        source = path.read_text()
        for line, text in enumerate(source.splitlines(), 1):
            found = [token for token in tokens if token in text]
            if not found:
                continue
            entry = {"path": str(path), "line": line, "text": text, "names": found}
            if path.suffix == ".md":
                historical_refs.append(entry)
            else:
                reason = (
                    negative_reference(source, line) if path.suffix == ".py" else None
                )
                if reason:
                    entry["reason"] = reason
                    negative_refs.append(entry)
                else:
                    references.append(entry)
    inventory_ok = (
        len(cases) == 43 and counts == EXPECTED_FAMILIES and not dangling and build_ok
    )
    source_ok = (
        all(row["status"] == "PASS" for row in sources) and not archive_mismatches
    )
    deletion_ok = all(
        row["directory_absent"] and row["not_in_inventory"] for row in deleted
    )
    passed = (
        inventory_ok
        and source_ok
        and deletion_ok
        and all(row["status"] == "PASS" for row in descriptors + semantics)
        and not references
        and not broken_links
        and not case_links
        and mapping_unchanged
        and mapping_history_ok
    )
    report = {
        "schema": 2,
        "status": "PASS" if passed else "FAIL",
        "scope": "current primitive names, static contracts, typed export, source preservation, inventory and dependency closure",
        "operator_count": len(cases),
        "families": counts,
        "expected_families": EXPECTED_FAMILIES,
        "inventory_status": "PASS" if inventory_ok else "FAIL",
        "cases": descriptors,
        "integer_semantics": semantics,
        "dangling_dependencies": dangling,
        "build_inventory": {
            "status": "PASS" if build_ok else "FAIL",
            "command": cmd,
            "exit_code": made.returncode,
            "names": build_names,
        },
        "retired_cases": deleted,
        "primitive_source_preservation": {
            "status": "PASS" if source_ok else "FAIL",
            "files": sources,
            "archive_mismatches": archive_mismatches,
            "snapshot_manifest": evidence(snapshot_manifest),
            "snapshot_archive": evidence(snapshot_archive),
        },
        "historical_name_map": {
            **evidence(metadata),
            "unchanged_from_before_change": mapping_unchanged,
            "historical_snapshot_integrity": mapping_history_ok,
        },
        "live_retired_references": references,
        "explicit_negative_references": negative_refs,
        "historical_documentation_references": historical_refs,
        "symlinks": [str(p) for p in links],
        "broken_symlinks": broken_links,
        "case_source_symlinks": case_links,
        "simulator_executed": False,
        "fpga_submitted": False,
        "full_model_compiled": False,
        "full_model_executed": False,
        "limitations": [
            "Integer semantics checks inspect source and descriptor structure; they do not replace numerical or emitted-AME verification.",
            "Historical documents and explicit rejection tests may name retired cases.",
        ],
    }
    (output / "operator-naming-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    deletion_report = {
        key: report[key]
        for key in (
            "schema",
            "status",
            "operator_count",
            "inventory_status",
            "retired_cases",
            "primitive_source_preservation",
            "build_inventory",
            "dangling_dependencies",
            "broken_symlinks",
            "live_retired_references",
            "explicit_negative_references",
            "simulator_executed",
            "fpga_submitted",
            "full_model_compiled",
            "full_model_executed",
        )
    }
    (output / "inventory-deletion.json").write_text(
        json.dumps(deletion_report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Current operator inventory and naming audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"Current inventory: {len(cases)} cases; typed export/configuration checks: {sum(r['status'] == 'PASS' for r in descriptors)}/{len(cases)}.",
        f"Integer source structure: {sum(r['status'] == 'PASS' for r in semantics)}/{len(semantics)}; 5 accumulating gemm_i8 and 7 overwrite matmul_i8.",
        f"Preserved primitive source files: {sum(r['status'] == 'PASS' for r in sources)}/{len(sources)}; retired directories absent: {sum(r['directory_absent'] for r in deleted)}/{len(deleted)}.",
        f"Build inventory: {len(build_names)}; dangling dependencies: {len(dangling)}; broken links: {len(broken_links)}; live retired references: {len(references)}.",
        "",
        "This audit performs no compilation, numerical simulation, FPGA submission or complete-model execution. Historical name maps and rename evidence remain unchanged.",
        "",
    ]
    (output / "operator-naming-audit.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
