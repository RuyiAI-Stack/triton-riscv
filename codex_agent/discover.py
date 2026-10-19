#!/usr/bin/env python3
"""Discover Triton-RISCV validation targets.

The discovery output is intentionally mechanical and machine-readable. Later
agent stages can use it to choose targets, run tests, classify failures, and
summarize project coverage.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PYTEST_ROOT = Path("python/examples")
LIT_ROOT = Path("test")


TL_OP_RE = re.compile(r"\btl\.([A-Za-z_][A-Za-z0-9_]*)")
RUN_RE = re.compile(r"^\s*(?://|;)\s*RUN:\s*(.*)$")
PASS_RE = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_-]*)")


@dataclass
class PytestTarget:
    kind: str
    path: str
    command: str
    implementation_files: list[str] = field(default_factory=list)
    test_functions: list[str] = field(default_factory=list)
    triton_kernels: list[str] = field(default_factory=list)
    tl_ops: list[str] = field(default_factory=list)
    parametrize_count: int = 0
    likely_area: str = "unknown"


@dataclass
class LitTarget:
    kind: str
    path: str
    command: str
    run_lines: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    dialect_keywords: list[str] = field(default_factory=list)
    likely_area: str = "unknown"


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def has_triton_jit_decorator(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text == "triton.jit" or text.startswith("triton.jit("):
            return True
    return False


def count_parametrize_decorators(node: ast.FunctionDef) -> int:
    count = 0
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text.startswith("pytest.mark.parametrize"):
            count += 1
    return count


def local_imported_python_files(tree: ast.AST, source_path: Path) -> list[Path]:
    files: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level < 1 or not node.module:
            continue

        base = source_path.parent
        for _ in range(node.level - 1):
            base = base.parent

        candidate = base / f"{node.module.replace('.', '/')}.py"
        if candidate.exists():
            files.append(candidate)
    return unique_sorted_path(files)


def unique_sorted_path(values: Iterable[Path]) -> list[Path]:
    return sorted(set(values), key=lambda path: path.as_posix())


def infer_pytest_area(path: Path, tl_ops: list[str]) -> str:
    name = path.stem.removeprefix("test_")
    path_text = path.as_posix()
    op_set = set(tl_ops)

    if "flaggems" in path_text:
        return "operator-migration"
    if "dot" in op_set or "matmul" in name or "mm" in name:
        return "matrix"
    if {"sum", "max", "min"} & op_set or "reduce" in name or "norm" in name:
        return "reduction"
    if {"load", "store"} & op_set and ("gather" in name or "scatter" in name):
        return "memory-indexing"
    if {"exp", "erf", "sqrt", "rsqrt", "log"} & op_set:
        return "math"
    if "mask" in name or "where" in op_set:
        return "masking"
    return "general"


def discover_pytest_target(path: Path, repo_root: Path) -> PytestTarget | None:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    implementation_paths = local_imported_python_files(tree, path)
    implementation_sources: list[str] = []
    for implementation_path in implementation_paths:
        try:
            implementation_sources.append(implementation_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue

    test_functions: list[str] = []
    triton_kernels: list[str] = []
    parametrize_count = 0

    trees = [tree]
    for implementation_source in implementation_sources:
        try:
            trees.append(ast.parse(implementation_source))
        except SyntaxError:
            continue

    for index, current_tree in enumerate(trees):
        for node in ast.walk(current_tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if index == 0 and node.name.startswith("test_"):
                test_functions.append(node.name)
                parametrize_count += count_parametrize_decorators(node)
            if has_triton_jit_decorator(node):
                triton_kernels.append(node.name)

    if not test_functions and not triton_kernels:
        return None

    relative_path = rel(path, repo_root)
    combined_source = "\n".join([source, *implementation_sources])
    tl_ops = unique_sorted(TL_OP_RE.findall(combined_source))

    return PytestTarget(
        kind="pytest",
        path=relative_path,
        command=f"python -m pytest -q {relative_path} -s",
        implementation_files=[rel(item, repo_root) for item in implementation_paths],
        test_functions=unique_sorted(test_functions),
        triton_kernels=unique_sorted(triton_kernels),
        tl_ops=tl_ops,
        parametrize_count=parametrize_count,
        likely_area=infer_pytest_area(path, tl_ops),
    )


def infer_lit_area(path: Path, passes: list[str], dialects: list[str]) -> str:
    path_text = path.as_posix().lower()
    pass_text = " ".join(passes)

    if "sanitizer" in path_text:
        return "sanitizer"
    if "tritontostructured" in path_text or "triton-to-structured" in pass_text:
        return "triton-to-structured"
    if "structuredtomemref" in path_text or "structured-to-memref" in pass_text:
        return "structured-to-memref"
    if "tritontolinalg" in path_text or "triton-to-linalg" in pass_text:
        return "triton-to-linalg"
    if "tritonarithtolinalg" in path_text or "triton-arith-to-linalg" in pass_text:
        return "triton-arith-to-linalg"
    if "tritontoptr" in path_text or "triton-to-ptr" in pass_text:
        return "triton-to-ptr"
    if "memref" in dialects:
        return "memref"
    return "general"


def discover_lit_target(path: Path, repo_root: Path) -> LitTarget | None:
    source = path.read_text(encoding="utf-8")
    run_lines = [match.group(1).strip() for line in source.splitlines() if (match := RUN_RE.match(line))]
    if not run_lines:
        return None

    passes = unique_sorted(PASS_RE.findall("\n".join(run_lines)))
    dialect_keywords = unique_sorted(
        keyword
        for keyword in [
            "tt",
            "tts",
            "tptr",
            "linalg",
            "memref",
            "scf",
            "arith",
            "bufferization",
            "llvm",
        ]
        if re.search(rf"\b{re.escape(keyword)}\.", source)
    )
    relative_path = rel(path, repo_root)

    return LitTarget(
        kind="lit",
        path=relative_path,
        command=f"llvm-lit -sv {relative_path}",
        run_lines=run_lines,
        passes=passes,
        dialect_keywords=dialect_keywords,
        likely_area=infer_lit_area(path, passes, dialect_keywords),
    )


def discover(repo_root: Path) -> dict:
    pytest_targets: list[PytestTarget] = []
    lit_targets: list[LitTarget] = []

    pytest_root = repo_root / PYTEST_ROOT
    if pytest_root.exists():
        for path in sorted(pytest_root.rglob("test_*.py")):
            target = discover_pytest_target(path, repo_root)
            if target is not None:
                pytest_targets.append(target)

    lit_root = repo_root / LIT_ROOT
    if lit_root.exists():
        for path in sorted(list(lit_root.rglob("*.mlir")) + list(lit_root.rglob("*.ll"))):
            target = discover_lit_target(path, repo_root)
            if target is not None:
                lit_targets.append(target)

    area_counts: dict[str, int] = {}
    for target in [*pytest_targets, *lit_targets]:
        area_counts[target.likely_area] = area_counts.get(target.likely_area, 0) + 1

    return {
        "schema_version": 1,
        "repo_root": repo_root.as_posix(),
        "summary": {
            "pytest_targets": len(pytest_targets),
            "lit_targets": len(lit_targets),
            "total_targets": len(pytest_targets) + len(lit_targets),
            "area_counts": dict(sorted(area_counts.items())),
        },
        "pytest_targets": [asdict(target) for target in pytest_targets],
        "lit_targets": [asdict(target) for target in lit_targets],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover Triton-RISCV pytest and lit validation targets."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write discovery JSON to this path instead of stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    result = discover(repo_root)
    content = json.dumps(result, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {output_path.relative_to(repo_root)}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
