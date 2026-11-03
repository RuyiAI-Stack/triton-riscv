#!/usr/bin/env python3
"""Discover operator-level validation targets.

This is the operator-focused MVP layer of the autonomous agent. It scans
operator implementation files, finds matching pytest tests, extracts Triton
kernel features, and emits one target per operator.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


OPERATOR_ROOT = Path("python/examples/flaggems")
OPERATOR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TL_OP_RE = re.compile(r"\btl\.([A-Za-z_][A-Za-z0-9_]*)")
TORCH_EXPR_RE = re.compile(r"\btorch\.[A-Za-z0-9_\.]+(?:\([^#\n]*\))?")


@dataclass
class OperatorTarget:
    name: str
    visibility: str
    implementation_file: str
    test_files: list[str] = field(default_factory=list)
    test_nodes: list[str] = field(default_factory=list)
    validation_command: str = ""
    triton_kernels: list[str] = field(default_factory=list)
    public_functions: list[str] = field(default_factory=list)
    tl_ops: list[str] = field(default_factory=list)
    torch_references: list[str] = field(default_factory=list)
    parametrize: list[str] = field(default_factory=list)
    risk_hints: list[str] = field(default_factory=list)


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_valid_operator_name(name: str) -> bool:
    return OPERATOR_NAME_RE.match(name) is not None


def parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(read_text(path))
    except SyntaxError:
        return None


def has_triton_jit_decorator(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text == "triton.jit" or text.startswith("triton.jit("):
            return True
    return False


def function_names(tree: ast.Module) -> tuple[list[str], list[str]]:
    triton_kernels: list[str] = []
    public_functions: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if has_triton_jit_decorator(node):
            triton_kernels.append(node.name)
        elif not node.name.startswith("_"):
            public_functions.append(node.name)
    return unique_sorted(triton_kernels), unique_sorted(public_functions)


def imported_modules(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level >= 1 and node.module:
            modules.append(node.module)
    return unique_sorted(modules)


def test_functions(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return unique_sorted(names)


def parametrize_decorators(tree: ast.Module, selected_tests: set[str]) -> list[str]:
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in selected_tests:
            continue
        for decorator in node.decorator_list:
            text = ast.unparse(decorator)
            if text.startswith("pytest.mark.parametrize"):
                values.append(text)
    return unique_sorted(values)


def operator_tokens(name: str) -> set[str]:
    tokens = {name}
    if "_and_" in name:
        tokens.add(name.split("_and_", 1)[0])
    return tokens


def select_relevant_tests(operator_name: str, tests: list[str]) -> list[str]:
    tokens = operator_tokens(operator_name)
    selected = [
        test_name
        for test_name in tests
        if any(token in test_name for token in tokens)
    ]
    return unique_sorted(selected)


def find_test_files(repo_root: Path, operator_name: str) -> list[Path]:
    root = repo_root / OPERATOR_ROOT
    matches: list[Path] = []
    direct_test = root / f"test_{operator_name}.py"
    if direct_test.exists():
        matches.append(direct_test)

    for path in sorted(root.glob("test_*.py")):
        source = read_text(path)
        if (
            f"from .{operator_name} import" in source
            or f"from .{operator_name} import (" in source
        ):
            matches.append(path)
    return sorted(set(matches), key=lambda path: path.as_posix())


def extract_torch_references(selected_tests_by_file: dict[Path, set[str]]) -> list[str]:
    refs: list[str] = []
    for path, selected_tests in selected_tests_by_file.items():
        source = read_text(path)
        tree = parse_python(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in selected_tests:
                continue
            segment = ast.get_source_segment(source, node) or ""
            refs.extend(match.group(0) for match in TORCH_EXPR_RE.finditer(segment))
    return unique_sorted(refs)


def infer_risk_hints(tl_ops: list[str]) -> list[str]:
    hints: list[str] = []
    op_set = set(tl_ops)
    if "erf" in op_set:
        hints.append("uses tl.erf; previous validation exposed linalg lowering risk")
    if {"exp", "log", "sqrt", "rsqrt"} & op_set:
        hints.append("uses transcendental math; check MLIR lowering and numerical tolerance")
    if "dot" in op_set:
        hints.append("uses tl.dot; check matrix lowering and dtype coverage")
    if {"load", "store"} & op_set:
        hints.append("uses memory operations; check masks, offsets, and boundary cases")
    if "where" in op_set:
        hints.append("uses tl.where; check mask lowering")
    return hints


def build_validation_command(test_nodes: list[str], test_files: list[str]) -> str:
    if test_nodes:
        targets = test_nodes
    else:
        targets = test_files
    if not targets:
        return ""
    return "python -m pytest -q " + " ".join(targets) + " -s"


def discover_operator(path: Path, repo_root: Path) -> OperatorTarget | None:
    tree = parse_python(path)
    if tree is None:
        return None

    implementation_source = read_text(path)
    triton_kernels, public_functions = function_names(tree)
    if not triton_kernels:
        return None

    operator_name = path.stem
    test_files = find_test_files(repo_root, operator_name)
    test_nodes: list[str] = []
    selected_tests_by_file: dict[Path, set[str]] = {}
    parametrizes: list[str] = []

    for test_file in test_files:
        test_tree = parse_python(test_file)
        if test_tree is None:
            continue
        selected = select_relevant_tests(operator_name, test_functions(test_tree))
        selected_tests_by_file[test_file] = set(selected)
        test_nodes.extend(f"{rel(test_file, repo_root)}::{name}" for name in selected)
        parametrizes.extend(parametrize_decorators(test_tree, set(selected)))

    test_file_strings = [rel(item, repo_root) for item in test_files]
    tl_ops = unique_sorted(TL_OP_RE.findall(implementation_source))

    return OperatorTarget(
        name=operator_name,
        visibility="internal" if operator_name.startswith("_") else "public",
        implementation_file=rel(path, repo_root),
        test_files=test_file_strings,
        test_nodes=unique_sorted(test_nodes),
        validation_command=build_validation_command(
            unique_sorted(test_nodes),
            test_file_strings,
        ),
        triton_kernels=triton_kernels,
        public_functions=public_functions,
        tl_ops=tl_ops,
        torch_references=extract_torch_references(selected_tests_by_file),
        parametrize=unique_sorted(parametrizes),
        risk_hints=infer_risk_hints(tl_ops),
    )


def discover_operators(repo_root: Path, *, include_invalid_names: bool = False) -> dict:
    root = repo_root / OPERATOR_ROOT
    operators: list[OperatorTarget] = []
    skipped_invalid_names: list[str] = []
    if root.exists():
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("test_") or path.name == "__init__.py":
                continue
            if not include_invalid_names and not is_valid_operator_name(path.stem):
                skipped_invalid_names.append(path.name)
                continue
            target = discover_operator(path, repo_root)
            if target is not None:
                operators.append(target)

    return {
        "schema_version": 1,
        "repo_root": repo_root.as_posix(),
        "summary": {
            "operators": len(operators),
            "public_operators": sum(
                1 for item in operators if item.visibility == "public"
            ),
            "internal_operators": sum(
                1 for item in operators if item.visibility == "internal"
            ),
            "operators_with_tests": sum(1 for item in operators if item.test_files),
            "operators_without_tests": sum(1 for item in operators if not item.test_files),
            "skipped_invalid_operator_files": len(skipped_invalid_names),
        },
        "skipped_invalid_operator_files": skipped_invalid_names,
        "operators": [asdict(item) for item in operators],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover operator-level Triton-RISCV validation targets."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--output",
        default=None,
        help="Write operator discovery JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--include-invalid-names",
        action="store_true",
        help="Include operator files whose stems are not valid Python identifiers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    result = discover_operators(repo_root, include_invalid_names=args.include_invalid_names)
    content = json.dumps(result, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")
        try:
            display_path = output_path.relative_to(repo_root)
        except ValueError:
            display_path = output_path
        print(f"wrote {display_path}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
