#!/usr/bin/env python3
"""Generate a Codex operator task file from the shared template."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "codex" / "templates" / "operator-task.md"
TASK_DIR = REPO_ROOT / "tasks" / "operators"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Triton-RISCV Codex operator task description."
    )
    parser.add_argument("operator", help="Operator name, for example silu or erf.")
    parser.add_argument(
        "--semantics",
        default="Describe the operator semantics before asking Codex to edit code.",
        help="Short operator semantics description.",
    )
    parser.add_argument(
        "--pytorch-reference",
        default=None,
        help="PyTorch reference expression, for example torch.erf(x).",
    )
    parser.add_argument(
        "--triton-reference",
        default=None,
        help="Existing Triton or FlagGems implementation path.",
    )
    parser.add_argument(
        "--inputs",
        default="input tensors required by the operator",
        help="Input contract text.",
    )
    parser.add_argument(
        "--outputs",
        default="output tensor matching the operator semantics",
        help="Output contract text.",
    )
    parser.add_argument(
        "--shapes",
        default="start with small one-dimensional shapes such as 512, 1023, and 1024",
        help="Shape coverage text.",
    )
    parser.add_argument(
        "--dtypes",
        default="start with torch.float32, then extend if supported",
        help="Dtype coverage text.",
    )
    parser.add_argument(
        "--implementation-file",
        default=None,
        help="Implementation file Codex may inspect or modify.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Pytest file Codex may inspect or modify.",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Command used to validate the operator.",
    )
    parser.add_argument(
        "--additional-files",
        default="no additional files",
        help="Additional files Codex should inspect.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output task file. Defaults to tasks/operators/<operator>.md.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing task file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operator = args.operator
    implementation_file = (
        args.implementation_file
        or f"python/examples/flaggems/{operator}.py"
    )
    test_file = args.test_file or f"python/examples/flaggems/test_{operator}.py"
    pytorch_reference = args.pytorch_reference or f"torch.<reference_for_{operator}>"
    triton_reference = args.triton_reference or implementation_file
    test_command = (
        args.test_command
        or f"python -m pytest -q {test_file} -s"
    )
    output_path = Path(args.output) if args.output else TASK_DIR / f"{operator}.md"
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    if output_path.exists() and not args.force:
        raise SystemExit(
            f"{output_path} already exists; pass --force to overwrite it."
        )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{OPERATOR_NAME}}": operator,
        "{{OPERATOR_SEMANTICS}}": args.semantics,
        "{{PYTORCH_REFERENCE}}": pytorch_reference,
        "{{TRITON_REFERENCE}}": triton_reference,
        "{{INPUTS}}": args.inputs,
        "{{OUTPUTS}}": args.outputs,
        "{{SHAPES}}": args.shapes,
        "{{DTYPES}}": args.dtypes,
        "{{ADDITIONAL_FILES_TO_INSPECT}}": args.additional_files,
        "{{IMPLEMENTATION_FILE}}": implementation_file,
        "{{TEST_FILE}}": test_file,
        "{{TEST_COMMAND}}": test_command,
    }

    content = template
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"generated {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
