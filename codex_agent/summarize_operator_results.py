#!/usr/bin/env python3
"""Summarize operator validation JSONL results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("agent-results")
FAILURE_SIGNATURES = (
    (re.compile(r"dialect `linalg'.*linalg\.generic", re.IGNORECASE), "unlowered linalg.generic"),
    (re.compile(r"dialect `tptr'.*not found", re.IGNORECASE), "unregistered tptr dialect"),
    (re.compile(r"dialect `ttx'.*not found", re.IGNORECASE), "unregistered ttx dialect"),
    (
        re.compile(r"unsupported linalg\.reduce for -lower-linalg-to-vir", re.IGNORECASE),
        "unsupported linalg.reduce form",
    ),
    (re.compile(r"unexpected op in ptr sequence", re.IGNORECASE), "unsupported pointer sequence"),
    (
        re.compile(r"triton\.language\.math.*has no attribute", re.IGNORECASE),
        "missing triton.language.math API",
    ),
    (
        re.compile(r"fp8.*not supported in this architecture|type fp8.*not supported", re.IGNORECASE),
        "unsupported FP8 target dtype",
    ),
    (re.compile(r"mismatched elements|assert_close", re.IGNORECASE), "numerical mismatch"),
)


def load_result_files(results_dir: Path, files: list[str]) -> list[Path]:
    if files:
        return [Path(item) for item in files]
    return sorted(results_dir.glob("operator-validation-*.jsonl"))


def read_results(paths: list[Path]) -> list[dict]:
    results: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"warning: ignored invalid JSON in {path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            item["_result_file"] = path.as_posix()
            results.append(item)
    return results


def latest_by_operator(results: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for item in results:
        latest[item["operator"]] = item
    return list(latest.values())


def failure_signature(item: dict) -> str:
    evidence = "\n".join(
        [
            *(item.get("error_excerpt") or []),
            item.get("likely_reason") or "",
        ]
    )
    for pattern, signature in FAILURE_SIGNATURES:
        if pattern.search(evidence):
            return signature

    reason = item.get("likely_reason")
    if reason:
        return reason

    excerpts = item.get("error_excerpt") or []
    if not excerpts:
        return "no stable error signature"
    signature = excerpts[0].lower()
    signature = re.sub(r"/tmp/tmp[^/\s'\"]+", "/tmp/<tmp>", signature)
    signature = re.sub(r":\d+(?::\d+)?", ":<line>", signature)
    signature = re.sub(r"\s+", " ", signature).strip()
    return signature[:160]


def build_failure_clusters(results: list[dict]) -> list[dict]:
    clusters: dict[tuple[str, str], list[str]] = {}
    for item in results:
        if item.get("status") != "failed":
            continue
        stage = item.get("failure_stage") or "unknown"
        signature = failure_signature(item)
        clusters.setdefault((stage, signature), []).append(item["operator"])

    return [
        {
            "stage": stage,
            "signature": signature,
            "operators": sorted(set(operators)),
        }
        for (stage, signature), operators in sorted(
            clusters.items(),
            key=lambda value: (-len(set(value[1])), value[0][0], value[0][1]),
        )
    ]


def render_markdown(results: list[dict], *, latest_only: bool) -> str:
    selected = latest_by_operator(results) if latest_only else results
    status_counts = Counter(item.get("status", "unknown") for item in selected)
    stage_counts = Counter(
        item.get("failure_stage") or "none"
        for item in selected
        if item.get("status") == "failed"
    )
    clusters = build_failure_clusters(selected)

    lines = [
        "# Operator Validation Summary",
        "",
        f"- Mode: {'latest result per operator' if latest_only else 'all attempts'}",
        f"- Results summarized: {len(selected)}",
        f"- Passed: {status_counts.get('passed', 0)}",
        f"- Failed: {status_counts.get('failed', 0)}",
        f"- Planned: {status_counts.get('planned', 0)}",
        f"- Skipped: {status_counts.get('skipped', 0)}",
        "",
        "## Failure Stages",
        "",
    ]

    if stage_counts:
        for stage, count in sorted(stage_counts.items()):
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Failure Clusters", ""])
    if clusters:
        lines.extend(
            [
                "| Stage | Stable Signature | Count | Operators |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for cluster in clusters:
            lines.append(
                "| {stage} | {signature} | {count} | {operators} |".format(
                    stage=cluster["stage"].replace("|", "\\|"),
                    signature=cluster["signature"].replace("|", "\\|"),
                    count=len(cluster["operators"]),
                    operators=", ".join(f"`{name}`" for name in cluster["operators"]),
                )
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Operator | Status | Failure Stage | Reason | Log |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for item in sorted(selected, key=lambda value: value["operator"]):
        lines.append(
            "| {operator} | {status} | {stage} | {reason} | {log} |".format(
                operator=item["operator"],
                status=item.get("status") or "",
                stage=item.get("failure_stage") or "",
                reason=(item.get("likely_reason") or "").replace("|", "\\|"),
                log=item.get("log_path") or "",
            )
        )

    failed = [item for item in selected if item.get("status") == "failed"]
    if failed:
        lines.extend(["", "## Failure Details", ""])
        for item in sorted(failed, key=lambda value: value["operator"]):
            lines.extend(
                [
                    f"### `{item['operator']}`",
                    "",
                    f"- Stage: {item.get('failure_stage') or 'unknown'}",
                    f"- Reason: {item.get('likely_reason') or 'unknown'}",
                    "",
                ]
            )
            excerpt = item.get("error_excerpt") or []
            if excerpt:
                safe_excerpt = [line.replace("```", "'''") for line in excerpt]
                lines.extend(["```text", *safe_excerpt, "```", ""])
            else:
                lines.extend(["No error excerpt was recorded for this result.", ""])

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize operator validation JSONL results."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific JSONL result files. Defaults to agent-results/operator-validation-*.jsonl.",
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    parser.add_argument(
        "--all-attempts",
        action="store_true",
        help="Summarize every attempt instead of only the latest result per operator.",
    )
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    paths = load_result_files(results_dir, args.files)
    results = read_results(paths)
    markdown = render_markdown(results, latest_only=not args.all_attempts)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {output_path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
