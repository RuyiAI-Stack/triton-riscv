from __future__ import annotations

import unittest

from codex_agent.summarize_operator_results import (
    build_failure_clusters,
    failure_signature,
    render_markdown,
)


class SummarizeOperatorResultsTests(unittest.TestCase):
    def test_clusters_equivalent_linalg_failures(self) -> None:
        results = [
            {
                "operator": "abs",
                "status": "failed",
                "failure_stage": "mlir-translate",
                "error_excerpt": [
                    "/tmp/tmp123/ll.mlir:10:5: error: Dialect `linalg' not found "
                    "for custom op 'linalg.generic'"
                ],
            },
            {
                "operator": "attention",
                "status": "failed",
                "failure_stage": "mlir-translate",
                "error_excerpt": [
                    "/tmp/tmp456/ll.mlir:99:5: error: Dialect `linalg' not found "
                    "for custom op 'linalg.generic'"
                ],
            },
        ]

        clusters = build_failure_clusters(results)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["signature"], "unlowered linalg.generic")
        self.assertEqual(clusters[0]["operators"], ["abs", "attention"])

    def test_normalizes_known_frontend_signature_and_renders_table(self) -> None:
        item = {
            "operator": "atan",
            "status": "failed",
            "failure_stage": "triton-frontend",
            "likely_reason": "Triton frontend compilation failed",
            "error_excerpt": [
                "AttributeError: module triton.language.math has no attribute atan"
            ],
        }

        self.assertEqual(
            failure_signature(item),
            "missing triton.language.math API",
        )
        markdown = render_markdown([item], latest_only=True)
        self.assertIn("## Failure Clusters", markdown)
        self.assertIn("missing triton.language.math API", markdown)
        self.assertIn("`atan`", markdown)

    def test_normalizes_buddy_dialect_and_reduction_signatures(self) -> None:
        ttx = {
            "error_excerpt": [
                "error: Dialect `ttx' not found for custom op 'ttx.cumsum'"
            ]
        }
        reduction = {
            "error_excerpt": [
                "error: unsupported linalg.reduce for -lower-linalg-to-vir"
            ]
        }

        self.assertEqual(failure_signature(ttx), "unregistered ttx dialect")
        self.assertEqual(
            failure_signature(reduction),
            "unsupported linalg.reduce form",
        )


if __name__ == "__main__":
    unittest.main()
