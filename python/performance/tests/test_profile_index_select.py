"""Run with the CPU benchmark environment and python/performance on PYTHONPATH."""

import unittest
from collections import defaultdict

import profile_index_select


class IndexSelectTraceTest(unittest.TestCase):
    def test_each_iteration_has_nested_host_intervals(self):
        events = profile_index_select.capture()["traceEvents"]
        calls = defaultdict(dict)
        for event in events:
            if event["ph"] != "X":
                continue
            args = event["args"]
            key = args["case"], args["phase"], args["iteration"], args["provider"]
            scopes = calls[key]
            self.assertNotIn(args["scope"], scopes)
            scopes[args["scope"]] = event

        cases = {key[0] for key in calls}
        self.assertEqual(len(cases), 3)
        for case in cases:
            for phase, count in (("correctness", 1), ("warmup", 5), ("measure", 20)):
                for iteration in range(1, count + 1):
                    key = case, phase, iteration
                    self.assertEqual(set(calls.pop((*key, "torch"))), {"provider_call"})
                    triton = calls.pop((*key, "triton-riscv"))
                    self.assertEqual(
                        set(triton),
                        {"provider_call", "output_allocation", "kernel_call"},
                    )
                    parent = triton["provider_call"]
                    allocation = triton["output_allocation"]
                    kernel = triton["kernel_call"]
                    # Trace Event timestamps are microseconds; allow 1 ns for rounding.
                    for child in (allocation, kernel):
                        self.assertEqual(
                            (parent["pid"], parent["tid"]), (child["pid"], child["tid"])
                        )
                        self.assertLessEqual(parent["ts"], child["ts"])
                        self.assertGreaterEqual(child["dur"], 0)
                        self.assertLessEqual(
                            child["ts"] + child["dur"],
                            parent["ts"] + parent["dur"] + 0.001,
                        )
                    self.assertLessEqual(
                        allocation["ts"] + allocation["dur"], kernel["ts"] + 0.001
                    )
        self.assertFalse(calls)
