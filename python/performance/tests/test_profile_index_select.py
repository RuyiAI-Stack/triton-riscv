"""Run with the CPU benchmark environment and python/performance on PYTHONPATH."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import profile_index_select

KERNEL_FIELDS = {"kernel_hash", "kernel_name", "ttir_file", "ttsharedir_file"}


class IndexSelectTraceTest(unittest.TestCase):
    def test_each_iteration_has_nested_host_intervals(self):
        events = profile_index_select.capture()["traceEvents"]
        calls = defaultdict(dict)
        for event in events:
            if event["ph"] != "X":
                continue
            args = event["args"]
            self.assertTrue(KERNEL_FIELDS.isdisjoint(args))
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

    def test_bundle_links_to_the_launched_specialization(self):
        from test_performance_index_select import (
            INDEX_SELECT_CASES,
            index_select_row_kernel,
            make_index_select_inputs,
            run_index_select,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "capture"
            command = [
                sys.executable,
                profile_index_select.__file__,
                "--bundle",
                str(bundle),
            ]
            subprocess.run(command, check=True)
            trace = (bundle / "trace.json").read_bytes()
            calls = defaultdict(list)
            for event in json.loads(trace)["traceEvents"]:
                args = event["args"]
                if args.get("scope") == "kernel_call":
                    self.assertEqual(args["provider"], "triton-riscv")
                    calls[args["case"]].append(args)
                else:
                    self.assertTrue(KERNEL_FIELDS.isdisjoint(args))

            # Check the mapping against the object returned by an independent
            # launch of each specialization: swapped, valid files must fail too.
            referenced = set()
            hashes = set()
            launched = []

            def launch(*args, **kwargs):
                launched.append(index_select_row_kernel[(1,)](*args, **kwargs))

            for rows, cols, picks in INDEX_SELECT_CASES:
                run_index_select(
                    *make_index_select_inputs(rows, cols, picks), launch=launch
                )
                self.assertEqual(len(launched), 1)
                compiled = launched.pop()
                hashes.add(compiled.hash)
                case = f"bench_index_select(rows={rows}, cols={cols}, picks={picks})"
                records = calls.pop(case)
                self.assertEqual(len(records), 26)
                expected = {"kernel_hash": compiled.hash, "kernel_name": compiled.name}
                for stage in ("ttir", "ttsharedir"):
                    content = compiled.asm[stage].encode("utf-8")
                    digest = hashlib.sha256(content).hexdigest()
                    path = Path("kernels") / f"{digest}.{stage}.mlir"
                    self.assertEqual((bundle / path).read_bytes(), content)
                    referenced.add(bundle / path)
                    expected[stage + "_file"] = path.as_posix()
                for args in records:
                    self.assertEqual({k: args[k] for k in KERNEL_FIELDS}, expected)
            self.assertFalse(calls)
            self.assertEqual(len(hashes), 3)
            self.assertEqual(set((bundle / "kernels").iterdir()), referenced)

            # A retry against this bundle must not replace the user's capture.
            retry = subprocess.run(command, check=False, capture_output=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual((bundle / "trace.json").read_bytes(), trace)
