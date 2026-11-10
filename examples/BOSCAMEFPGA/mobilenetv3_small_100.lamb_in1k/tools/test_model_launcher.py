#!/usr/bin/env python3
"""Offline launcher/ABI regressions; mock adapters do not validate Triton math."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "export_model", ROOT / "tools/export_model.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class Tensor:
    """Minimal graph shape fixture, requiring neither torch nor model weights."""

    def __init__(self, elements):
        self.elements = elements
        self.shape = (elements,)

    def numel(self):
        return self.elements

    def detach(self):
        return self

    contiguous = clone = detach


class GraphTests(unittest.TestCase):
    def graph(self):
        return EXPORT.Graph(None, Tensor(8))

    def test_argument_count_cannot_be_silently_truncated_by_zip(self):
        graph = self.graph()
        with self.assertRaisesRegex(ValueError, "argument count"):
            graph.emit("bad", "relu_count8", ["input", "input"], Tensor(8))
        self.assertEqual(graph.nodes, [])

    def test_duplicate_output_cannot_overwrite_a_live_tensor(self):
        graph = self.graph()
        graph.emit("relu", "relu_count8", ["input"], Tensor(8))
        with self.assertRaisesRegex(ValueError, "duplicate graph output"):
            graph.emit("relu", "relu_count8", ["input"], Tensor(8))

    def test_residual_input_stays_live_until_output_is_allocated(self):
        graph = self.graph()
        graph.values.update({name: Tensor(16) for name in ("a", "b", "c", "d")})
        graph.nodes = [
            dict(args=["input", "a"], output="a"),
            dict(args=["a", "b"], output="b"),
            dict(args=["a", "b", "c"], output="c"),
            dict(args=["c", "d"], output="d"),
        ]
        graph.output = "d"
        offsets, size = graph.memory_plan()
        self.assertEqual(offsets, dict(a=0, b=64, c=128, d=0))
        self.assertEqual(size, 192)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get("CC") or shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("A native C compiler is required")
        cls.directory = tempfile.TemporaryDirectory(prefix="mobilenet-launcher-test-")
        cls.output = Path(cls.directory.name)
        graph = SimpleNamespace(
            nodes=[
                dict(
                    name="fixture.act",
                    case="hardswish_count1024",
                    args=["input", "activation"],
                    shapes=[(1024,), (1024,)],
                    output="activation",
                ),
                dict(
                    name="fixture.classifier",
                    case="linear_m1_n1000_k1024",
                    args=["activation", "weight", "bias", "logits"],
                    shapes=[(1, 1024), (1000, 1024), (1000,), (1, 1000)],
                    output="logits",
                ),
            ],
            output="logits",
        )
        golden = struct.pack("<1000f", *(i / 1000 for i in range(1000)))
        blob = bytes(1024 * 4 + 1000 * 1024 * 4)
        constants = dict(input=0, weight=1024 * 4, bias=len(blob))
        blob += golden
        blob += bytes(EXPORT.align64(len(blob)) - len(blob))
        golden_offset = len(blob)
        (cls.output / "model-data.bin").write_bytes(blob + golden)
        EXPORT.write_runner(
            graph,
            cls.output,
            constants,
            dict(activation=0, logits=4096),
            golden_offset,
            999,
        )
        cls.golden = golden
        (cls.output / "adapters.c").write_text(r"""
#include "support.h"
#include "model_support.h"
#include <stdlib.h>
#include <string.h>
void _mlir_ciface_kernel_hardswish_count1024(MemRef1 *x, MemRef1 *y) {
  if (getenv("MOCK_CONTEXT")) model_report_failure();
  for (unsigned i = 0; i < 1024; ++i) ((float *)y->aligned)[i] = ((float *)x->aligned)[i];
}
void _mlir_ciface_kernel_linear_m1_n1000_k1024(MemRef2 *x, MemRef2 *w,
                                            MemRef1 *b, MemRef2 *y) {
  (void)x; (void)w;
  for (unsigned i = 0; i < 1000; ++i) ((float *)y->aligned)[i] = ((float *)b->aligned)[i];
  const char *failure = getenv("MOCK_FAILURE");
  if (failure && !strcmp(failure, "mismatch")) ((float *)y->aligned)[0] += 1.0f;
  if (failure && !strcmp(failure, "nan")) ((float *)y->aligned)[0] = __builtin_nanf("");
}
""")
        cls.executable = cls.output / "check-launcher"
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-DHOST_TEST",
            "-I" + str(ROOT),
            "-I" + str(ROOT / "model"),
            str(ROOT / "support.c"),
            str(ROOT / "model/model_support.c"),
            str(cls.output / "runner.c"),
            str(cls.output / "model_blob.S"),
            str(cls.output / "adapters.c"),
            "-lm",
            "-o",
            str(cls.executable),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_launcher(self, **environment):
        clean = {
            key: value
            for key, value in os.environ.items()
            if key not in ("MOCK_FAILURE", "MOCK_CONTEXT", "MODEL_LOGITS_FILE")
        }
        return subprocess.run(
            [str(self.executable)],
            env={**clean, **environment},
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_generated_runner_prints_ordered_named_call_pairs_and_summary(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = re.findall(
            r"\[mobilenet-op\] (\d)/(\d) symbol=(\S+) case=(\S+) name=(\S+) (BEGIN|END)",
            result.stdout,
        )
        self.assertEqual(
            [(event[0], event[-1]) for event in events],
            [("1", "BEGIN"), ("1", "END"), ("2", "BEGIN"), ("2", "END")],
        )
        self.assertTrue(all(event[1] == "2" for event in events))
        self.assertIn(
            "kernel_returned=1 validation_executed=1 checked=1000 errors=0",
            result.stdout,
        )
        self.assertIn("started=2 completed=2 top1=999 expected_top1=999", result.stdout)
        self.assertIn("top5=999,998,997,996,995", result.stdout)
        self.assertIn("timer=clock-ticks", result.stdout)
        self.assertNotIn("cycles=", result.stdout)
        fnv = 0xCBF29CE484222325
        for value in self.golden:
            fnv = ((fnv ^ value) * 0x100000001B3) & ((1 << 64) - 1)
        self.assertIn(f"logits_fnv64=0x{fnv:016x}", result.stdout)

    def test_bad_logits_fail_with_index_and_bit_patterns(self):
        result = self.run_launcher(MOCK_FAILURE="mismatch")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(
            "mismatch index=0 actual_f32_bits=0x3f800000 expected_f32_bits=0x00000000",
            result.stdout,
        )
        self.assertIn("top1 mismatch", result.stdout)
        self.assertIn(": FAIL errors=", result.stdout)

    def test_nonfinite_logits_cannot_pass(self):
        result = self.run_launcher(MOCK_FAILURE="nan")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("mismatch index=0", result.stdout)
        self.assertIn("max_abs_error_f32_bits=0x7f800000", result.stdout)

    def test_failure_hook_reports_the_current_call_before_adapter_returns(self):
        result = self.run_launcher(MOCK_CONTEXT="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "FAIL in_flight=1 operators=2 started=1 completed=0 kernel_returned=0",
            result.stdout,
        )
        self.assertIn("name=fixture.act FAILED", result.stdout)


if __name__ == "__main__":
    unittest.main()
