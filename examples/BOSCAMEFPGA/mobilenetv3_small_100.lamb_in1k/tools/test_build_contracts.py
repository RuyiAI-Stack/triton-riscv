#!/usr/bin/env python3
"""Offline regressions for profile selection, adapter hooks and build evidence."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))
from cases import describe  # noqa: E402
from export import adapter_source, digest  # noqa: E402
import report  # noqa: E402


class BuildContracts(unittest.TestCase):
    @unittest.skipUnless(
        (REPO / "buddy-mlir/llvm/build/bin/clang").is_file() and shutil.which("ld.lld"),
        "RISC-V LLVM tools are unavailable",
    )
    def test_weak_trace_got_does_not_pad_binary_after_workspace(self):
        llvm = REPO / "buddy-mlir/llvm/build/bin"
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "weak.c"
            source.write_text("""extern void nr_kernel_begin(const char *) __attribute__((weak));
unsigned char arena[16 * 1024 * 1024]
  __attribute__((section(".workspace"), aligned(64), used));
void __attribute__((section(".text.init"))) _start(void) {
  if (nr_kernel_begin) nr_kernel_begin("probe");
  for (;;) __asm__ volatile("wfi");
}
""")
            obj, elf, image = base / "weak.o", base / "weak.elf", base / "weak.bin"
            subprocess.run(
                [
                    str(llvm / "clang"),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64imafdc_zicsr_zifencei",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-mno-relax",
                    "-fno-pie",
                    "-O2",
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ],
                check=True,
            )
            # Confirm this regression exercises an actual weak-hook GOT entry.
            relocs = subprocess.check_output(
                [str(llvm / "llvm-readelf"), "-r", str(obj)], text=True
            )
            self.assertIn("R_RISCV_GOT_HI20", relocs)
            subprocess.run(
                [
                    "ld.lld",
                    "-m",
                    "elf64lriscv",
                    "-T",
                    str(ROOT / "link_nh.ld"),
                    str(obj),
                    "-o",
                    str(elf),
                ],
                check=True,
            )
            subprocess.run(
                [str(llvm / "llvm-objcopy"), "-O", "binary", str(elf), str(image)],
                check=True,
            )
            sections = subprocess.check_output(
                [str(llvm / "llvm-readelf"), "-S", str(elf)], text=True
            )
            self.assertNotIn(" .got ", sections)
            self.assertLess(image.stat().st_size, 65536)

    def test_generated_adapter_brackets_validation_and_complete_grid(self):
        text = adapter_source(describe("relu_count8"))
        self.assertLess(
            text.index('nr_kernel_begin("relu_count8")'), text.index("if (!a0")
        )
        self.assertLess(text.index("x, y, z);"), text.index("if (ame_fence)"))
        self.assertLess(
            text.index("if (ame_fence)"), text.index('nr_kernel_end("relu_count8")')
        )
        self.assertIn("nr_kernel_begin(const char *) __attribute__((weak))", text)
        self.assertIn("nr_kernel_end(const char *) __attribute__((weak))", text)

    def test_build_report_rejects_modified_adapter_and_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            case = describe("relu_count8")
            directory = output / case["name"]
            directory.mkdir()
            for name in (
                "adapter.c",
                "contract-manifest.json",
                "kernel.ttir",
                "kernel.frontend.mlir",
            ):
                (directory / name).write_text("original " + name)
            manifest = dict(case)
            manifest.update(
                kernel_module_sha256=digest(case["kernel_source"]),
                exporter_sha256=digest(TOOLS / "export.py"),
                cases_sha256=digest(ROOT / "cases.py"),
                adapter_sha256=digest(directory / "adapter.c"),
                contract_manifest_sha256=digest(directory / "contract-manifest.json"),
                ttir_sha256=digest(directory / "kernel.ttir"),
            )
            (directory / "frontend.json").write_text(json.dumps(manifest))
            self.assertEqual(
                report.success(case["name"], output, "ttir")["status"], "PASS"
            )
            for name in ("adapter.c", "contract-manifest.json"):
                path = directory / name
                original = path.read_text()
                path.write_text("modified")
                with self.subTest(artifact=name), self.assertRaises(ValueError):
                    report.success(case["name"], output, "ttir")
                path.write_text(original)

    @unittest.skipUnless(shutil.which("cc"), "host C compiler is unavailable")
    def test_adapter_hooks_run_around_kernel_and_fence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            adapter = base / "adapter.c"
            adapter.write_text(adapter_source(describe("relu_count8")))
            harness = base / "harness.c"
            harness.write_text("""#include "support.h"
#include <assert.h>
static unsigned phase;
void nr_kernel_begin(const char *name) { assert(phase == 0); phase = 1; }
void _mlir_ciface_triton_relu_count8(void *x, void *y,
                                    int a, int b, int c, int d, int e, int f) {
  assert(phase == 1); phase = 2;
}
void ame_fence(void) { assert(phase == 2); phase = 3; }
void nr_kernel_end(const char *name) { assert(phase == 3); phase = 4; }
extern void _mlir_ciface_kernel_relu_count8(MemRef1 *, MemRef1 *);
int main(void) {
  float x[8] __attribute__((aligned(64)));
  float y[8] __attribute__((aligned(64)));
  MemRef1 a = make_1(x, 8), b = make_1(y, 8);
  _mlir_ciface_kernel_relu_count8(&a, &b);
  assert(phase == 4);
  return 0;
}
""")
            executable = base / "test"
            subprocess.run(
                [
                    "cc",
                    "-O2",
                    "-DHOST_TEST",
                    "-I" + str(ROOT),
                    str(adapter),
                    str(harness),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_model_rechecks_an_existing_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            (model / ".export-inputs.json").write_text(
                json.dumps(["checkpoint=", "image="], indent=2) + "\n"
            )
            for name in (
                "runner.c",
                "model_blob.S",
                "model-manifest.json",
                "model-data.bin",
                "golden-logits.f32",
            ):
                (model / name).write_text("fixture")
            (model / "model-cases.mk").write_text("MODEL_CASES := relu_count8\n")
            kernels = base / "kernels"
            directory = kernels / "relu_count8"
            directory.mkdir(parents=True)
            target = directory / "kernel.o"
            target.write_text("existing")
            (directory / "adapter.c").write_text("existing")
            fake = base / "make-case"
            calls = base / "calls.jsonl"
            fake.write_text(
                "#!" + sys.executable + "\n"
                "import json, pathlib, sys\n"
                "with pathlib.Path(" + repr(str(calls)) + ").open('a') as out:\n"
                "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake.chmod(0o755)
            for unused in range(2):
                result = subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "-f",
                        str(ROOT / "model/model.mk"),
                        str(target),
                        "OUT=" + str(model),
                        "KERNELS=" + str(kernels),
                        "MAKE=" + str(fake),
                        "PYTHON=" + sys.executable,
                        "CHECKPOINT=",
                        "IMAGE=",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            invocations = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(invocations), 2)
            for argv in invocations:
                self.assertEqual(argv[argv.index("-C") + 1], str(ROOT / "relu_count8"))
                self.assertIn("asm", argv)
                self.assertIn("OUT=" + str(directory), argv)

    def test_operator_output_ignores_inherited_model_output(self):
        # Execute the actual operator Makefile parser with a command-line OUT,
        # as propagated by a model build. No compiler or frontend is invoked.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            kernels, model = base / "kernels", base / "model"
            for makefile, profile in (
                ("common.mk", "nh-only"),
                ("board.mk", "nh-only"),
                ("board.mk", "nh-e6"),
            ):
                result = subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "-f",
                        str(ROOT / makefile),
                        "--eval=print-output: ; @printf '%s\\n' '$(OUT)'",
                        "print-output",
                        "OUT=" + str(model),
                        "KERNELS=" + str(kernels),
                        "BOARD_PROFILE=" + profile,
                    ],
                    cwd=ROOT / "relu_count8",
                    text=True,
                    capture_output=True,
                )
                with self.subTest(makefile=makefile, profile=profile):
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        result.stdout.strip(), str(kernels / "relu_count8")
                    )
            self.assertFalse(model.exists())

    def test_profile_switch_relinks_image_and_respects_kernel_directory(self):
        # A fake compiler makes this test independent of Triton and the RISC-V
        # toolchain. Make still resolves the actual profile-specific inputs.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            llvm = base / "llvm"
            llvm.mkdir()
            script = (
                "#!" + sys.executable + "\n"
                "import pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "out = pathlib.Path(args[args.index('-o') + 1])\n"
                "if pathlib.Path(sys.argv[0]).name == 'clang':\n"
                "    out.write_text('nh-only' if '-DBOSC_NH_ONLY=1' in args else 'nh-e6')\n"
                "else:\n"
                "    obj = next(pathlib.Path(x) for x in args if x.endswith('/board_support.o'))\n"
                "    out.write_text(obj.read_text())\n"
            )
            for name in ("clang", "ld.lld"):
                path = llvm / name
                path.write_text(script)
                path.chmod(0o755)
            kernels = base / "kernels"
            directory = kernels / "relu_count8"
            directory.mkdir(parents=True)
            (directory / "adapter.c").write_text("stub")
            (directory / "kernel.o").write_text("stub")
            elf = directory / "relu_count8.elf"
            for profile in ("nh-only", "nh-e6", "nh-only"):
                result = subprocess.run(
                    [
                        "make",
                        "--no-print-directory",
                        "-f",
                        str(ROOT / "board.mk"),
                        str(elf),
                        "BOARD_PROFILE=" + profile,
                        "KERNELS=" + str(kernels),
                        "BUDDY_ROOT=" + str(REPO / "buddy-mlir"),
                        "LLVM_BIN=" + str(llvm),
                        "PYTHON=" + sys.executable,
                    ],
                    cwd=ROOT / "relu_count8",
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(elf.read_text(), profile)
                stamp = json.loads((directory / ".board-profile").read_text())
                self.assertEqual(stamp, ["profile=" + profile])


if __name__ == "__main__":
    unittest.main()
