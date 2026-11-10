#!/usr/bin/env python3
"""Check C fixture bounds/oracles with host numerical stubs of logical shapes.

The stubs write exactly the KERNEL_CONFIG output shape. This catches stale
launcher element counts; it does not validate generated target instructions.
"""

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe  # noqa: E402

NAMES = ("add_128x768", "bias_add_128x768", "gelu_128x3072", "gelu_128x768")


def main():
    output = ROOT / "build/launch-numerics"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        c = describe(name)
        n = math.prod(c["shape"])
        args = c["arguments"]
        params = ", ".join(f"MemRef{a['rank']} *a{i}" for i, a in enumerate(args))
        lines = [
            '#include "support.h"',
            "#include <math.h>",
            f"void _mlir_ciface_kernel_{name}({params}) {{",
        ]
        for i in range(len(args)):
            lines.append(f"  float *p{i}=(float *)a{i}->aligned+a{i}->offset;")
        if c["family"] == "add":
            body = "p0[i]+p1[i]"
        elif c["family"] == "bias_add":
            body = f"p0[i]+p1[i%{c['shape'][-1]}]"
        else:
            body = "0.5f*p0[i]*(1.0f+(float)erf((double)(p0[i]*0.7071067811865475f)))"
        lines += [
            f"  for (size_t i=0;i<{n};++i) p{len(args) - 1}[i]={body};",
            "}",
            "extern int launch(void);",
            "int main(void) { return launch(); }",
        ]
        source = output / (name + ".c")
        source.write_text("\n".join(lines) + "\n")
        binary = output / name
        command = [
            shutil.which("cc") or "cc",
            "-std=c11",
            "-O2",
            "-DHOST_TEST",
            "-I" + str(ROOT),
            str(ROOT / name / "launch.c"),
            str(ROOT / "support.c"),
            str(source),
            "-lm",
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        (output / (name + ".log")).write_text(run.stdout + run.stderr)
        record = {
            "case": name,
            "status": "PASS" if run.returncode == 0 else "FAIL",
            "checked_elements": n,
            "validation": "launcher fixture/oracle only; host numerical stub; target kernel not executed",
        }
        records.append(record)
        print(record["status"], name, flush=True)
    (output / "report.json").write_text(
        json.dumps({"cases": records, "board_executed": False}, indent=2) + "\n"
    )
    return int(any(r["status"] != "PASS" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
