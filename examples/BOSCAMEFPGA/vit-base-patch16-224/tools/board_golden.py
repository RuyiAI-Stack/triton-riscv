#!/usr/bin/env python3
"""Extract the independent HOST_TEST launch oracle, never a kernel.

The only launch transformation redirects original expected comparisons into
output writes.  The typed kernel adapter is a deliberate no-op that records
the actual operand shapes, workspace offsets and original input bytes.  No
Triton, BOSCAME or simulator output participates in this oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE))
from cases import describe  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c_float(value):
    """Emit a valid C float literal, including exact zero tolerances."""
    result = f"{value:.9g}"
    if "." not in result and "e" not in result.lower():
        result += ".0"
    return result + "f"


def remove_compare(source):
    match = re.search(r"static void compare\s*\(", source)
    if not match:
        return source
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[: match.start()] + source[end:]


HEADER = r"""#ifndef GOLDEN_CAPTURE_H
#define GOLDEN_CAPTURE_H
#include "support.h"
void golden_begin(void);
void golden_arg(const char *, const char *, int, int, void *, int64_t,
                const int64_t *, const int64_t *);
void golden_end(void);
void golden_float(float *, float, float, float);
int32_t golden_int32(int32_t *, int32_t);
#define compare(actual, expected, at, rt, errors, maximum) \
  golden_float(&(actual), (expected), (at), (rt))
#endif
"""

RUNTIME = r"""#include "golden-capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  const char *name, *dtype;
  int rank, output;
  unsigned char *data;
  int64_t offset, sizes[4], strides[4];
  size_t bytes, elements;
} Arg;
static Arg args[8];
static int argcnt, active;
static unsigned char *seen;
static size_t seen_size, checked;
static float tolerance_atol, tolerance_rtol;
static int tolerance_set;

static void save(const char *path, const void *data, size_t count) {
  FILE *file = fopen(path, "wb");
  if (!file || fwrite(data, 1, count, file) != count || fclose(file)) abort();
}
void golden_begin(void) {
  if (active) { fputs("unexpected multiple invocations\n", stderr); abort(); }
  active = 1;
}
void golden_arg(const char *name, const char *dtype, int rank, int output,
                void *aligned, int64_t offset, const int64_t *sizes,
                const int64_t *strides) {
  if (argcnt >= 8 || rank < 1 || rank > 4 || offset < 0) abort();
  Arg *arg = &args[argcnt++];
  arg->name=name; arg->dtype=dtype; arg->rank=rank; arg->output=output;
  arg->offset=offset; arg->elements=1;
  size_t width = strcmp(dtype, "i8") == 0 ? 1 : 4;
  arg->data = (unsigned char *)aligned + offset * width;
  for (int i=0; i<rank; ++i) {
    if (sizes[i] <= 0) abort();
    arg->sizes[i]=sizes[i]; arg->strides[i]=strides[i];
    arg->elements *= sizes[i];
  }
  int64_t contiguous_stride = 1;
  for (int i=rank-1; i>=0; --i) {
    if (strides[i] != contiguous_stride) abort();
    contiguous_stride *= sizes[i];
  }
  arg->bytes=arg->elements*width;
  char path[128];
  snprintf(path,sizeof(path),"round-000.%s.%s.bin",name,output?"initial":"input");
  save(path,arg->data,arg->bytes);
  if (output) {
    if (seen) abort();
    seen_size=arg->elements;
    seen=calloc(seen_size,1);
    if (!seen) abort();
  }
}
static void mark(void *pointer, size_t width) {
  for (int i=0;i<argcnt;++i) if(args[i].output) {
    Arg *arg=&args[i];
    ptrdiff_t byte=(unsigned char *)pointer-arg->data;
    if(byte<0 || (size_t)byte>=arg->bytes || byte%width) abort();
    size_t element=(size_t)byte/width;
    if(seen[element]) { fputs("duplicate expected element\n",stderr); abort(); }
    seen[element]=1; ++checked; return;
  }
  abort();
}
void golden_float(float *actual, float expected, float at, float rt) {
  mark(actual,4); *actual=expected;
  if (tolerance_set && (tolerance_atol!=at || tolerance_rtol!=rt)) abort();
  tolerance_set=1; tolerance_atol=at; tolerance_rtol=rt;
}
int32_t golden_int32(int32_t *actual, int32_t expected) {
  mark(actual,4); *actual=expected; return expected;
}
void golden_end(void) {
  if(!active || checked!=seen_size) {
    fprintf(stderr,"coverage checked=%zu required=%zu\n",checked,seen_size);
    abort();
  }
  FILE *meta=fopen("capture.json","w");
  if(!meta) abort();
  fprintf(meta,"{\"round\":0,\"checked\":%zu,\"atol\":%.9g,\"rtol\":%.9g,\"arguments\":[",checked,tolerance_atol,tolerance_rtol);
  for(int i=0;i<argcnt;++i) {
    Arg *arg=&args[i];
    fprintf(meta,"%s{\"name\":\"%s\",\"dtype\":\"%s\",\"rank\":%d,\"output\":%s,\"workspace_offset\":%td,\"bytes\":%zu,\"elements\":%zu,\"shape\":[",i?",":"",arg->name,arg->dtype,arg->rank,arg->output?"true":"false",arg->data-(unsigned char *)workspace(0),arg->bytes,arg->elements);
    for(int j=0;j<arg->rank;++j) fprintf(meta,"%s%lld",j?",":"",(long long)arg->sizes[j]);
    fputs("],\"strides\":[",meta);
    for(int j=0;j<arg->rank;++j) fprintf(meta,"%s%lld",j?",":"",(long long)arg->strides[j]);
    fputs("]}",meta);
    if(arg->output) {
      char path[128]; snprintf(path,sizeof(path),"round-000.%s.expected.bin",arg->name);
      save(path,arg->data,arg->bytes);
    } else {
      char path[128];
      snprintf(path,sizeof(path),"round-000.%s.input.bin",arg->name);
      FILE *input = fopen(path, "rb");
      unsigned char *original = malloc(arg->bytes);
      if (!input || !original || fread(original, 1, arg->bytes, input) != arg->bytes ||
          fclose(input) || memcmp(original, arg->data, arg->bytes)) abort();
      free(original);
    }
  }
  fputs("]}\n",meta); fclose(meta);
}
int launch(void);
int main(void) { int status=launch(); golden_end(); return status; }
"""


def generate(name, dest, compiler):
    if sys.byteorder != "little":
        raise ValueError(
            "host golden generation requires little-endian f32/i32 storage"
        )
    case = describe(name)
    if case["family"] not in ("gemm_i8", "matmul_i8", "patch_embed"):
        raise ValueError("unsupported family: " + case["family"])
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    source_path = Path(case["launch_source"])
    original = source_path.read_text()
    (dest / "launch-original.c").write_text(original)
    source = remove_compare(original)
    dtype = case["arguments"][-1]["dtype"]
    if dtype == "i32":
        source, count = re.subn(
            r"int32_t got = (c\[[^\]]+\]), dlt = got - want;",
            r"int32_t got = golden_int32(&(\1), want), dlt = got - want;",
            source,
        )
        if count != 1:
            raise ValueError("unexpected integer oracle structure: " + name)
    elif "compare(" not in source:
        raise ValueError("missing float oracle compare: " + name)
    source = '#include "golden-capture.h"\n' + source
    (dest / "launch-capture.c").write_text(source)
    (dest / "golden-capture.h").write_text(HEADER)
    (dest / "capture-runtime.c").write_text(RUNTIME)
    params = ", ".join(
        f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
    )
    stub = [
        '#include "golden-capture.h"',
        "/* Deliberate no-op; no kernel or simulation is linked. */",
        f"void _mlir_ciface_kernel_{name}({params}) {{",
        "  golden_begin();",
    ]
    for i, a in enumerate(case["arguments"]):
        output = int(i == len(case["arguments"]) - 1)
        stub.append(
            f'  golden_arg("{a["name"]}", "{a["dtype"]}", {a["rank"]}, {output}, a{i}->aligned, a{i}->offset, a{i}->sizes, a{i}->strides);'
        )
    stub.append("}")
    (dest / "adapter-noop.c").write_text("\n".join(stub) + "\n")
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-I",
        str(SUITE),
        "-I",
        str(dest),
        str(dest / "launch-capture.c"),
        str(dest / "capture-runtime.c"),
        str(dest / "adapter-noop.c"),
        str(SUITE / "support.c"),
        "-lm",
        "-o",
        str(dest / "oracle-host"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr) from error
    start = time.monotonic()
    proc = subprocess.run(
        [str(dest / "oracle-host")],
        cwd=dest,
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    (dest / "oracle-host.log").write_text(proc.stdout + proc.stderr)
    capture = json.loads((dest / "capture.json").read_text())
    output = next(a for a in capture["arguments"] if a["output"])
    if capture["checked"] != output["elements"]:
        raise ValueError("incomplete oracle coverage: " + name)
    captured_output = dest / f"round-000.{output['name']}.expected.bin"
    (dest / "golden.bin").write_bytes(captured_output.read_bytes())
    if captured_output.stat().st_size != output["bytes"]:
        raise ValueError("wrong golden output byte count: " + name)
    if dtype == "f32" and any(
        not math.isfinite(value[0])
        for value in struct.iter_unpack("<f", captured_output.read_bytes())
    ):
        raise ValueError("nonfinite independent expected output: " + name)
    output["file"] = "golden.bin"
    output["sha256"] = sha256(dest / output["file"])
    exact = dtype == "i32" or (capture["atol"] == 0 and capture["rtol"] == 0)
    output["tolerance"] = {
        "atol": capture["atol"],
        "rtol": capture["rtol"],
        "comparison": "exact"
        if exact
        else "abs(actual-expected) <= atol + rtol*abs(expected)",
    }
    for arg in capture["arguments"]:
        kind = "initial" if arg["output"] else "input"
        arg["initial_file"] = f"round-000.{arg['name']}.{kind}.bin"
        arg["initial_sha256"] = sha256(dest / arg["initial_file"])
    manifest = {
        "version": 1,
        "case": name,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "kernel_source_sha256": sha256(Path(case["kernel_source"])),
        "configuration_sha256": hashlib.sha256(
            json.dumps(case, sort_keys=True).encode()
        ).hexdigest(),
        "oracle": "Independent launch.c HOST_TEST C scalar expected formula; adapter deliberately no-op; no kernel/simulator is linked",
        "status": "reference_generated_only_not_board_validated",
        "host_compile_command": command,
        "host_oracle_seconds": elapsed,
        "round_count": 1,
        "validation": {
            "all_expected_elements_written_exactly_once": True,
            "original_inputs_unchanged_by_oracle": True,
            "contiguous_operand_strides_verified": True,
            "host_endianness": sys.byteorder,
        },
        "rounds": [
            {
                "round": 0,
                "arguments": capture["arguments"],
                "outputs": [output],
                "checked": capture["checked"],
            }
        ],
    }
    (dest / "golden_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Retain the original launch initialization and real adapter call. Replace
    # only the subsequent expensive expected calculation with blob reads.
    pattern = re.compile(r"_mlir_ciface_kernel_" + re.escape(name) + r"\s*\(([^;]+)\);")
    calls = list(pattern.finditer(original))
    if len(calls) != 2 or not original[: calls[0].start()].rstrip().endswith(
        "extern void"
    ):
        raise ValueError(
            "expected one extern declaration and one original adapter call: " + name
        )
    call = calls[-1]
    call_args = [a.strip() for a in call.group(1).split(",")]
    output_ref = call_args[-1].lstrip("&")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", output_ref):
        raise ValueError("unsupported output memref call expression: " + name)
    ctype = {"f32": "float", "i32": "int32_t"}[dtype]
    board = original[: call.end()]
    board += f"""\n  /* Independent host C reference: see golden_manifest.json. */
  extern const {ctype} vit_golden_values[];
  const {ctype} *actual = (const {ctype} *){output_ref}.aligned + {output_ref}.offset;
  unsigned errors = 0;
  float maximum = 0.0f;
  for (size_t element = 0; element < {output["elements"]}u; ++element) {{
    {ctype} got = actual[element], want = vit_golden_values[element];
"""
    if dtype == "i32":
        board += """    int64_t difference = (int64_t)got - (int64_t)want;
    if (difference < 0) difference = -difference;
    if ((float)difference > maximum) maximum = (float)difference;
    if (got != want) ++errors;
"""
    else:
        at, rt = capture["atol"], capture["rtol"]
        board += f"""    float difference = got - want;
    if (difference < 0) difference = -difference;
    if (difference > maximum) maximum = difference;
    if (!check_close(got, want, {c_float(at)}, {c_float(rt)})) ++errors;
"""
    board += "  }\n"
    postchecks = re.findall(
        r"/\* VIT_GOLDEN_POSTCHECK_BEGIN \*/(.*?)/\* VIT_GOLDEN_POSTCHECK_END \*/",
        original,
        re.S,
    )
    if len(postchecks) > 1:
        raise ValueError("unexpected multiple golden postcheck regions: " + name)
    if postchecks:
        # Preserve original guard/input checks around the inexpensive output
        # comparison. Only the expensive numeric oracle is replaced by a blob.
        board += postchecks[0]
        manifest["validation"]["original_guard_and_input_postchecks_preserved"] = True
    board += f'''
  return print_check("{name}", errors, maximum);
}}
'''
    (dest / "launch_board.c").write_text(board)
    (dest / "golden.S").write_text(
        "/* Generated independent C oracle; contains no board outputs. */\n"
        '.section .vit_golden,"a",@progbits\n.balign 64\n'
        ".global vit_golden_values\n.type vit_golden_values,@object\n"
        "vit_golden_values:\n"
        f'.incbin "{dest / output["file"]}"\n'
        ".size vit_golden_values, .-vit_golden_values\n"
        '.section .note.GNU-stack,"",@progbits\n'
    )
    verify_board_fixture(case, dest, capture, compiler, bool(postchecks))
    manifest["validation"]["board_initialization_matches_original_fixture"] = True
    manifest["validation"][
        "board_comparison_accepts_expected_and_rejects_corruption"
    ] = True
    if postchecks:
        manifest["validation"]["board_postchecks_reject_input_and_guard_corruption"] = (
            True
        )
    (dest / "golden_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{name}: {output['elements']} values, {output['bytes']} bytes, host oracle {elapsed:.3f}s",
        flush=True,
    )
    return manifest


def verify_board_fixture(case, dest, capture, compiler, postchecks=False):
    """Replay generated board initialization, then inject reference or corruption.

    This tests fixture preservation and checker behavior only. It deliberately
    replaces the kernel and must never be presented as hardware validation.
    """
    params = ", ".join(
        f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"])
    )
    output = next(a for a in capture["arguments"] if a["output"])
    ctype = {"f32": "float", "i32": "int32_t"}[output["dtype"]]
    lines = [
        '#include "support.h"',
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        f"const {ctype} vit_golden_values[{output['elements']}];",
        "static void fixture(const char *path, void *actual, size_t bytes) {",
        '  FILE *input = fopen(path, "rb");',
        "  void *wanted = malloc(bytes);",
        "  if (!input || !wanted || fread(wanted,1,bytes,input)!=bytes ||",
        "      fclose(input) || memcmp(wanted,actual,bytes)) abort();",
        "  free(wanted);",
        "}",
        f"void _mlir_ciface_kernel_{case['name']}({params}) {{",
    ]
    for i, arg in enumerate(capture["arguments"]):
        lines += [
            f"  if (a{i}->offset != 0 || (unsigned char *)a{i}->aligned -",
            f"      (unsigned char *)workspace(0) != {arg['workspace_offset']}) abort();",
        ]
        for axis, size in enumerate(arg["shape"]):
            lines += [
                f"  if (a{i}->sizes[{axis}] != {size} || a{i}->strides[{axis}] != {arg['strides'][axis]}) abort();"
            ]
        fixture_path = json.dumps(str(dest / arg["initial_file"]))
        lines += [f"  fixture({fixture_path}, a{i}->aligned, {arg['bytes']});"]
    last = len(capture["arguments"]) - 1
    lines += [
        f"  memcpy(a{last}->aligned, vit_golden_values, {output['bytes']});",
        f'  if (getenv("VIT_GOLDEN_CORRUPT")) (({ctype} *)a{last}->aligned)[0] += 1;',
    ]
    if postchecks:
        lines += [
            '  if (getenv("VIT_GOLDEN_CORRUPT_INPUT")) ((unsigned char *)a0->aligned)[0] ^= 1;',
            '  if (getenv("VIT_GOLDEN_CORRUPT_GUARD")) ((unsigned char *)a0->aligned)[-1] ^= 1;',
        ]
    lines += ["}", "int launch(void);", "int main(void) { return launch(); }"]
    harness = dest / "fixture-check.c"
    harness.write_text("\n".join(lines) + "\n")
    # A tiny assembly incbin is shared by host and RISC-V and carries the
    # exact same 32-bit representation. The C declaration is only extern.
    harness.write_text(
        harness.read_text().replace(
            f"const {ctype} vit_golden_values[",
            f"extern const {ctype} vit_golden_values[",
        )
    )
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-DHOST_TEST",
        "-I",
        str(SUITE),
        str(dest / "launch_board.c"),
        str(harness),
        str(SUITE / "support.c"),
        str(dest / "golden.S"),
        "-lm",
        "-o",
        str(dest / "fixture-check"),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    accepted = subprocess.run(
        [str(dest / "fixture-check")], capture_output=True, text=True
    )
    if accepted.returncode or ": PASS errors=00000000" not in accepted.stdout:
        raise ValueError(
            "generated board fixture/checker rejected original expected values: "
            + accepted.stdout
            + accepted.stderr
        )
    import os

    env = dict(os.environ, VIT_GOLDEN_CORRUPT="1")
    rejected = subprocess.run(
        [str(dest / "fixture-check")], env=env, capture_output=True, text=True
    )
    if rejected.returncode != 1 or ": FAIL errors=00000001" not in rejected.stdout:
        raise ValueError(
            "generated board checker failed corruption test: "
            + rejected.stdout
            + rejected.stderr
        )
    logs = accepted.stdout + rejected.stdout
    if postchecks:
        for kind in ("INPUT", "GUARD"):
            rejected = subprocess.run(
                [str(dest / "fixture-check")],
                env=dict(os.environ, **{f"VIT_GOLDEN_CORRUPT_{kind}": "1"}),
                capture_output=True,
                text=True,
            )
            if (
                rejected.returncode != 1
                or ": FAIL errors=00000001" not in rejected.stdout
            ):
                raise ValueError(
                    f"generated board checker failed {kind.lower()} corruption test: "
                    + rejected.stdout
                    + rejected.stderr
                )
            logs += rejected.stdout
    (dest / "fixture-check.log").write_text(logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    generate(args.case, args.output, args.cc)


if __name__ == "__main__":
    main()
