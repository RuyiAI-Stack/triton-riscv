#!/usr/bin/env python3
"""Generate independent integer oracles and a three-round board diagnostic.

The generated launcher calls the real typed adapter and checks every i32 bit,
every input byte and both guards of A/B/C. It never links a host replacement
kernel. Normal and signed-extreme rounds have independent NumPy i64 oracles;
only the final modulo-i32 cast changes their representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE))
from cases import describe  # noqa: E402

I8 = [-128, -127, -1, 0, 1, 126, 127]
I32_BITS = [0x80000000, 0x80000030, 0xFFFFFFFF, 1, 0x7FFFFFD0, 0x7FFFFFFF]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv64(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def inputs(m, n, k, stress):
    i, j, q = np.arange(m)[:, None], np.arange(n)[:, None], np.arange(k)[None, :]
    if stress:
        a = np.take(np.array(I8, dtype=np.int8), (i * 13 + q * 3 + q // 7) % 7)
        b = np.take(np.array(I8, dtype=np.int8), (j * 11 + q * 5 + q // 7) % 7)
        c = np.take(
            np.array(I32_BITS, dtype="<u4"), (i * 3 + np.arange(n)[None, :]) % 6
        ).view("<i4")
    else:
        a = ((i * 13 + q * 7) % 31 - 15).astype(np.int8)
        b = ((j * 11 + q * 5) % 29 - 14).astype(np.int8)
        c = ((i * 3 + np.arange(n)[None, :]) % 7 - 3).astype("<i4")
    return tuple(np.ascontiguousarray(value) for value in (a, b, c))


LAUNCH = r"""/* Generated diagnostic; real kernel and typed ABI are unchanged. */
#include "board_support.h"
#include "board_case.h"
#define M @M@
#define N @N@
#define K @K@
#define OVERWRITE @OVERWRITE@
extern void _mlir_ciface_kernel_@NAME@(MemRef2 *, MemRef2 *, MemRef2 *);
extern const uint32_t vit_accumulator_golden_normal[];
extern const uint32_t vit_accumulator_golden_stress[];
unsigned vit_accumulator_order;
static const int8_t extreme_i8[] = {-128, -127, -1, 0, 1, 126, 127};
static const uint32_t extreme_i32[] = {
    0x80000000u, 0x80000030u, 0xffffffffu, 1u, 0x7fffffd0u, 0x7fffffffu};
static int8_t input_a(unsigned trial, unsigned i, unsigned q) {
  return trial ? extreme_i8[(i * 13u + q * 3u + q / 7u) % 7u]
               : (int8_t)((int)((i * 13u + q * 7u) % 31u) - 15);
}
static int8_t input_b(unsigned trial, unsigned j, unsigned q) {
  return trial ? extreme_i8[(j * 11u + q * 5u + q / 7u) % 7u]
               : (int8_t)((int)((j * 11u + q * 5u) % 29u) - 14);
}
static uint32_t initial_c(unsigned trial, unsigned i, unsigned j) {
  uint32_t value = trial ? extreme_i32[(i * 3u + j) % 6u]
                         : (uint32_t)((int)((i * 3u + j) % 7u) - 3);
  /* Same A/B with a different old C proves overwrite independence. */
  return OVERWRITE && trial == 2 ? ~value : value;
}
static uint64_t hash_bytes(const void *data, size_t bytes) {
  const unsigned char *p = data;
  uint64_t value = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0; i < bytes; ++i)
    value = (value ^ p[i]) * UINT64_C(0x100000001b3);
  return value;
}
static void guard_fill(unsigned char *data, size_t bytes) {
  for (unsigned i = 0; i < 64; ++i) {
    data[-64 + (int)i] = 0xc7u;
    data[bytes + i] = 0xc7u;
  }
}
static unsigned guard_check(const unsigned char *data, size_t bytes) {
  unsigned errors = 0;
  for (unsigned i = 0; i < 64; ++i)
    errors += data[-64 + (int)i] != 0xc7u || data[bytes + i] != 0xc7u;
  return errors;
}
int launch(void) {
  const size_t a_bytes = (size_t)M * K, b_bytes = (size_t)N * K;
  const size_t c_bytes = (size_t)M * N * 4;
  int8_t *a = workspace(64);
  int8_t *b = workspace(vit_align64(64 + a_bytes + 64) + 64);
  uint32_t *c = workspace(vit_align64((size_t)((unsigned char *)b -
                                           (unsigned char *)workspace(0)) +
                                   b_bytes + 64) + 64);
  unsigned total_errors = 0;
  for (unsigned trial = @FIRST_ROUND@; trial < @END_ROUND@; ++trial) {
    nh_e6_cache_writes();
    guard_fill((unsigned char *)a, a_bytes);
    guard_fill((unsigned char *)b, b_bytes);
    guard_fill((unsigned char *)c, c_bytes);
    for (unsigned i = 0; i < M; ++i)
      for (unsigned q = 0; q < K; ++q)
        a[(size_t)i * K + q] = input_a(trial, i, q);
    for (unsigned j = 0; j < N; ++j)
      for (unsigned q = 0; q < K; ++q)
        b[(size_t)j * K + q] = input_b(trial, j, q);
    for (unsigned i = 0; i < M; ++i)
      for (unsigned j = 0; j < N; ++j)
        c[(size_t)i * N + j] = initial_c(trial, i, j);
    MemRef2 A = make_2(a, M, K), B = make_2(b, N, K), C = make_2(c, M, N);
    vit_accumulator_order = trial;
    _mlir_ciface_kernel_@NAME@(&A, &B, &C);
    const uint32_t *expected = trial ? vit_accumulator_golden_stress
                                     : vit_accumulator_golden_normal;
    unsigned errors = 0, input_errors = 0;
    for (size_t element = 0; element < (size_t)M * N; ++element)
      errors += c[element] != expected[element];
    for (unsigned i = 0; i < M; ++i)
      for (unsigned q = 0; q < K; ++q)
        input_errors += a[(size_t)i * K + q] != input_a(trial, i, q);
    for (unsigned j = 0; j < N; ++j)
      for (unsigned q = 0; q < K; ++q)
        input_errors += b[(size_t)j * K + q] != input_b(trial, j, q);
    unsigned guard_errors = guard_check((unsigned char *)a, a_bytes) +
                            guard_check((unsigned char *)b, b_bytes) +
                            guard_check((unsigned char *)c, c_bytes);
    vit_board_validation_round(trial, (unsigned)((size_t)M * N), errors,
                               guard_errors, input_errors,
                               hash_bytes(c, c_bytes));
    total_errors += errors + guard_errors + input_errors;
  }
  return print_check("@NAME@", total_errors, total_errors ? 1.0f : 0.0f);
}
"""


def generate(name, destination, selected_round=None):
    case = describe(name)
    if case["family"] not in ("gemm_i8", "matmul_i8"):
        raise ValueError("accumulator diagnostics require an integer matmul case")
    m, n, k = case["shape"]
    overwrite = case["family"] == "matmul_i8"
    round_ids = [selected_round] if selected_round is not None else [0, 1, 2]
    if any(round_id not in (0, 1, 2) for round_id in round_ids):
        raise ValueError("diagnostic round must be 0, 1 or 2")
    destination.mkdir(parents=True, exist_ok=True)
    rounds = []
    for label, stress in (("normal", False), ("stress", True)):
        if selected_round is not None and stress != (selected_round != 0):
            continue
        a, b, c = inputs(m, n, k, stress)
        result = a.astype(np.int64) @ b.astype(np.int64).T
        if not overwrite:
            result += c.astype(np.int64)
        expected = result.astype("<i4").tobytes()
        filename = "accumulator-" + label + ".bin"
        (destination / filename).write_bytes(expected)
        rounds.append(
            {
                "name": label,
                "inputs_sha256": {
                    key: sha256(value.tobytes())
                    for key, value in zip(("A", "B", "C"), (a, b, c))
                },
                "output_sha256": sha256(expected),
                "output_fnv64": fnv64(expected),
                "file": filename,
                "bytes": len(expected),
            }
        )
    assembly = ['.section .vit_golden,"a",@progbits', ".balign 64"]
    for entry in rounds:
        symbol = "vit_accumulator_golden_" + entry["name"]
        assembly += [
            ".global " + symbol,
            symbol + ":",
            '.incbin "' + str((destination / entry["file"]).resolve()) + '"',
            ".balign 64",
        ]
    if selected_round is not None:
        missing = "stress" if selected_round == 0 else "normal"
        available = rounds[0]["name"]
        assembly += [
            ".global vit_accumulator_golden_" + missing,
            ".set vit_accumulator_golden_"
            + missing
            + ",vit_accumulator_golden_"
            + available,
        ]
    (destination / "accumulator-golden.S").write_text("\n".join(assembly) + "\n")
    replacements = {
        "NAME": name,
        "M": m,
        "N": n,
        "K": k,
        "OVERWRITE": int(overwrite),
        "FIRST_ROUND": round_ids[0],
        "END_ROUND": round_ids[-1] + 1,
    }
    launch = LAUNCH
    for key, value in replacements.items():
        launch = launch.replace("@" + key + "@", str(value))
    (destination / "accumulator-launch.c").write_text(launch)
    manifest = {
        "schema": 1,
        "case": name,
        "shape": [m, n, k],
        "oracle": "independent host NumPy signed-i64 matmul plus old C when required, final modulo-i32 cast",
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "kernel_source_sha256": sha256(Path(case["kernel_source"]).read_bytes()),
        "orders": [
            ["forward", "tail-first", "reverse"][round_id] for round_id in round_ids
        ],
        "round_ids": round_ids,
        "expected_invocations": len(round_ids),
        "expected_checked": m * n * len(round_ids),
        "bit_identical_i32": True,
        "guards": {"bytes_per_side_per_argument": 64, "byte": 199},
        "inputs_checked_bytewise": True,
        "overwrite_round_2_initial_c_complemented": overwrite,
        "rounds": rounds,
    }
    by_name = {entry["name"]: entry for entry in rounds}
    manifest["invocations"] = []
    for round_id in round_ids:
        reference = by_name["normal" if round_id == 0 else "stress"]
        input_hashes = dict(reference["inputs_sha256"])
        if overwrite and round_id == 2:
            _, _, old_c = inputs(m, n, k, True)
            input_hashes["C"] = sha256(np.bitwise_not(old_c).tobytes())
        manifest["invocations"].append(
            {
                "round": round_id,
                "order": ["forward", "tail-first", "reverse"][round_id],
                "inputs_sha256": input_hashes,
                "output_sha256": reference["output_sha256"],
                "output_fnv64": reference["output_fnv64"],
            }
        )
    (destination / "accumulator-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    generate(args.case, args.output.resolve(), args.round)


if __name__ == "__main__":
    main()
