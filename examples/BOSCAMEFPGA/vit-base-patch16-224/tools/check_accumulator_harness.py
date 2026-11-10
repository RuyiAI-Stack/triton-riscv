#!/usr/bin/env python3
"""Test diagnostic coverage, integer oracle samples and deliberate corruption.

This is a HOST_TEST of the validation harness with an explicit entry stub. It
does not establish candidate compiler or FPGA correctness. The stub checks all
real program IDs, scalar-computes output corners independently, then copies
reference tile interiors so guard/input/output detection can be falsified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from accumulator_validation import BUDDY, CASES, SUITE
from board_accumulator import generate

sys.path.insert(0, str(SUITE))
from cases import describe  # noqa: E402

STUB = r"""
#include "board_support.h"
#include <stdio.h>
#include <stdlib.h>
typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } D1;
typedef struct { int64_t rank; void *descriptor; } U;
extern unsigned vit_accumulator_order;
extern const uint32_t vit_accumulator_golden_normal[];
extern const uint32_t vit_accumulator_golden_stress[];
static _Alignas(64) unsigned char arena[VIT_WORKSPACE_BYTES];
static unsigned calls, rounds, faults, observed_errors, scalar_samples;
void *workspace(size_t offset) { if(offset>=sizeof arena) abort(); return arena+offset; }
void nh_e6_cache_writes(void) {}
void vit_board_validation_round(unsigned round,unsigned count,unsigned errors,
                                unsigned guards,unsigned inputs,uint64_t hash) {
  if(round!=@FIRST@u+rounds++ || count!=@M@u*@N@u || calls!=rounds*@PROGRAMS@u) abort();
  if(faults && round==@FAULT_ROUND@u && !(errors+guards+inputs)) abort();
  if((!faults || round!=@FAULT_ROUND@u) && errors+guards+inputs) abort();
  observed_errors+=errors+guards+inputs;
  printf("round=%u checked=%u errors=%u guards=%u inputs=%u hash=%016llx\n",
         round,count,errors,guards,inputs,(unsigned long long)hash);
}
int print_check(const char *name,unsigned errors,float maximum) {
  (void)name;(void)maximum;
  if(rounds!=@ROUNDS@ || errors!=observed_errors || !scalar_samples) abort();
  if(faults && !errors) abort();
  return errors!=0;
}
void _mlir_ciface_triton_@NAME@(U *ua,U *ub,U *uc,
                             int32_t nx,int32_t ny,int32_t nz,
                             int32_t x,int32_t y,int32_t z) {
  if(nx!=@NX@ || ny!=@NY@ || nz!=1) abort();
  unsigned ordinal=calls++%@PROGRAMS@u, trial=vit_accumulator_order;
  int expected_x=(int)(ordinal/(unsigned)ny), expected_y=(int)(ordinal%(unsigned)ny);
  if(trial) expected_x=nx-1-expected_x;
  if(trial==2) expected_y=ny-1-expected_y;
  if(x!=expected_x || y!=expected_y || z) abort();
  D1 *da=ua->descriptor,*db=ub->descriptor,*dc=uc->descriptor;
  const int8_t *a=da->aligned,*b=db->aligned;
  uint32_t *c=dc->aligned;
  const uint32_t *gold=trial?vit_accumulator_golden_stress:vit_accumulator_golden_normal;
  unsigned m0=(unsigned)x*32u,n0=(unsigned)y*64u;
  unsigned me=m0+32u<@M@u?m0+32u:@M@u,ne=n0+64u<@N@u?n0+64u:@N@u;
  for(unsigned i=m0;i<me;++i) for(unsigned j=n0;j<ne;++j) {
    if((i==m0 || i+1==me) && (j==n0 || j+1==ne)) {
      uint32_t value=@OVERWRITE@?0:c[(size_t)i*@N@+j];
      for(unsigned q=0;q<@K@u;++q)
        value+=(uint32_t)((int32_t)a[(size_t)i*@K@+q]*(int32_t)b[(size_t)j*@K@+q]);
      if(value!=gold[(size_t)i*@N@+j]) abort();
      ++scalar_samples;
    }
    c[(size_t)i*@N@+j]=gold[(size_t)i*@N@+j];
  }
  if(trial==@FAULT_ROUND@u && ordinal+1==@PROGRAMS@u) {
    if(faults==1) c[0]^=1;
    if(faults==2) ((unsigned char *)c)[(size_t)@M@*@N@*4]^=1;
    if(faults==3) ((unsigned char *)da->aligned)[0]^=1;
  }
}
int main(int argc,char **argv) {
  faults=argc>1?(unsigned)atoi(argv[1]):0;
  int status=launch();
  return !!status!=!!faults;
}
"""


def check(name, kernels, destination, compiler, corrupt, selected_round=None):
    case = describe(name)
    generate(name, destination, selected_round)
    config = [
        sys.executable,
        str(SUITE / "tools/board_config.py"),
        "--case",
        name,
        "--output",
        str(destination),
        "--accumulator-validation",
    ]
    if selected_round is not None:
        config += ["--accumulator-round", str(selected_round)]
    subprocess.run(config, check=True)
    subprocess.run(
        config + ["--adapter-source", str(kernels / name / "adapter.c")], check=True
    )
    m, n, k = case["shape"]
    stub = STUB
    for key, value in {
        "NAME": name,
        "M": m,
        "N": n,
        "K": k,
        "NX": case["grid"][0],
        "NY": case["grid"][1],
        "PROGRAMS": case["grid"][0] * case["grid"][1],
        "OVERWRITE": int(case["family"] == "matmul_i8"),
        "FIRST": 0 if selected_round is None else selected_round,
        "ROUNDS": 3 if selected_round is None else 1,
        "FAULT_ROUND": 1 if selected_round is None else selected_round,
    }.items():
        stub = stub.replace("@" + key + "@", str(value))
    (destination / "stub.c").write_text(stub)
    executable = destination / "harness-check"
    command = [compiler, "-std=c11", "-O2", "-DHOST_TEST"]
    for directory in (
        SUITE,
        SUITE / "tools",
        destination,
        BUDDY / "runtime/boscame/nh_e6",
        BUDDY / "runtime/boscame/nr",
    ):
        command += ["-I", str(directory)]
    command += [
        str(destination / file)
        for file in (
            "accumulator-launch.c",
            "adapter.c",
            "accumulator-golden.S",
            "stub.c",
        )
    ]
    command += ["-o", str(executable)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    runs = []
    for fault in range(4 if corrupt else 1):
        process = subprocess.run(
            [str(executable), str(fault)], check=True, capture_output=True, text=True
        )
        runs.append(
            {
                "fault": ("none", "output", "guard", "input")[fault],
                "stdout": process.stdout,
            }
        )
    return {
        "case": name,
        "status": "PASS",
        "scope": "host harness only",
        "round_ids": [0, 1, 2] if selected_round is None else [selected_round],
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=SUITE / "build/kernels")
    parser.add_argument("--case", action="append")
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--round", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="vit-accumulator-harness-") as temporary:
        for name in args.case or CASES:
            records.append(
                check(
                    name,
                    args.kernels.resolve(),
                    Path(temporary) / name,
                    args.compiler,
                    name.endswith("1x1000x768"),
                    args.round,
                )
            )
            print("PASS host harness: " + name, flush=True)
    result = {
        "schema": 1,
        "status": "PASS",
        "scope": "host harness selftest, not compiler/board execution",
        "cases": records,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
