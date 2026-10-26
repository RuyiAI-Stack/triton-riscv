# RUN: %PYTHON %s %t
"""Registered AOT entry, strict inputs, real identities and driver isolation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import triton
import triton.language as tl
from triton.backends import _find_concrete_subclasses, backends
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.triton_shared import compiler
from triton.backends.triton_shared.contract import (
    BoscameOptions,
    export_contract,
)
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.experimental import gluon
from triton.experimental.gluon._runtime import GluonASTSource


@triton.jit
def copy_kernel(X, Y):
    i = tl.arange(0, 16)
    x = tl.load(X + i)
    tl.store(Y + i, x + 1.0)


@triton.jit
def conditional_kernel(X, Y, condition):
    i = tl.arange(0, 16)
    if condition:
        tl.store(Y + i, tl.load(X + i) + 1.0)
    else:
        tl.store(Y + i, tl.load(X + i) + 2.0)


@gluon.jit
def empty_gluon_kernel():
    pass


def reject(call, diagnostic):
    try:
        call()
    except (ValueError, TypeError) as error:
        assert diagnostic in str(error), str(error)
    else:
        raise AssertionError("expected rejection: " + diagnostic)


work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=True)
target = GPUTarget("boscame", "generic", 1)
source = ASTSource(copy_kernel, {"X": "*fp32", "Y": "*fp32"}, {})
assert _find_concrete_subclasses(compiler, BaseBackend) is compiler.CPUBackend
assert backends["triton_shared"].compiler is compiler.CPUBackend
assert type(make_backend(target)) is compiler.CPUBackend
cpu = make_backend(GPUTarget("cpu", "generic", 0))
assert cpu._aot is None
assert cpu.parse_options({}).enable_fp_fusion is False
assert not issubclass(type(make_backend(target)._aot), BaseBackend)

for grid in (
    None,
    (),
    (1,),
    (1, 1),
    (0, 1, 1),
    (-1, 1, 1),
    (True, 1, 1),
    (1.0, 1, 1),
    (1, 1, 2**31),
    (1, 1, 1, 1),
):
    reject(
        lambda grid=grid: export_contract(
            source, target=target, launch_grid=grid
        ),
        "launch_grid",
    )
for options, diagnostic in (
    ({"cpu_arch": "native"}, "unknown"),
    ({"num_warps": 4}, "unknown"),
    ({"debug": 1}, "boolean"),
    ({"enable_fp_fusion": True}, "forbids"),
    ({"allow_fp_reassoc": True}, "forbids"),
):
    reject(
        lambda options=options: export_contract(
            source, target=target, launch_grid=(1, 1, 1), options=options
        ),
        diagnostic,
    )
for invalid in (
    GPUTarget("boscame", "cpu", 1),
    GPUTarget("boscame", "generic", 32),
):
    reject(lambda invalid=invalid: make_backend(invalid), "requires GPUTarget")
reject(
    lambda: export_contract(
        source, target=GPUTarget("cpu", "generic", 0), launch_grid=(1, 1, 1)
    ),
    "requires the boscame",
)
reject(
    lambda: export_contract(
        source,
        target=target,
        launch_grid=(1, 1, 1),
        caller_contract="assume-all",
    ),
    "caller_contract",
)
reject(
    lambda: export_contract(
        source, target=target, launch_grid=(1, 1, 1), math_accuracy={"exp": {}}
    ),
    "four-field",
)
reject(
    lambda: export_contract(
        source, target=target, launch_grid=(1, 1, 1), entry_name="bad name"
    ),
    "entry_name",
)
reject(
    lambda: export_contract(
        GluonASTSource(empty_gluon_kernel, {}, {}),
        target=target,
        launch_grid=(1, 1, 1),
    ),
    "not Gluon",
)

# Both successful export paths and ordinary compile rejection must work with
# device construction disabled. Poison unrelated CPU/IME configuration too.
with (
    patch(
        "triton.runtime.driver._create_driver",
        side_effect=AssertionError("driver accessed"),
    ),
    patch.dict(
        os.environ,
        {
            "TRITON_RISCV_USE_IME": "1",
            "TRITON_RISCV_OPENMP_THREADS": "invalid",
            "TRITON_RISCV_CROSS_COMPILE": "1",
            "BUDDY_MLIR_BINARY_DIR": "/unavailable",
        },
    ),
):
    artifact = export_contract(
        source, target=target, launch_grid=(1, 1, 1), entry_name="aot_copy"
    )
    reject(lambda: triton.compile(source, target=target), "AOT-only")
    artifact.write(work / "artifact")
    path = work / "input.ttir"
    path.write_text(artifact.ttir)
    second = export_contract(
        path, target=target, launch_grid=(1, 1, 1), entry_name="aot_copy"
    )
    assert artifact.contract == second.contract
    promised = export_contract(
        path,
        target=target,
        launch_grid=(1, 1, 1),
        caller_contract="args_aligned64_nonalias",
        math_accuracy={
            "exp": {
                "domain": "all-f32",
                "special_values": "ieee",
                "max_abs_error": 1e-6,
                "max_rel_error": 1e-5,
            }
        },
    )

manifest = artifact.manifest
assert manifest["verified"] is True and manifest["contract_schema_version"] == 1
assert manifest["entry"]["name"] == "aot_copy"
assert manifest["entry"]["grid"] == [1, 1, 1]
assert len(manifest["entry"]["arguments"]) == 8
assert json.loads(json.dumps(manifest["effective_options"])) == json.loads(
    json.dumps(BoscameOptions().__dict__)
)
assert manifest["effective_promises"] == {
    "caller_contract": None,
    "math_accuracy": None,
}
assert (
    "contract.abi" not in artifact.contract
    and "contract.math" not in artifact.contract
)
assert (
    "contract.math" in promised.contract and "contract.abi" in promised.contract
)
for output, digest in (
    (artifact.ttir, "ttir_sha256"),
    (artifact.contract, "contract_sha256"),
):
    assert (
        hashlib.sha256(output.encode()).hexdigest()
        == manifest["outputs"][digest]
    )
identity = manifest["tool"]["build"]
assert identity["contract_schema_version"] == 1
for implementation in (
    "include/triton-shared/Analysis/KernelContract.h",
    "lib/Analysis/KernelContract.cpp",
):
    assert len(identity["frontend_files"][implementation]) == 64
assert identity["frontend_revision"] and identity["triton_revision"]
assert identity["llvm_build_files"] and identity["frontend_files"]
assert (
    manifest["triton"]["binary_sha256"]
    and manifest["triton"]["compiler_sha256"]
)
serialized = json.dumps(manifest)
assert (
    str(Path.home()) not in serialized
    and str(Path.cwd().resolve()) not in serialized
)
for forbidden in (
    "tt.",
    "tts.",
    "tptr.",
    "plan.",
    "bosc_ame.",
    "unrealized_conversion_cast",
):
    assert forbidden not in artifact.contract, forbidden

tool = os.environ["TRITON_SHARED_OPT_PATH"]


def raw_export(text, **kwargs):
    path = work / "raw.ttir"
    path.write_text(text)
    return export_contract(path, target=target, launch_grid=(1, 1, 1), **kwargs)


minimal = "module {tt.func public @entry() {tt.return}}"
for text, diagnostic in (
    ("module {func.func @f() {return}}", "Triton TTIR input only"),
    (
        minimal.replace(
            "module {", "module attributes {contract.version = 7 : i32} {"
        ),
        "V1",
    ),
    (
        minimal.replace(
            "@entry()",
            "@entry() attributes {contract.compute_graph = {version = 4 : i64}}",
        ),
        "V18",
    ),
    (
        minimal.replace(
            "@entry()",
            "@entry() attributes {gpu.known_grid_size = array<i32: 2, 1, 1>}",
        ),
        "launch-grid conflicts",
    ),
):
    reject(lambda text=text: raw_export(text), diagnostic)
conflicting = minimal.replace(
    "module {",
    "module attributes {contract.abi = {args_aligned_bytes = 32 : i64}} {",
)
reject(
    lambda: raw_export(conflicting, caller_contract="args_aligned64_nonalias"),
    "caller-contract conflicts",
)
math = '{exp = {domain = "all-f32", special_values = "ieee", max_abs_error = 0.0 : f64, max_rel_error = 0.0 : f64}}'
conflicting = minimal.replace(
    "@entry()", "@entry() attributes {contract.math = " + math + "}"
)
reject(
    lambda: raw_export(
        conflicting,
        math_accuracy={
            "exp": {
                "domain": "all-f32",
                "special_values": "ieee",
                "max_abs_error": 1e-6,
                "max_rel_error": 0,
            }
        },
    ),
    "math accuracy conflicts",
)

# A verified incomplete graph is an honest handoff, with its limitation intact.
reject(
    lambda: raw_export("""module {
  tt.func private @external()
  tt.func public @entry() {tt.call @external() : () -> () tt.return}
}"""),
    "uninlined calls are unsupported",
)
incomplete = export_contract(
    ASTSource(
        conditional_kernel, {"X": "*fp32", "Y": "*fp32", "condition": "i1"}, {}
    ),
    target=target,
    launch_grid=(1, 1, 1),
)
assert incomplete.manifest["entry"]["graph"]["complete"] is False
assert incomplete.manifest["entry"]["graph"]["limitation"]

# Inspect the preparation stage to check uses before inlining/symbol DCE.
path = work / "rename.ttir"
path.write_text("""module {
  tt.func public @entry() {tt.return}
  tt.func private @caller() {tt.call @entry() : () -> () tt.return}
}""")
renamed = subprocess.check_output(
    [tool, str(path), "--triton-contract-prepare=entry-name=renamed"], text=True
)
assert "tt.call @renamed" in renamed and "@entry" not in renamed
print(
    "PASS BOSCAME AOT API: AST/TTIR, plugin, promises, manifest, limits, rename, no driver"
)
