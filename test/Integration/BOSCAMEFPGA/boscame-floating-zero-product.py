# RUN: %PYTHON %s %t contract-pipeline-opt
"""Eliminate zero-product storage without reassociating floating arithmetic."""

import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools")
)
from boscame_buffer_sim import Machine  # noqa: E402

A, B, C = "tensor<2x4xf32>", "tensor<4x2xf32>", "tensor<2x2xf32>"
M = "memref<2x2xf32>"
IDENTITY = "affine_map<(i,j)->(i,j)>"


def constant(name, value):
    return f"%{name} = arith.constant dense<{value}> : {C}\n"


def dot(accumulator="%zero", result="%result"):
    return f"{result} = tt.dot %a, %b, {accumulator} : {A} * {B} -> {C}\n"


def store(value, destination="%out", suffix=""):
    return (
        f"%buffer{suffix} = bufferization.to_buffer {value} : {C} to {M}\n"
        f"memref.copy %buffer{suffix}, {destination} : {M} to {M}\n"
    )


def explicit_add(swapped=False, flags="", multiple=False):
    lhs, rhs = ("%y", "%x") if swapped else ("%x", "%y")
    return (
        f"%empty = tensor.empty() : {C}\n"
        f"%init = linalg.fill ins(%scalar : f32) outs(%empty : {C}) -> {C}\n"
        f"%product = linalg.matmul ins(%a, %b : {A}, {B}) "
        f"outs(%init : {C}) -> {C}\n"
        + (store("%product", "%side", "_side") if multiple else "")
        + f"""
%result = linalg.generic {{
    indexing_maps = [{IDENTITY}, {IDENTITY}, {IDENTITY}],
    iterator_types = ["parallel", "parallel"]}}
    ins(%zero, %product : {C}, {C}) outs(%empty : {C}) {{
  ^bb0(%x: f32, %y: f32, %unused: f32):
    %sum = arith.addf {lhs}, {rhs} {flags} : f32
    linalg.yield %sum : f32
}} -> {C}
"""
    )


def module(name, body, *, side=False):
    extra = f", %side: {M}" if side else ""
    return f"""module {{
func.func @{name}(%am: memref<2x4xf32>, %bm: memref<4x2xf32>,
    %out: {M}{extra}) {{
  %a = bufferization.to_tensor %am restrict : memref<2x4xf32> to {A}
  %b = bufferization.to_tensor %bm restrict : memref<4x2xf32> to {B}
  {body}
  return
}}
}}
"""


def cases():
    for name, value in (("zero", "0.0"), ("negative_zero", "-0.0")):
        yield (
            name,
            module(name, constant("zero", value) + dot() + store("%result")),
        )
    for name, value in (
        ("nonzero", "1.0"),
        ("nan", "0x7FC01234"),
        ("positive_inf", "0x7F800000"),
        ("negative_inf", "0xFF800000"),
    ):
        yield (
            name,
            module(name, constant("zero", value) + dot() + store("%result")),
        )
    for name, swapped, flags in (
        ("explicit", False, ""),
        ("swapped", True, ""),
        ("fastmath", True, "fastmath<nnan>"),
    ):
        body = (
            constant("zero", "0.0")
            + "%scalar = arith.constant 0.0 : f32\n"
            + explicit_add(swapped, flags)
            + store("%result")
        )
        yield name, module(name, body)
    body = (
        constant("zero", "0.0")
        + "%scalar = arith.constant 0.0 : f32\n"
        + explicit_add(multiple=True)
        + store("%result")
    )
    yield "multiple_product", module("multiple_product", body, side=True)
    body = (
        constant("zero", "0.0")
        + store("%zero", "%side", "_side")
        + dot()
        + store("%result")
    )
    yield "multiple_zero", module("multiple_zero", body, side=True)
    body = (
        constant("zero", "0.0")
        + "%lower = arith.constant 0 : index\n"
        + "%upper = arith.constant 3 : index\n"
        + "%step = arith.constant 1 : index\n"
        + f"%result = scf.for %iv = %lower to %upper step %step "
        f"iter_args(%acc = %zero) -> ({C}) {{\n"
        + dot("%acc", "%next")
        + f"scf.yield %next : {C}\n}}\n"
        + store("%result")
    )
    yield "loop_k", module("loop_k", body)
    body = constant("zero", "0.0")
    for index in range(3):
        body += dot(
            "%zero" if index == 0 else f"%next{index - 1}", f"%next{index}"
        )
    yield "multi_k", module("multi_k", body + store("%next2"))


def run(opt, source, *passes):
    result = subprocess.run(
        [opt, *passes], input=source, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "LLVM ERROR" not in result.stderr
    return result.stdout


def matrix_allocations(text):
    return re.findall(r"(%[\w]+) = memref.alloc\(\).*: memref<2x2xf32>", text)


def bitwise_equal(actual, expected):
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert np.array_equal(
        actual.view(np.uint32)[finite], expected.view(np.uint32)[finite]
    ), (actual, expected)


def reference(a, b, initial, iterations=1):
    total = np.full((2, 2), initial, dtype=np.float32)
    with np.errstate(all="ignore"):
        for _ in range(iterations):
            product = np.zeros((2, 2), dtype=np.float32)
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        mul = np.float32(a[i, k] * b[k, j])
                        product[i, j] = np.float32(product[i, j] + mul)
            total = np.add(total, product, dtype=np.float32)
    return total


def numerics(work, name, source, opt, initial, iterations=1):
    loops = run(
        opt,
        source,
        "--convert-linalg-to-loops",
        "--lower-affine",
        "--canonicalize",
    )
    # The strict text interpreter accepts decimal floating constants. Decode
    # their actual emitted f32 bits only for this test transport; compiler IR
    # remains untouched, and NaN payload equality is not claimed by the oracle.
    loops = re.sub(
        r"(arith.constant )0x([0-9A-Fa-f]{8})( : f32)",
        lambda match: (
            match[1]
            + repr(
                struct.unpack("<f", int(match[2], 16).to_bytes(4, "little"))[0]
            )
            + match[3]
        ),
        loops,
    )
    path = work / f"{name}.loops.mlir"
    path.write_text(loops)
    guard = bytes([0xD7]) * 64
    inputs = [
        (
            np.array([[1, 2, 3, 4], [-2, -4, -8, -16]], dtype=np.float32),
            np.array([[1, -1], [2, -2], [4, -4], [8, -8]], dtype=np.float32),
        ),
        (
            np.array(
                [[-0.0, 0.0, -0.0, 0.0], [1e20, 1, -1e20, 1]], dtype=np.float32
            ),
            np.ones((4, 2), dtype=np.float32),
        ),
        (
            np.array(
                [[np.inf, 0, -np.inf, 1], [np.nan, -0.0, 1, 0]],
                dtype=np.float32,
            ),
            np.array([[1, 0], [0, -0.0], [1, 1], [1, -1]], dtype=np.float32),
        ),
    ]
    for a, b in inputs:
        machine = Machine(path, name)
        buffers = {
            "%arg0": bytearray(a.tobytes() + guard),
            "%arg1": bytearray(b.tobytes() + guard),
            "%arg2": bytearray(bytes(16) + guard),
        }
        if name in ("multiple_product", "multiple_zero"):
            buffers["%arg3"] = bytearray(bytes(16) + guard)
        machine.launch(buffers)
        actual = np.frombuffer(
            buffers["%arg2"], dtype=np.float32, count=4
        ).reshape(2, 2)
        bitwise_equal(actual, reference(a, b, initial, iterations))
        assert buffers["%arg0"] == a.tobytes() + guard
        assert buffers["%arg1"] == b.tobytes() + guard
        assert buffers["%arg2"][16:] == guard
        if "%arg3" in buffers:
            side = np.frombuffer(
                buffers["%arg3"], dtype=np.float32, count=4
            ).reshape(2, 2)
            expected = (
                reference(a, b, 0.0)
                if name == "multiple_product"
                else np.zeros((2, 2), dtype=np.float32)
            )
            bitwise_equal(side, expected)
            assert buffers["%arg3"][16:] == guard


def main():
    work, opt = Path(sys.argv[1]), sys.argv[2]
    work.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, source in cases():
        (work / f"{name}.ttir").write_text(source)
        output = run(
            opt,
            source,
            "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias",
        )
        outputs[name] = output
        (work / f"{name}.mlir").write_text(output)
        assert "tensor<" not in output and "bufferization." not in output
        assert output.count("linalg.matmul") == (3 if name == "multi_k" else 1)
        assert "arith.mulf" not in output and "math.fma" not in output
        if name == "loop_k":
            # The immutable zero product initializer is rematerialized in
            # the loop. Its content is not an invariant cross-scope read.
            loop = output[output.index("scf.for") :]
            assert "linalg.fill" in loop and "arith.addf" in loop
            assert "contract.compute_graph = {complete = true" in output
            assert "memref.copy" not in loop[: loop.index("linalg.matmul")]
        if name in ("zero", "negative_zero", "explicit", "swapped", "fastmath"):
            allocations = matrix_allocations(output)
            assert len(allocations) == 1, (name, output)
            assert output.count("memref.copy") == 1, (name, output)
            if name != "negative_zero":
                generic = output[output.index("linalg.generic") :]
                assert generic.count("arith.addf") == 1
                assert f"ins({allocations[0]} :" in generic
                assert f"outs({allocations[0]} :" in generic
                add = re.search(r"arith.addf (%[\w]+), (%[\w]+)", generic)
                assert add and add[1] != add[2], (name, generic)
            else:
                # Upstream arith folds x + -0 under round-to-nearest. The
                # storage normalization itself always retains the exact add.
                assert "arith.addf" not in output
            assert (
                "fastmath" not in output
                if name != "fastmath"
                else "fastmath<nnan>" in output
            )
        if name in (
            "nonzero",
            "nan",
            "positive_inf",
            "negative_inf",
            "multiple_product",
            "multi_k",
        ):
            assert len(matrix_allocations(output)) >= 2, (name, output)
            assert re.search(
                r"linalg.generic[^\n]* ins\(%[\w]+, %[\w]+ :", output
            ), name
    for name, initial in (
        ("zero", 0.0),
        ("negative_zero", -0.0),
        ("explicit", 0.0),
        ("swapped", 0.0),
        ("nonzero", 1.0),
        ("nan", np.nan),
        ("positive_inf", np.inf),
        ("negative_inf", -np.inf),
        ("multiple_product", 0.0),
        ("multiple_zero", 0.0),
    ):
        numerics(work, name, outputs[name], opt, initial)
    numerics(work, "multi_k", outputs["multi_k"], opt, 0.0, 3)
    numerics(work, "loop_k", outputs["loop_k"], opt, 0.0, 3)
    # A nonzero C accumulated inside the reduction gives 1, whereas the
    # required zero-product followed by C + product gives 2. This witness
    # makes the nonzero/multi-K checks sensitive to forbidden reassociation.
    a = np.array([[1e20, 1, -1e20, 1]] * 2, dtype=np.float32)
    b = np.ones((4, 2), dtype=np.float32)
    assert np.array_equal(reference(a, b, 1.0), np.full((2, 2), 2.0))
    reassociated = np.float32(1.0)
    for value in a[0]:
        reassociated = np.float32(reassociated + value)
    assert reassociated == np.float32(1.0)
    print(
        "PASS: floating zero-product storage and strict arithmetic boundaries"
    )


if __name__ == "__main__":
    main()
