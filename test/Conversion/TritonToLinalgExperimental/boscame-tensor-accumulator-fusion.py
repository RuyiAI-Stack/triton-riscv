# RUN: %PYTHON %s %t triton-shared-opt
"""Integer accumulator semantics, local tensor fusion, and SSA preservation."""

import re
import subprocess
import sys
from pathlib import Path

A = "tensor<16x64xi8>"
B = "tensor<64x16xi8>"
C = "tensor<16x16xi32>"
IDENTITY = "affine_map<(i,j)->(i,j)>"


def product(initial=0):
    return f"""
  %zero = arith.constant {initial} : i32
  %empty = tensor.empty() : {C}
  %init = linalg.fill ins(%zero : i32) outs(%empty : {C}) -> {C}
  %product = linalg.matmul {{cast = #linalg.type_fn<cast_signed>,
      contract.origin_id = 731 : i64}}
      ins(%a, %b : {A}, {B}) outs(%init : {C}) -> {C}
"""


def add(lhs="%c", rhs="%product", *, swapped=False, broadcast=False, flags=""):
    lhs_type = "tensor<1x16xi32>" if broadcast else C
    lhs_map = "affine_map<(i,j)->(0,j)>" if broadcast else IDENTITY
    x, y = ("%y", "%x") if swapped else ("%x", "%y")
    return f"""
  %sum = linalg.generic {{
      indexing_maps = [{lhs_map}, {IDENTITY}, {IDENTITY}],
      iterator_types = ["parallel", "parallel"]}}
      ins({lhs}, {rhs} : {lhs_type}, {C}) outs(%empty : {C}) {{
    ^bb0(%x: i32, %y: i32, %unused: i32):
      %next = arith.addi {x}, {y} {flags} : i32
      linalg.yield %next : i32
  }} -> {C}
"""


def function(name, body, *, accumulator=C, returns=None):
    returns = returns or C
    return f"""
func.func @{name}(%a: {A}, %b: {B}, %c: {accumulator}) -> {returns} {{
{body}
}}
"""


def source():
    functions = [
        function("fuse", product() + add() + f"return %sum : {C}"),
        function(
            "swapped",
            product()
            + add("%product", "%c", swapped=True)
            + f"return %sum : {C}",
        ),
        function(
            "dense_zero",
            f"""
  %empty = tensor.empty() : {C}
  %zero = arith.constant dense<0> : {C}
  %product = linalg.matmul {{cast = #linalg.type_fn<cast_signed>}}
      ins(%a, %b : {A}, {B}) outs(%zero : {C}) -> {C}
"""
            + add()
            + f"return %sum : {C}",
        ),
        function(
            "late_accumulator",
            product()
            + f"%updated = arith.addi %c, %c : {C}\n"
            + add("%updated")
            + f"return %c, %sum : {C}, {C}",
            returns=f"({C}, {C})",
        ),
        function(
            "multiple_consumers",
            product() + add() + f"return %sum, %product : {C}, {C}",
            returns=f"({C}, {C})",
        ),
        function(
            "broadcast",
            product() + add(broadcast=True) + f"return %sum : {C}",
            accumulator="tensor<1x16xi32>",
        ),
        function("nonzero_product", product(7) + add() + f"return %sum : {C}"),
        function(
            "overflow_flags",
            product() + add(flags="overflow<nsw>") + f"return %sum : {C}",
        ),
        function(
            "effect_barrier",
            product()
            + "func.call @effect() : () -> ()\n"
            + add()
            + f"return %sum : {C}",
        ),
        function(
            "old_accumulator_live",
            product() + add() + f"return %c, %sum : {C}, {C}",
            returns=f"({C}, {C})",
        ),
        f"""
tt.func public @dot_nonzero(%a: {A}, %b: {B}) -> {C} {{
  %c = arith.constant dense<17> : {C}
  %result = tt.dot %a, %b, %c : {A} * {B} -> {C}
  tt.return %result : {C}
}}
tt.func public @dot_loop(%a: {A}, %b: {B}, %c: {C}, %upper: index) -> {C} {{
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %iv = %zero to %upper step %one
      iter_args(%acc = %c) -> ({C}) {{
    %next = tt.dot %a, %b, %acc : {A} * {B} -> {C}
    scf.yield %next : {C}
  }}
  tt.return %result : {C}
}}
func.func @legacy_loop(%a: {A}, %b: {B}, %c: {C}, %upper: index) -> {C} {{
  %zero_index = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %iv = %zero_index to %upper step %one
      iter_args(%acc = %c) -> ({C}) {{
    {product() + add("%acc")}
    scf.yield %sum : {C}
  }}
  return %result : {C}
}}
func.func @loaded_accumulator(%a: {A}, %b: {B}, %c: memref<16x16xi32>)
    -> {C} {{
  %storage = memref.alloc() : memref<16x16xi32>
  memref.copy %c, %storage {{
      contract.access = {{kind = "load", boundary = "external_ddr",
        mask = {{kind = "none"}}, origin_id = 732 : i64}}
  }} : memref<16x16xi32> to memref<16x16xi32>
  %snapshot = bufferization.to_tensor %storage restrict writable
      : memref<16x16xi32> to {C}
  {product() + add("%snapshot")}
  return %sum : {C}
}}
func.func @branches(%a: {A}, %b: {B}, %c: {C}, %full: i1) -> {C} {{
  %result = scf.if %full -> ({C}) {{
    {product() + add()}
    scf.yield %sum : {C}
  }} else {{
    {product() + add("%product", "%c", swapped=True)}
    scf.yield %sum : {C}
  }}
  return %result : {C}
}}
tt.func public @floating(%a: tensor<16x64xf32>, %b: tensor<64x16xf32>,
    %c: tensor<16x16xf32>) -> tensor<16x16xf32> {{
  %result = tt.dot %a, %b, %c : tensor<16x64xf32> * tensor<64x16xf32>
      -> tensor<16x16xf32>
  tt.return %result : tensor<16x16xf32>
}}
""",
    ]
    return (
        "module {\nfunc.func private @effect()\n"
        + "\n".join(functions)
        + "\n}\n"
    )


def bodies(text):
    matches = list(re.finditer(r"func.func (?:private )?@(\w+)\(", text))
    return {
        match[1]: text[match.start() : matches[index + 1].start()]
        if index + 1 < len(matches)
        else text[match.start() :]
        for index, match in enumerate(matches)
    }


def source_functions(text):
    """Retain each original positive/negative case as an isolated module."""
    pattern = r"(?:tt|func)\.func (?:public |private )?@(\w+)\("
    matches = list(re.finditer(pattern, text))
    for index, match in enumerate(matches):
        limit = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        opening = text.find("{", match.end(), limit)
        if opening < 0:
            continue
        depth = 1
        closing = opening + 1
        while depth:
            depth += (text[closing] == "{") - (text[closing] == "}")
            closing += 1
        yield match[1], text[match.start() : closing]


def main():
    prefix, opt = sys.argv[1:]
    input_path = Path(prefix + ".mlir")
    outputs = {}
    for name, function_text in source_functions(source()):
        module = (
            "module {\nfunc.func private @effect()\n" + function_text + "\n}\n"
        )
        input_path.write_text(module)
        result = subprocess.run(
            [
                opt,
                str(input_path),
                "--triton-to-linalg-experimental=contract=true "
                "caller-contract=args_aligned64_nonalias",
                "--mlir-print-ir-after-failure",
            ],
            capture_output=True,
            text=True,
        )
        if name == "broadcast":
            assert result.returncode != 0 and "V25" in result.stderr, (
                result.stderr
            )
            assert (
                "linalg.matmul" in result.stderr
                and "linalg.generic" in result.stderr
            )
            continue
        assert result.returncode == 0, (name, result.stderr)
        body = result.stdout
        assert "bosc_ame." not in body and "plan." not in body, name
        assert "tensor<" not in body and "bufferization." not in body, name
        assert "contract.task" in body or (
            name in ("dot_loop", "legacy_loop") and "complete = false" in body
        ), name
        outputs[name] = body
        Path(prefix + "." + name + ".mlir").write_text(body)

    for name in ("fuse", "swapped", "dense_zero", "old_accumulator_live"):
        body = outputs[name]
        assert body.count("linalg.matmul") == 1, name
        assert "linalg.generic" not in body and "linalg.fill" not in body, name
    for name in (
        "multiple_consumers",
        "nonzero_product",
        "overflow_flags",
        "effect_barrier",
    ):
        body = outputs[name]
        assert body.count("linalg.matmul") == 1, name
        assert "linalg.generic" in body and "linalg.fill" in body, name
    late = outputs["late_accumulator"]
    assert late.count("linalg.generic") == 1
    assert late.index("linalg.generic") < late.index("linalg.matmul")
    assert "linalg.fill" not in late
    assert outputs["branches"].count("linalg.matmul") == 2
    assert "linalg.generic" not in outputs["branches"]
    nonzero = outputs["dot_nonzero"]
    assert "arith.constant 17 : i32" in nonzero and "linalg.matmul" in nonzero
    assert nonzero.count("linalg.fill") == 1 and "linalg.generic" not in nonzero
    for name in ("dot_loop", "legacy_loop"):
        loop = outputs[name]
        # Dynamic bounds are retained with an explicit incomplete scope; they
        # must not gain a false static loop summary or a product-add sequence.
        assert "complete = false" in loop and "contract.scope" in loop
        assert "linalg.generic" not in loop and "linalg.fill" not in loop
    lineage = outputs["loaded_accumulator"]
    assert "accumulator_source_load_origin_ids = array<i64: 732>" in lineage
    assert "accumulator_source_lineage_complete = true" in lineage
    assert "linalg.matmul" in outputs["floating"]
    assert "arith.addf" in outputs["floating"]
    assert "origin_id = 731" in outputs["fuse"]
    old = outputs["old_accumulator_live"]
    # Returning old C keeps it live. One-Shot must copy before the update.
    assert "memref.copy %arg2," in old
    assert "outs(%arg2 : memref<16x16xi32" not in old
    print("contract accumulator fusion: all positive and negative cases passed")


if __name__ == "__main__":
    main()
