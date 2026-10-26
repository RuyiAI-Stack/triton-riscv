# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""Failure exit codes and memory contracts of the experimental BOSCAME entry."""

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from Inputs.pipeline import buddy_opt_path, experimental_pipeline  # noqa: E402

OPT = buddy_opt_path()
MODE = "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias"


def copy_kernel(size=16, prefix="", mask=False, gather=False, scatter=False):
    text = f"""module {{
      tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %indices: !tt.ptr<i32>) {{
        %r = tt.make_range {{start = 0 : i32, end = {size} : i32}} : tensor<{size}xi32>
        %s = tt.splat %src : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>>
        %d = tt.splat %dst : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>>
        {prefix}
        %sp = tt.addptr %s, %r : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32>
        %dp = tt.addptr %d, %r : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32>
    """
    if gather or scatter:
        text += f"""%ip = tt.splat %indices : !tt.ptr<i32> -> tensor<{size}x!tt.ptr<i32>>
        %ipp = tt.addptr %ip, %r : tensor<{size}x!tt.ptr<i32>>, tensor<{size}xi32>
        %idx = tt.load %ipp : tensor<{size}x!tt.ptr<i32>>
        %gp = tt.addptr {"%s" if gather else "%d"}, %idx : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32>
        """
    text += f"%v = tt.load {'%gp' if gather else '%sp'} : tensor<{size}x!tt.ptr<f32>>\n"
    if mask:
        text += f"""%limit = arith.constant dense<15> : tensor<{size}xi32>
        %mask = arith.cmpi slt, %r, %limit : tensor<{size}xi32>
        tt.store %dp, %v, %mask : tensor<{size}x!tt.ptr<f32>>
        """
    else:
        text += f"tt.store {'%gp' if scatter else '%dp'}, %v : tensor<{size}x!tt.ptr<f32>>\n"
    return text + "tt.return\n}\n}\n"


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def invoke(name, source, args=None, expect=None):
        args = experimental_pipeline() if args is None else args
        file = work / (name + ".mlir")
        file.write_text(source)
        result = subprocess.run(
            [OPT, str(file), *args], text=True, capture_output=True
        )
        if expect:
            assert result.returncode != 0, name + " silently succeeded"
            assert expect in result.stderr, (name, result.stderr)
        else:
            assert result.returncode == 0, (name, result.stderr)
        return result.stdout

    missing = invoke(
        "missing",
        "module {}",
        ("--triton-to-linalg-experimental=contract=true",),
    )
    assert "contract.version" in missing and "contract.abi" not in missing
    invoke(
        "old-backend-contract",
        'module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {}',
        ("--triton-to-linalg-experimental=contract=true",),
        "target attributes are forbidden in the neutral contract",
    )
    inherited = invoke(
        "inherited",
        "module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}} {}",
        ("--triton-to-linalg-experimental=contract=true",),
    )
    assert "args_aligned_bytes = 64" in inherited
    short = invoke(
        "short-store",
        copy_kernel(mask=True),
    )
    assert "vector<15xf32>" in short and short.count("bosc_ame.dma.start") == 1
    unaligned = copy_kernel(
        prefix="%one = arith.constant dense<1> : tensor<16xi32>\n%shift = arith.addi %r, %one : tensor<16xi32>"
    ).replace("%s, %r", "%s, %shift")
    unaligned = invoke(
        "unaligned",
        unaligned,
    )
    assert "vector.load" in unaligned
    assert "bosc_ame.direct_ddr" in unaligned
    assert unaligned.count("bosc_ame.dma.start") == 1
    invoke(
        "capacity",
        copy_kernel(size=131072),
        expect="SPM workspace capacity exceeded",
    )
    for name, options in (
        ("gather", {"gather": True}),
        ("scatter", {"scatter": True}),
    ):
        invoke(name, copy_kernel(**options), expect="unsupported")
    # Typed signed matmul selection no longer requires a backend source marker.
    ordinary = invoke(
        "unmarked",
        """module {
      func.func @target(%a: memref<16x64xi8>, %b: memref<64x16xi8, strided<[1, 64]>>, %c: memref<16x16xi32>) {
        linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%a, %b : memref<16x64xi8>, memref<64x16xi8, strided<[1, 64]>>) outs(%c : memref<16x16xi32>)
        return
      }
    }""",
    )
    assert "linalg.matmul" not in ordinary and "bosc_ame.tile.mma" in ordinary
    invoke(
        "dynamic-target",
        """module {
      func.func @target(%a: memref<?x64xi8>, %b: memref<64x16xi8, strided<[1, 64]>>, %c: memref<?x16xi32>) {
        linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%a, %b : memref<?x64xi8>, memref<64x16xi8, strided<[1, 64]>>) outs(%c : memref<?x16xi32>)
        return
      }
    }""",
        expect="matmul planning requires positive static rank-two buffers",
    )
    attrs = invoke(
        "attrs",
        """module {
      tt.func public @attrs(%x: i32 {test.arg = "arg"}) -> (i32 {test.result = "result"}) attributes {test.custom = "custom"} {
        tt.return %x : i32
      }
      tt.func private @hidden(%x: i32 {test.arg = "private_arg"}) -> (i32 {test.result = "private_result"}) attributes {test.private = "hidden"} { tt.return %x : i32 }
    }""",
    )
    assert "func.func @attrs" in attrs and 'test.custom = "custom"' in attrs
    assert 'test.arg = "arg"' in attrs and 'test.result = "result"' in attrs
    assert (
        "func.func private @hidden" in attrs
        and 'test.private = "hidden"' in attrs
    )
    assert (
        'test.arg = "private_arg"' in attrs
        and 'test.result = "private_result"' in attrs
    )
    signature = attrs.split("func.func @attrs", 1)[1].split(") ->", 1)[0]
    assert signature.count("%arg") == 7
    floating = invoke(
        "floating",
        """module {
      tt.func public @floating(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
        %c = arith.constant dense<0.0> : tensor<16x16xf32>
        %r = tt.dot %a, %b, %c : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
        tt.return %r : tensor<16x16xf32>
      }
    }""",
        args=(MODE,),
    )
    assert "linalg.matmul" in floating and "bosc_ame.tile.mma" not in floating
    # CHECK: PASS failures: old backend metadata, gather/scatter, SPM capacity, dynamic target
    print(
        "PASS failures: old backend metadata, gather/scatter, SPM capacity, dynamic target"
    )
    # CHECK: PASS routes: exact short store and unaligned load use planned RVV boundaries
    print(
        "PASS routes: exact short store and unaligned load use planned RVV boundaries"
    )
    # CHECK: PASS defaults: optional neutral ABI, function/argument/result attributes, single program-ID conversion, floating dot
    print(
        "PASS defaults: optional neutral ABI, function/argument/result attributes, single program-ID conversion, floating dot"
    )


if __name__ == "__main__":
    main()
