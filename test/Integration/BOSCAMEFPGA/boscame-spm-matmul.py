# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
"""SPM matmul inputs/output: direct tiles, physical layouts and vector packing."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from Inputs.pipeline import buddy_opt_path, experimental_pipeline  # noqa: E402


def physical_pipeline():
    # Explicitly placed buffer-form IR uses the backend compatibility entry.
    # It cannot cross the neutral frontend contract boundary.
    return [
        "--empty-tensor-to-alloc-tensor",
        "--one-shot-bufferize=bufferize-function-boundaries=true buffer-alignment=64",
        "--canonicalize",
        "--cse",
        "--linalg-to-boscame=enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias",
        "--promote-buffers-to-stack",
        "--canonicalize",
        "--cse",
    ]


def source_for(m, n, k, row_major_b, spm_c):
    pitch = ((k + 63) // 64) * 64
    a_count = m * pitch
    b_pitch = n if row_major_b else pitch
    b_count = k * n if row_major_b else n * pitch
    b_strides = f"{n}, 1" if row_major_b else f"1, {pitch}"
    c_count = m * n
    # A shifted C descriptor exercises physical offsets independently of the
    # bank-relative spm.view offset. Padding before/after C must survive.
    c_space = ", #bosc_ame.spm<1>" if spm_c else ""
    c_offset = 64 if spm_c else 0
    c_source = "%c_spm" if spm_c else "%arg2"
    c_setup = ""
    c_finish = ""
    if spm_c:
        c_setup = f"""
    %c_spm = bosc_ame.spm.view {{bank = 1 : i32, offset = 16384 : i64}} : memref<{c_count + 128}xi32, #bosc_ame.spm<1>>
    %c_flat = memref.subview %c_spm[64] [{c_count}] [1] : memref<{c_count + 128}xi32, #bosc_ame.spm<1>> to memref<{c_count}xi32, strided<[1], offset: 64>, #bosc_ame.spm<1>>
    %ct = bosc_ame.dma.start %arg2, %c_flat, %c_bytes {{channel = 0 : i32}} : (memref<{c_count}xi32>, memref<{c_count}xi32, strided<[1], offset: 64>, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %ct : !bosc_ame.dma_token<0>
"""
        c_finish = f"""
    %out = bosc_ame.dma.start %c_flat, %arg2, %c_bytes {{channel = 0 : i32}} : (memref<{c_count}xi32, strided<[1], offset: 64>, #bosc_ame.spm<1>>, memref<{c_count}xi32>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %out : !bosc_ame.dma_token<0>
"""
    text = f"""module attributes {{bosc_ame.dma.abi = "args_aligned64_nonalias"}} {{
  func.func @gemm(%arg0: memref<{a_count}xi8>, %arg1: memref<{b_count}xi8>, %arg2: memref<{c_count}xi32>) {{
    %a_bytes = arith.constant {a_count} : index
    %b_bytes = arith.constant {b_count} : index
    %c_bytes = arith.constant {c_count * 4} : index
    %a_spm = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : memref<{a_count}xi8, #bosc_ame.spm<0>>
    %b_spm = bosc_ame.spm.view {{bank = 1 : i32, offset = 0 : i64}} : memref<{b_count}xi8, #bosc_ame.spm<1>>
    %at = bosc_ame.dma.start %arg0, %a_spm, %a_bytes {{channel = 0 : i32}} : (memref<{a_count}xi8>, memref<{a_count}xi8, #bosc_ame.spm<0>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %at : !bosc_ame.dma_token<0>
    %bt = bosc_ame.dma.start %arg1, %b_spm, %b_bytes {{channel = 0 : i32}} : (memref<{b_count}xi8>, memref<{b_count}xi8, #bosc_ame.spm<1>>, index) -> !bosc_ame.dma_token<0>
    bosc_ame.dma.wait %bt : !bosc_ame.dma_token<0>
    %a = memref.reinterpret_cast %a_spm to offset: [0], sizes: [{m}, {k}], strides: [{pitch}, 1] : memref<{a_count}xi8, #bosc_ame.spm<0>> to memref<{m}x{k}xi8, strided<[{pitch}, 1]>, #bosc_ame.spm<0>>
    %b = memref.reinterpret_cast %b_spm to offset: [0], sizes: [{k}, {n}], strides: [{b_strides}] : memref<{b_count}xi8, #bosc_ame.spm<1>> to memref<{k}x{n}xi8, strided<[{b_strides}]>, #bosc_ame.spm<1>>
    {c_setup}
    %c = memref.reinterpret_cast {c_source} to offset: [{c_offset}], sizes: [{m}, {n}], strides: [{n}, 1] : memref<{c_count + 128 if spm_c else c_count}xi32{c_space}> to memref<{m}x{n}xi32, strided<[{n}, 1], offset: {c_offset}>{c_space}>
    linalg.matmul {{cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}} ins(%a, %b : memref<{m}x{k}xi8, strided<[{pitch}, 1]>, #bosc_ame.spm<0>>, memref<{k}x{n}xi8, strided<[{b_strides}]>, #bosc_ame.spm<1>>) outs(%c : memref<{m}x{n}xi32, strided<[{n}, 1], offset: {c_offset}>{c_space}>)
    {c_finish}
    return
  }}
}}
"""
    return text, pitch, b_pitch


def floating_source(row_major_b):
    b_row, b_col = (
        ("%row_scaled", "%col") if row_major_b else ("%row", "%col_scaled")
    )
    return f"""module {{
  tt.func public @floating(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {{
    %r = tt.make_range {{start = 0 : i32, end = 16 : i32}} : tensor<16xi32>
    %row = tt.expand_dims %r {{axis = 1 : i32}} : tensor<16xi32> -> tensor<16x1xi32>
    %col = tt.expand_dims %r {{axis = 0 : i32}} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen_r = arith.constant dense<16> : tensor<16x1xi32>
    %sixteen_c = arith.constant dense<16> : tensor<1x16xi32>
    %row_scaled = arith.muli %row, %sixteen_r : tensor<16x1xi32>
    %col_scaled = arith.muli %col, %sixteen_c : tensor<1x16xi32>
    %rows = tt.broadcast %row_scaled : tensor<16x1xi32> -> tensor<16x16xi32>
    %cols = tt.broadcast %col : tensor<1x16xi32> -> tensor<16x16xi32>
    %offset = arith.addi %rows, %cols : tensor<16x16xi32>
    %b_rows = tt.broadcast {b_row} : tensor<16x1xi32> -> tensor<16x16xi32>
    %b_cols = tt.broadcast {b_col} : tensor<1x16xi32> -> tensor<16x16xi32>
    %b_offset = arith.addi %b_rows, %b_cols : tensor<16x16xi32>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %cs = tt.splat %c : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %ap = tt.addptr %as, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %bp = tt.addptr %bs, %b_offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %cp = tt.addptr %cs, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %av = tt.load %ap : tensor<16x16x!tt.ptr<f32>>
    %bv = tt.load %bp : tensor<16x16x!tt.ptr<f32>>
    %cv = tt.load %cp : tensor<16x16x!tt.ptr<f32>>
    %result = tt.dot %av, %bv, %cv : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.store %cp, %result : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def check_floating(work, opt, rng):
    mode = experimental_pipeline()
    for row_major_b in (True, False):
        name = "floating-row-b" if row_major_b else "floating-column-b"
        source = work / (name + ".ttir")
        lowered = work / (name + ".mlir")
        loops = work / (name + ".loops.mlir")
        source.write_text(floating_source(row_major_b))
        result = subprocess.run(
            [opt, str(source), *mode, "-o", str(lowered)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        text = lowered.read_text()
        assert "bosc_ame.tile.mma" not in text
        assert "linalg.matmul" not in text
        assert "arith.mulf" in text and "vector<1xf32>" in text
        assert "vector.load" in text and "vector.store" in text
        assert all(
            "#bosc_ame.spm" not in line
            for line in text.splitlines()
            if "memref.load " in line or "memref.store " in line
        )
        subprocess.run(
            [
                opt,
                str(lowered),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
            capture_output=True,
        )
        machine = Machine(loops, "floating")
        guard = bytes([0xD3]) * 128
        for _ in range(2):
            a = rng.normal(size=(16, 16)).astype(np.float32)
            b = rng.normal(size=(16, 16)).astype(np.float32)
            initial = rng.normal(size=(16, 16)).astype(np.float32)
            stored_b = b if row_major_b else b.T.copy()
            buffers = {
                "%arg0": bytearray(a.tobytes() + guard),
                "%arg1": bytearray(stored_b.tobytes() + guard),
                "%arg2": bytearray(initial.tobytes() + guard),
            }
            machine.launch(buffers)
            # Spell out the original dot's independent f32 multiply/add
            # operations, followed by its one separate nonzero C addition.
            expected = np.zeros((16, 16), dtype=np.float32)
            for k in range(16):
                product = np.multiply(
                    a[:, k, None], b[None, k, :], dtype=np.float32
                )
                expected = np.add(expected, product, dtype=np.float32)
            expected = np.add(initial, expected, dtype=np.float32)
            actual = np.frombuffer(
                buffers["%arg2"], dtype=np.float32, count=256
            ).reshape(16, 16)
            np.testing.assert_array_equal(actual, expected)
            assert buffers["%arg0"] == a.tobytes() + guard
            assert buffers["%arg1"] == stored_b.tobytes() + guard
            assert buffers["%arg2"][-len(guard) :] == guard
        llvm = subprocess.run(
            [opt, str(lowered), "--lower-bosc-ame"],
            capture_output=True,
            text=True,
        )
        assert llvm.returncode == 0, llvm.stderr
        translated = subprocess.run(
            [os.environ["BUDDY_TRANSLATE"], "--buddy-to-llvmir"],
            input=llvm.stdout,
            capture_output=True,
            text=True,
        )
        assert translated.returncode == 0, translated.stderr
        print(
            f"PASS {name}: standard Linalg, exact f32 oracle, nonzero C and guards"
        )

    unsupported = """module {
      func.func @bad(%b: memref<16x16xf32>, %c: memref<16x16xf32>, %offset: index) {
        %storage = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<128xf32, #bosc_ame.spm<0>>
        %a = memref.reinterpret_cast %storage to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<128xf32, #bosc_ame.spm<0>> to memref<16x16xf32, strided<[16, 1]>, #bosc_ame.spm<0>>
        linalg.matmul ins(%a, %b : memref<16x16xf32, strided<[16, 1]>, #bosc_ame.spm<0>>, memref<16x16xf32>) outs(%c : memref<16x16xf32>)
        return
      }
    }"""
    for name, source, error in (
        (
            "floating-short-backing",
            unsupported,
            "leaves its spm.view allocation",
        ),
        (
            "floating-dynamic-offset",
            unsupported.replace("offset: [0]", "offset: [%offset]").replace(
                "strided<[16, 1]>", "strided<[16, 1], offset: ?>"
            ),
            "known nonnegative descriptor offset",
        ),
    ):
        path = work / (name + ".mlir")
        path.write_text(source)
        result = subprocess.run(
            [opt, str(path), *physical_pipeline()],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, name + " silently succeeded"
        assert error in result.stderr, result.stderr
    print(
        "PASS bounded floating snapshots: reject short backing and unknown offsets"
    )


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = buddy_opt_path()
    rng = np.random.default_rng(823)
    cases = [
        ("direct", 16, 16, 64, False, False),
        ("spm-c", 16, 16, 64, False, True),
        ("blocked", 4, 16, 256, False, True),
        ("row-b", 4, 64, 64, True, True),
        ("partial-k", 4, 16, 80, False, True),
    ]
    for name, m, n, k, row_b, spm_c in cases:
        source, pitch, _ = source_for(m, n, k, row_b, spm_c)
        original, lowered, loops = (
            work / (name + suffix)
            for suffix in (".mlir", ".lowered.mlir", ".loops.mlir")
        )
        original.write_text(source)
        result = subprocess.run(
            [
                opt,
                str(original),
                *physical_pipeline(),
                "-o",
                str(lowered),
            ],
            text=True,
            capture_output=True,
        )
        if row_b:
            assert result.returncode != 0
            assert (
                "AME tile addresses or parent pitch are not legal"
                in result.stderr
            )
            print("PASS row-b: reject unsupported physical B layout")
            continue
        if name == "partial-k":
            assert result.returncode != 0
            assert "K divisible by 64" in result.stderr
            print(
                "PASS partial-k: retain source and diagnose v1 frontend padding requirement"
            )
            continue
        assert result.returncode == 0, result.stderr
        text = lowered.read_text()
        assert text.count("bosc_ame.tile.mma") == (k + 63) // 64
        assert "linalg.matmul" not in text
        assert all(
            "#bosc_ame.spm" not in line
            for line in text.splitlines()
            if "memref.load " in line or "memref.store " in line
        )
        if name == "direct":
            # Keep the producer's A/B windows; legal external C is direct.
            assert text.count("bosc_ame.spm.view") == 2
            assert text.count("bosc_ame.dma.start") == 2
            assert all(
                "bosc_ame.direct_ddr" in line
                for line in text.splitlines()
                if "bosc_ame.tile.load_acc " in line
                or "bosc_ame.tile.store_acc " in line
            )
        subprocess.run(
            [
                opt,
                str(lowered),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
            capture_output=True,
        )
        # This wide-integer oracle is independent of packing and tile indices.
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        a[0, ::2] = -128
        b[::2, 0] = 127
        a_storage = np.full((m, pitch), 0x31, dtype=np.int8)
        a_storage[:, :k] = a
        if row_b:
            b_storage = b.copy()
        else:
            b_storage = np.full((n, pitch), -17, dtype=np.int8)
            b_storage[:, :k] = b.T
        guard = bytes([0xB7]) * 128
        machine = Machine(loops, "gemm")
        for _ in range(2):
            initial = rng.integers(-(1 << 31), 1 << 31, (m, n), dtype=np.int32)
            initial[0] = np.iinfo(np.int32).max
            buffers = {
                "%arg0": bytearray(a_storage.tobytes() + guard),
                "%arg1": bytearray(b_storage.tobytes() + guard),
                "%arg2": bytearray(initial.tobytes() + guard),
            }
            machine.launch(buffers)
            expected = (
                a.astype(np.int64) @ b.astype(np.int64)
                + initial.astype(np.int64)
            ).astype(np.int32)
            np.testing.assert_array_equal(
                np.frombuffer(
                    buffers["%arg2"], dtype=np.int32, count=m * n
                ).reshape(m, n),
                expected,
            )
            assert buffers["%arg0"] == a_storage.tobytes() + guard
            assert buffers["%arg1"] == b_storage.tobytes() + guard
            assert buffers["%arg2"][-len(guard) :] == guard
            if spm_c:
                assert (
                    machine.spm[1][16384 : 16384 + 256] == bytes([0x5A]) * 256
                )
                end = 16384 + 256 + m * n * 4
                assert machine.spm[1][end : end + 256] == bytes([0x5A]) * 256
        print(f"PASS {name}: exact signed i8/i32 wrap, initial C and guards")

    # Retain the exact 464-byte physical span fixture. V1 requires initialized
    # logical input content; an undersized physical backing must fail the
    # address proof before that content limitation is considered.
    source, _, _ = source_for(4, 16, 80, False, False)
    source = "\n".join(
        line
        for line in source.splitlines()
        if "%at = bosc_ame.dma.start" not in line
        and "bosc_ame.dma.wait %at " not in line
    )
    source = source.replace("memref<512xi8", "memref<480xi8")
    negative = work / "short-backing.mlir"
    negative.write_text(source)
    result = subprocess.run(
        [
            opt,
            str(negative),
            *physical_pipeline(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "read or partial inheritance has no initialized content version"
        in result.stderr
    )
    negative.write_text(source.replace("memref<480xi8", "memref<448xi8"))
    result = subprocess.run(
        [opt, str(negative), *physical_pipeline()],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SPM" in result.stderr
    print(
        "PASS bounded vector packing: v1 content boundary and short backing rejection"
    )

    source, _, _ = source_for(16, 16, 64, False, False)
    source = "\n".join(
        line
        for line in source.splitlines()
        if "%at = bosc_ame.dma.start" not in line
        and "bosc_ame.dma.wait %at " not in line
    )
    source = source.replace(
        "memref<1024xi8, #bosc_ame.spm<0>>",
        "memref<512xi8, #bosc_ame.spm<0>>",
    )
    negative = work / "direct-out-of-bounds.mlir"
    negative.write_text(source)
    result = subprocess.run(
        [
            opt,
            str(negative),
            *physical_pipeline(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "leaves its spm.view allocation" in result.stderr
    print("PASS bounded direct tiles: reject an oversized reinterpret cast")
    check_floating(work, opt, rng)


if __name__ == "__main__":
    main()

# CHECK: PASS direct:
# CHECK: PASS spm-c:
# CHECK: PASS blocked:
# CHECK: PASS row-b:
# CHECK: PASS partial-k:
# CHECK: PASS bounded vector packing:
# CHECK: PASS bounded direct tiles:
# CHECK: PASS floating-row-b:
# CHECK: PASS floating-column-b:
# CHECK: PASS bounded floating snapshots:
