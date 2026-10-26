# RUN: %PYTHON %s %t contract-pipeline-opt
"""Launch/loop proofs, compact snapshots, graph rebuilding and pitched DMA."""

import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ["BUDDY_ROOT"]) / "examples/FPGA-BOSCAME/tools"
sys.path.insert(0, str(TOOLS))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_sim import op_name, read_module  # noqa: E402

SYMBOL = "triton_matmul_i8_196x768x768"
BUFFERIZE = []
BACKEND = [
    "--bosc-ame-plan",
    "--bosc-ame-plan-verify",
    "--bosc-ame-emit",
    "--canonicalize",
    "--cse",
]


def walk(ops):
    for op in ops:
        yield op
        for block in op.blocks:
            yield from walk(block)


def check_graph(text, functions=1):
    bodies = re.split(r"  func.func @", text)[1:]
    assert len(bodies) == functions
    for function in bodies:
        tasks = re.findall(r"contract.task = \{([^}]+)\}", function)
        ids = [int(re.search(r"(?:^|, )id = (\d+)", task)[1]) for task in tasks]
        assert sorted(ids) == list(range(12)), ids
        assert "contract.compute_graph = {complete = true" in function
        assert function.count("contract.loop_carried_slots") == 2
        for task in tasks:
            node = int(re.search(r"(?:^|, )id = (\d+)", task)[1])
            predecessors = re.search(
                r"predecessors = array<i64(?:: ([^>]+))?>", task
            )[1]
            branch = set(range(0, 5) if node < 5 else range(5, 12))
            assert (
                not predecessors
                or set(map(int, predecessors.split(","))) <= branch
            )
        slots = re.findall(
            r"contract.loop_carried_slots = \[\{([^}]+)\}\]", function
        )
        for index, slot in enumerate(slots):
            branch = set(range(0, 5) if index == 0 else range(5, 12))
            for field in (
                "backedge_producer_compute_ids",
                "exit_consumer_compute_ids",
                "init_producer_compute_ids",
                "iter_arg_consumer_compute_ids",
            ):
                values = re.search(field + r" = array<i64: ([^>]+)>", slot)[1]
                assert set(map(int, values.split(","))) <= branch
        # Full/tail clones share the source event, but keep distinct task IDs.
        dots = [task for task in tasks if 'kind = "dot"' in task]
        assert len(dots) == 2 and all(
            "input_values = array<i64:" in task for task in dots
        )


def copy_source():
    """Two partial program axes, unequal pitches and nonzero descriptor offsets."""
    return """
module {
  tt.func public @pitched(%src: !tt.ptr<i32>, %dst: !tt.ptr<i32>) {
    %bm = arith.constant 32 : i32
    %oa = arith.constant 16 : i32
    %ob = arith.constant 32 : i32
    %m = arith.constant dense<36> : tensor<32x1xi32>
    %n = arith.constant dense<80> : tensor<1x32xi32>
    %pa = arith.constant dense<128> : tensor<32x1xi32>
    %pb = arith.constant dense<160> : tensor<32x1xi32>
    %other = arith.constant dense<7> : tensor<32x32xi32>
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %xs = arith.muli %x, %bm : i32
    %ys = arith.muli %y, %bm : i32
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %xb = tt.splat %xs : i32 -> tensor<32xi32>
    %yb = tt.splat %ys : i32 -> tensor<32xi32>
    %rows = arith.addi %xb, %r : tensor<32xi32>
    %cols = arith.addi %yb, %r : tensor<32xi32>
    %rr = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cc = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %mr = arith.cmpi slt, %rr, %m : tensor<32x1xi32>
    %mc = arith.cmpi slt, %cc, %n : tensor<1x32xi32>
    %mrb = tt.broadcast %mr : tensor<32x1xi1> -> tensor<32x32xi1>
    %mcb = tt.broadcast %mc : tensor<1x32xi1> -> tensor<32x32xi1>
    %mask = arith.andi %mrb, %mcb : tensor<32x32xi1>
    %ac = arith.muli %rr, %pa : tensor<32x1xi32>
    %bc = arith.muli %rr, %pb : tensor<32x1xi32>
    %s = tt.addptr %src, %oa : !tt.ptr<i32>, i32
    %d = tt.addptr %dst, %ob : !tt.ptr<i32>, i32
    %sb = tt.splat %s : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>>
    %db = tt.splat %d : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>>
    %sr = tt.addptr %sb, %ac : tensor<32x1x!tt.ptr<i32>>, tensor<32x1xi32>
    %dr = tt.addptr %db, %bc : tensor<32x1x!tt.ptr<i32>>, tensor<32x1xi32>
    %sa = tt.broadcast %sr : tensor<32x1x!tt.ptr<i32>> -> tensor<32x32x!tt.ptr<i32>>
    %da = tt.broadcast %dr : tensor<32x1x!tt.ptr<i32>> -> tensor<32x32x!tt.ptr<i32>>
    %cb = tt.broadcast %cc : tensor<1x32xi32> -> tensor<32x32xi32>
    %sp = tt.addptr %sa, %cb : tensor<32x32x!tt.ptr<i32>>, tensor<32x32xi32>
    %dp = tt.addptr %da, %cb : tensor<32x32x!tt.ptr<i32>>, tensor<32x32xi32>
    %v = tt.load %sp, %mask, %other : tensor<32x32x!tt.ptr<i32>>
    tt.store %dp, %v, %mask : tensor<32x32x!tt.ptr<i32>>
    tt.return
  }
}
"""


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    opt = sys.argv[2]

    def run(name, source, grid=None, passes=(), reject=None):
        path, output = work / (name + ".ttir"), work / (name + ".mlir")
        path.write_text(source)
        options = "contract=true caller-contract=args_aligned64_nonalias"
        if grid:
            options += " launch-grid=" + grid
        result = subprocess.run(
            [
                opt,
                str(path),
                "--triton-to-linalg-experimental=" + options,
                *passes,
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if reject:
            assert result.returncode > 0 and reject in result.stderr, (
                name,
                result.stderr,
            )
            return output, result.stderr
        assert result.returncode == 0, (name, result.stderr)
        return output, output.read_text()

    source = (Path(__file__).parent / "Inputs/matmul-tiled-i8.ttir").read_text()
    grid = SYMBOL + ":7,6,1"
    _, text = run("specialized", source, grid)
    assert text.count("scf.if") == 1 and "arith.minsi" not in text
    assert text.count("linalg.matmul") == 2
    assert "bosc_ame." not in text and "plan." not in text
    assert "arith.cmpi slt, %arg6, %c6_i32" in text
    full, tail = text.split("} else {")
    assert "contract.fill" not in full
    assert tail.count("contract.fill") == 2
    assert full.count('kind = "none"') == 4
    assert tail.count('kind = "none"') == 1  # B remains full in both branches.
    assert tail.count("valid_shape = array<i64: 4, 128>") == 3
    assert text.count("cast_signed") == 2
    assert (
        text.count('kind = "fill"') == 2
    )  # Only tail A/C padding remains; there is no product initialization.
    check_graph(text)

    path, buffer = run("buffer", source, grid, BUFFERIZE)
    for branch in buffer.split("} else {"):
        allocations = re.findall(
            r"memref.alloc\(\).*?: (memref<[^>]+>)", branch
        )
        assert sorted(allocations) == sorted(
            ["memref<32x128xi32>"] + ["memref<32x128xi8>", "memref<16384xi8>"]
        ), allocations
        assert "strided<[1, 128]>" in branch
    assert "memref<23936x" not in buffer and "memref<97664x" not in buffer
    # All plain local copies vanished; the four provenance copies per branch
    # include the final store of the C snapshot carried through the K loop.
    assert buffer.count("memref.copy") == buffer.count("contract.access") == 8

    backend = work / "backend.mlir"
    subprocess.run(
        [
            opt,
            str(path),
            *BACKEND,
            "-o",
            str(backend),
        ],
        check=True,
        capture_output=True,
    )
    text = backend.read_text()
    assert "memref.alloc(" not in text and "bufferization." not in text
    ops = list(walk(read_module(backend)))
    branches = next(op for op in ops if op_name(op.text) == "scf.if").blocks
    for branch in branches:
        windows = [
            op.text
            for op in walk(branch)
            if op_name(op.text) == "bosc_ame.spm.view"
        ]
        assert len(windows) == 2, windows
        assert "memref.alloca" not in "\n".join(op.text for op in walk(branch))
        assert any("memref<32x128xi32" in window for window in windows)
        assert any("memref<32x128xi8" in window for window in windows)
    for line in text.splitlines():
        if (
            "bosc_ame.tile.load_a " in line
            or "bosc_ame.tile.load_acc " in line
            or "bosc_ame.tile.store_acc " in line
        ):
            assert "strided<[128, 1]" in line, line
        elif "bosc_ame.tile.load_b " in line:
            assert (
                "strided<[1, 768]" in line and "bosc_ame.direct_ddr" in line
            ), line
    lowered = work / "hardware.mlir"
    subprocess.run(
        [
            opt,
            str(backend),
            "--lower-bosc-ame",
            "--canonicalize",
            "--cse",
            "-o",
            str(lowered),
        ],
        check=True,
        capture_output=True,
    )
    hardware = lowered.read_text()
    constants = dict(
        re.findall(r"(%[\w.]+) = llvm.mlir.constant\((\d+) : i64\)", hardware)
    )
    pitches = re.findall(
        r'"bosc_ame.intr.(mlae8|mlbe8|mlce32|msce32).m"\(([^)]+)\)', hardware
    )
    assert len(pitches) == 24
    for name, operands in pitches:
        assert constants[operands.split(",")[-1].strip()] == (
            "768" if name == "mlbe8" else "128" if name == "mlae8" else "512"
        )

    run("no-grid", source, reject="kernel contract V4")
    _, text = run(
        "all-full",
        source.replace("dense<196>", "dense<192>"),
        SYMBOL + ":6,6,1",
    )
    assert (
        "scf.if" not in text
        and "arith.minsi" not in text
        and "contract.fill" not in text
    )
    for label, contract in [
        ("wide-tail", ":8,6,1"),
        ("overflow-end", ":67108864,6,1"),
        ("overflow", ":67108865,6,1"),
    ]:
        run(label, source, SYMBOL + contract, reject="kernel contract V4")
    unknown = source.replace(
        "%arg2: !tt.ptr<i32>)", "%arg2: !tt.ptr<i32>, %unknown: i32)"
    ).replace("%1 = arith.muli %0,", "%1 = arith.muli %unknown,")
    run("unproved", unknown, grid, reject="kernel contract V4")
    _, text = run(
        "nonunit-loop",
        source.replace(
            "%c1_i32 = arith.constant 1", "%c1_i32 = arith.constant 2"
        ),
        grid,
    )
    assert text.count("scf.if") == 1 and "arith.minsi" not in text
    body = source[
        source.index("module {") + len("module {") : source.rfind("}")
    ]
    _, text = run(
        "two-functions",
        "module {\n" + body + body.replace(SYMBOL, "second") + "}\n",
        grid + ";second:7,6,1",
    )
    check_graph(text, functions=2)

    path, text = run("pitched", copy_source(), "pitched:2,3,1", BUFFERIZE)
    assert text.count("scf.if") == 3 and "arith.minsi" not in text
    assert "strided<[128, 1]" in text and "strided<[160, 1]" in text
    assert "memref<32x32xi32>" in text
    loops = work / "pitched.loops.mlir"
    subprocess.run(
        [
            opt,
            str(path),
            *BACKEND,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            str(loops),
        ],
        check=True,
        capture_output=True,
    )
    machine = Machine(loops, "pitched")
    a = (np.arange(16 + 36 * 128 + 16, dtype=np.int32) * 37 - 913).astype("<i4")
    c = np.full(32 + 36 * 160 + 16, -3127, dtype="<i4")
    expected = c.copy()
    for row in range(36):
        expected[32 + row * 160 : 32 + row * 160 + 80] = a[
            16 + row * 128 : 16 + row * 128 + 80
        ]
    buffers = {"%arg0": bytearray(a.tobytes()), "%arg1": bytearray(c.tobytes())}
    for x, y in reversed(list(itertools.product(range(2), range(3)))):
        machine.launch(
            buffers,
            {
                "%arg3": 3,
                "%arg2": 2,
                "%arg4": 1,
                "%arg5": x,
                "%arg6": y,
                "%arg7": 0,
            },
        )
    np.testing.assert_array_equal(
        np.frombuffer(buffers["%arg1"], dtype="<i4"), expected
    )
    assert buffers["%arg0"] == a.tobytes()
    totals = {"read": 0, "write": 0}
    for name, start, end, kind in machine.accesses:
        base, pitch = (64, 512) if name == "%arg0" else (128, 640)
        row, col = divmod(start - base, pitch)
        assert 0 <= row < 36 and 0 <= col < col + end - start <= 80 * 4
        assert kind == ("read" if name == "%arg0" else "write")
        totals[kind] += end - start
    assert totals == {"read": 36 * 80 * 4, "write": 36 * 80 * 4}

    # Non-square logical B [256,64]: its DDR pitch is 384, its compact
    # snapshot pitch is 256. Poisoned excluded K lanes distinguish a wrong
    # physical transpose or either endpoint using the other one's pitch.
    narrow = (
        Path(__file__).parent / "Inputs/legacy/k-blocked-64x64x256.ttir"
    ).read_text()
    narrow = narrow.replace(
        "%c256n = arith.constant dense<256>",
        "%c256n = arith.constant dense<384>",
    )
    narrow = narrow.replace(
        "%b = tt.load %pb3 :",
        """
      %limit = arith.constant dense<128> : tensor<256x1xi32>
      %valid = arith.cmpi slt, %k, %limit : tensor<256x1xi32>
      %mask = tt.broadcast %valid : tensor<256x1xi1> -> tensor<256x64xi1>
      %other = arith.constant dense<-2> : tensor<256x64xi8>
      %b = tt.load %pb3, %mask, %other :""",
    )
    path, text = run("nonsquare-b", narrow, passes=BUFFERIZE)
    assert "memref<16384xi8>" in text
    assert "strided<[1, 384]" in text and "strided<[1, 256]" in text
    loops = work / "nonsquare-b.loops.mlir"
    subprocess.run(
        [
            opt,
            str(path),
            *BACKEND,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
            "-o",
            str(loops),
        ],
        check=True,
        capture_output=True,
    )
    machine = Machine(loops, "k_blocked")
    rng = np.random.default_rng(715)
    a = rng.integers(-128, 128, (64, 256), dtype=np.int8)
    b = rng.integers(-128, 128, (64, 384), dtype=np.int8)
    expected_b = b[:, :256].copy()
    expected_b[:, 128:] = -2
    expected = (a.astype(np.int64) @ expected_b.astype(np.int64).T).astype(
        "<i4"
    )
    guard = bytes([0xA7]) * 1024
    buffers = {
        "%arg0": bytearray(a.tobytes() + guard),
        "%arg1": bytearray(b.tobytes() + guard),
        "%arg2": bytearray(64 * 64 * 4) + guard,
    }
    machine.launch(buffers)
    np.testing.assert_array_equal(
        np.frombuffer(buffers["%arg2"], dtype="<i4", count=4096).reshape(
            64, 64
        ),
        expected,
    )
    assert (
        buffers["%arg0"] == a.tobytes() + guard
        and buffers["%arg1"] == b.tobytes() + guard
    )
    assert buffers["%arg2"][-len(guard) :] == guard
    reads = [
        (start, end)
        for name, start, end, kind in machine.accesses
        if name == "%arg1" and kind == "read"
    ]
    assert sum(end - start for start, end in reads) == 64 * 128
    assert all(
        0 <= start // 384 < 64
        and start % 384 < start % 384 + end - start <= 128
        for start, end in reads
    )
    print(
        "PASS tile specialization: proofs, unique graph IDs, compact reuse, parent pitches and offset DMA"
    )


if __name__ == "__main__":
    main()
