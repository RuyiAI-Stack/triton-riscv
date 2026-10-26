# RUN: %PYTHON %s %t contract-pipeline-opt
"""Prove K-loop partitions without changing tensor recurrence or f32 order."""

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
from boscame_sim import op_name, result_name  # noqa: E402

BLOCK = 4


def source(m, n, k, *, multiple=False, unsafe=None):
    lower, upper = 0, (k + BLOCK - 1) // BLOCK
    argument = ""
    if unsafe == "negative":
        lower = -1
    elif unsafe == "overflow":
        lower, upper = 536870911, 536870913
    elif unsafe == "dynamic":
        argument = ", %limit: i32"
    bound = "%limit" if unsafe == "dynamic" else "%upper"
    loop_result = "%result:2" if multiple else "%result"
    recurrence = ", %other_acc = %second" if multiple else ""
    loop_types = "tensor<4x4xf32>" + (", tensor<4x4xf32>" if multiple else "")
    extra_body = (
        "%other_next = arith.addf %other_acc, %next : tensor<4x4xf32>"
        if multiple
        else ""
    )
    yielded = "%next, %other_next" if multiple else "%next"
    final = (
        "%final = arith.addf %result#0, %result#1 : tensor<4x4xf32>"
        if multiple
        else ""
    )
    output = "%final" if multiple else "%result"
    loop = "scf.for unsigned" if unsafe == "unsigned" else "scf.for"
    return f"""module {{
  tt.func public @k_tail(%A: !tt.ptr<f32>, %B: !tt.ptr<f32>,
      %C: !tt.ptr<f32>{argument}) {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %tile = arith.constant 4 : i32
    %lower = arith.constant {lower} : i32
    %upper = arith.constant {upper} : i32
    %m = arith.constant dense<{m}> : tensor<4x1xi32>
    %n = arith.constant dense<{n}> : tensor<1x4xi32>
    %n_stride = arith.constant dense<{n}> : tensor<4x1xi32>
    %k_rows = arith.constant dense<{k}> : tensor<4x1xi32>
    %k_cols = arith.constant dense<{k}> : tensor<1x4xi32>
    %other = arith.constant dense<0.0> : tensor<4x4xf32>
    %initial = arith.constant dense<1.25> : tensor<4x4xf32>
    %second = arith.constant dense<2.5> : tensor<4x4xf32>
    %range = tt.make_range {{start = 0 : i32, end = 4 : i32}} : tensor<4xi32>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %row_base = arith.muli %pid_m, %tile : i32
    %col_base = arith.muli %pid_n, %tile : i32
    %row_splat = tt.splat %row_base : i32 -> tensor<4xi32>
    %col_splat = tt.splat %col_base : i32 -> tensor<4xi32>
    %row_vector = arith.addi %row_splat, %range : tensor<4xi32>
    %col_vector = arith.addi %col_splat, %range : tensor<4xi32>
    %rows = tt.expand_dims %row_vector {{axis = 1 : i32}} : tensor<4xi32> -> tensor<4x1xi32>
    %cols = tt.expand_dims %col_vector {{axis = 0 : i32}} : tensor<4xi32> -> tensor<1x4xi32>
    %valid_rows = arith.cmpi slt, %rows, %m : tensor<4x1xi32>
    %valid_cols = arith.cmpi slt, %cols, %n : tensor<1x4xi32>
    %row_mask = tt.broadcast %valid_rows : tensor<4x1xi1> -> tensor<4x4xi1>
    %col_mask = tt.broadcast %valid_cols : tensor<1x4xi1> -> tensor<4x4xi1>
    %a_rows = arith.muli %rows, %k_rows : tensor<4x1xi32>
    %a_offsets = tt.broadcast %a_rows : tensor<4x1xi32> -> tensor<4x4xi32>
    %a_base = tt.splat %A : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %a_row_ptrs = tt.addptr %a_base, %a_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %b_base = tt.splat %B : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %b_cols = tt.broadcast %cols : tensor<1x4xi32> -> tensor<4x4xi32>
    {loop_result} = {loop} %iv = %lower to {bound} step %one
        iter_args(%acc = %initial{recurrence}) -> ({loop_types}) : i32 {{
      %k_base = arith.muli %iv, %tile : i32
      %k_splat = tt.splat %k_base : i32 -> tensor<4xi32>
      %ks = arith.addi %k_splat, %range : tensor<4xi32>
      %ka = tt.expand_dims %ks {{axis = 0 : i32}} : tensor<4xi32> -> tensor<1x4xi32>
      %kb = tt.expand_dims %ks {{axis = 1 : i32}} : tensor<4xi32> -> tensor<4x1xi32>
      %a_k = tt.broadcast %ka : tensor<1x4xi32> -> tensor<4x4xi32>
      %ap = tt.addptr %a_row_ptrs, %a_k : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %valid_ka = arith.cmpi slt, %ka, %k_cols : tensor<1x4xi32>
      %mask_ka = tt.broadcast %valid_ka : tensor<1x4xi1> -> tensor<4x4xi1>
      %mask_a = arith.andi %row_mask, %mask_ka : tensor<4x4xi1>
      %a = tt.load %ap, %mask_a, %other : tensor<4x4x!tt.ptr<f32>>
      %b_rows = arith.muli %kb, %n_stride : tensor<4x1xi32>
      %b_offsets = tt.broadcast %b_rows : tensor<4x1xi32> -> tensor<4x4xi32>
      %b_row_ptrs = tt.addptr %b_base, %b_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %bp = tt.addptr %b_row_ptrs, %b_cols : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %valid_kb = arith.cmpi slt, %kb, %k_rows : tensor<4x1xi32>
      %mask_kb = tt.broadcast %valid_kb : tensor<4x1xi1> -> tensor<4x4xi1>
      %mask_b = arith.andi %mask_kb, %col_mask : tensor<4x4xi1>
      %b = tt.load %bp, %mask_b, %other : tensor<4x4x!tt.ptr<f32>>
      %next = tt.dot %a, %b, %acc : tensor<4x4xf32> * tensor<4x4xf32> -> tensor<4x4xf32>
      {extra_body}
      scf.yield {yielded} : {loop_types}
    }}
    {final}
    %c_rows = arith.muli %rows, %n_stride : tensor<4x1xi32>
    %c_offsets = tt.broadcast %c_rows : tensor<4x1xi32> -> tensor<4x4xi32>
    %c_offset = arith.addi %c_offsets, %b_cols : tensor<4x4xi32>
    %c_base = tt.splat %C : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %cp = tt.addptr %c_base, %c_offset : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %valid = arith.andi %row_mask, %col_mask : tensor<4x4xi1>
    tt.store %cp, {output}, %valid : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def dictionaries(text, marker):
    for match in re.finditer(re.escape(marker) + r"\s*\{", text):
        begin = match.end() - 1
        depth = 0
        for end in range(begin, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                yield text[begin : end + 1]
                break


class PoisonAllocations(Machine):
    def run(self, op):
        result = super().run(op)
        if op_name(op.text) in ("memref.alloc", "memref.alloca"):
            storage = self.values[result_name(op.text)].data
            storage[:] = bytes([0xA5]) * len(storage)
        return result


def oracle(a, b, multiple):
    m, k = a.shape
    n = b.shape[1]
    total = np.full((m, n), 1.25, dtype=np.float32)
    other = np.full((m, n), 2.5, dtype=np.float32)
    with np.errstate(all="ignore"):
        for start in range(0, k, BLOCK):
            product = np.zeros((m, n), dtype=np.float32)
            for row, col in np.ndindex(m, n):
                for lane in range(BLOCK):
                    term = (
                        np.float32(a[row, start + lane] * b[start + lane, col])
                        if start + lane < k
                        else np.float32(0)
                    )
                    product[row, col] = np.float32(product[row, col] + term)
            total = np.add(total, product, dtype=np.float32)
            if multiple:
                other = np.add(other, total, dtype=np.float32)
        return np.add(total, other, dtype=np.float32) if multiple else total


def check_events(text, m, n, k):
    rows = {BLOCK} | ({m % BLOCK} if m % BLOCK else set())
    cols = {BLOCK} | ({n % BLOCK} if n % BLOCK else set())
    ks = ({BLOCK} if k >= BLOCK else set()) | (
        {k % BLOCK} if k % BLOCK else set()
    )
    expected = (
        {(0, r, t) for r, t in itertools.product(rows, ks)}
        | {(1, t, c) for t, c in itertools.product(ks, cols)}
        | {(3, r, c) for r, c in itertools.product(rows, cols)}
    )
    actual = set()
    for event in dictionaries(text, "contract.access ="):
        origin = int(re.search(r"origin_id = (\d+)", event)[1])
        shape = re.search(r"valid_shape = array<i64: (\d+), (\d+)>", event)
        actual.add((origin, *(map(int, shape.groups()) if shape else (4, 4))))
        assert "-9223372036854775808" not in event
        if origin != 3:
            assert "other = 0.000000e+00 : f32" in event
    assert actual == expected, (actual, expected)


def check_tensor_recurrence(path, expected_loops, multiple):
    count = 0
    lines = path.read_text().splitlines()
    for position, line in enumerate(lines):
        if " = scf.for " not in line:
            continue
        count += 1
        name, _, arity = result_name(line.strip()).partition(":")
        assert int(arity or "1") == (2 if multiple else 1), line
        results = [f"{name}#{i}" for i in range(2)] if multiple else [name]
        indent = len(line) - len(line.lstrip())
        following = []
        for user in lines[position + 1 :]:
            if not user.strip():
                continue
            level = len(user) - len(user.lstrip())
            if level < indent:
                break
            if level == indent:
                following.append(user)
        # Every complete-loop result feeds the corresponding tail expression
        # in the same block, without reinitialization.
        for result in results:
            assert any(
                "linalg.generic" in user
                and re.search(
                    r"ins\([^)]*" + re.escape(result) + r"(?=[, :])", user
                )
                for user in following
            ), (line, result)
    assert count == expected_loops, (count, expected_loops)


def numerics(path, text, m, n, k, multiple):
    rng = np.random.default_rng(7381 + k)
    a = rng.integers(-3, 4, (m, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, n)).astype(np.float32)
    a[0] = np.resize(np.array([1e20, 1, -1e20, 1], dtype=np.float32), k)
    b[:, 0] = 1
    expected = oracle(a, b, multiple)
    guard = bytes([0xD7]) * 64
    grid = ((m + 3) // 4, (n + 3) // 4, 1)
    programs = list(itertools.product(*(range(d) for d in grid)))
    arguments = re.findall(
        r'(%arg\d+): i32 \{contract.program = \{axis = (\d) : i32, role = "([^"]+)"\}',
        text,
    )
    assert len(arguments) == 6
    for order in (programs, list(reversed(programs))):
        machine = PoisonAllocations(path, "k_tail")
        buffers = {
            "%arg0": bytearray(a.tobytes() + guard),
            "%arg1": bytearray(b.tobytes() + guard),
            "%arg2": bytearray(bytes([0x93]) * (m * n * 4) + guard),
        }
        for pid in order:
            machine.launch(
                buffers,
                {
                    name: grid[int(axis)]
                    if role == "num_programs"
                    else pid[int(axis)]
                    for name, axis, role in arguments
                },
            )
        assert buffers["%arg2"][: m * n * 4] == expected.tobytes()
        assert buffers["%arg0"] == a.tobytes() + guard
        assert buffers["%arg1"] == b.tobytes() + guard
        assert buffers["%arg2"][m * n * 4 :] == guard
        totals = {"%arg0": 0, "%arg1": 0, "%arg2": 0}
        for name, start, end, kind in machine.accesses:
            count, pitch = (
                (m, k)
                if name == "%arg0"
                else ((k, n) if name == "%arg1" else (m, n))
            )
            row, col = divmod(start, pitch * 4)
            assert (
                0 <= row < count and 0 <= col < col + end - start <= pitch * 4
            )
            assert kind == ("write" if name == "%arg2" else "read")
            totals[name] += end - start
        assert totals == {
            "%arg0": m * k * 4 * grid[1],
            "%arg1": k * n * 4 * grid[0],
            "%arg2": m * n * 4,
        }, totals


def main():
    work, opt = Path(sys.argv[1]), sys.argv[2]
    work.mkdir(parents=True, exist_ok=True)

    def run(input_, *passes):
        result = subprocess.run(
            [opt, *passes], input=input_, text=True, capture_output=True
        )
        assert result.returncode >= 0 and "PLEASE submit" not in result.stderr
        return result

    for label, m, n, k, multiple in (
        ("k-m-n-tail", 5, 6, 11, False),
        ("divisible", 5, 4, 8, False),
        ("short", 5, 4, 3, False),
        ("only-k-tail", 4, 4, 11, False),
        ("multiple-carried", 5, 4, 11, True),
    ):
        input_ = source(m, n, k, multiple=multiple)
        (work / (label + ".ttir")).write_text(input_)
        result = run(
            input_,
            "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias "
            f"launch-grid=k_tail:{(m + 3) // 4},{(n + 3) // 4},1",
            "--kernel-contract-verify",
            "--mlir-print-ir-before=canonicalize",
            "--mlir-disable-threading",
        )
        (work / (label + ".log")).write_text(result.stderr)
        assert result.returncode == 0, (label, result.stderr[-4000:])
        text = result.stdout
        (work / (label + ".mlir")).write_text(text)
        if multiple:
            # The partition preserves arbitrary SSA iter_args; the current
            # content graph still models only one private buffer slot.
            assert "contract.compute_graph = {complete = false" in text
            assert "one private buffer-carried slot" in text
        else:
            assert "contract.compute_graph = {complete = true" in text
        assert "bosc_ame." not in text and "plan." not in text
        assert "arith.minsi" not in text and "arith.maxsi" not in text
        assert "arith.addf" in text and "math.fma" not in text
        ids = [
            int(re.search(r"\bid = (\d+)", task)[1])
            for task in dictionaries(text, "contract.task =")
        ]
        assert sorted(ids) == list(range(len(ids))), ids
        branches = (2 if m % 4 else 1) * (2 if n % 4 else 1)
        assert text.count("linalg.matmul") == branches * (
            2 if k > 4 and k % 4 else 1
        )
        check_events(text, m, n, k)
        if k >= 4 and not multiple:
            assert text.count("contract.loop_carried_slots") == branches
        tensor_dumps = [
            dump
            for dump in re.split(
                r"// -----// IR Dump Before[^\n]*\n", result.stderr
            )
            if "bufferization.to_tensor" in dump
            and "linalg.matmul" in dump
            and "tensor<4x4xf32>" in dump
        ]
        assert tensor_dumps, (label, result.stderr)
        tensor = tensor_dumps[-1]
        (work / (label + ".tensor.mlir")).write_text(tensor)
        if k > 4:
            assert "iter_args" in tensor
            if multiple:
                assert "-> (tensor<4x4xf32>, tensor<4x4xf32>)" in tensor
        for line in tensor.splitlines():
            if "linalg.matmul" in line:
                assert line.count("tensor<4x4xf32>") >= 3, line
        if k > 4 and k % 4:
            check_tensor_recurrence(
                work / (label + ".tensor.mlir"), branches, multiple
            )
        lowered = run(
            text,
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--canonicalize",
        )
        assert lowered.returncode == 0, lowered.stderr
        path = work / (label + ".loops.mlir")
        path.write_text(lowered.stdout)
        numerics(path, text, m, n, k, multiple)

    for unsafe in ("negative", "overflow", "dynamic", "unsigned"):
        result = run(
            source(4, 4, 11, unsafe=unsafe),
            "--triton-to-linalg-experimental=contract=true caller-contract=args_aligned64_nonalias launch-grid=k_tail:1,1,1",
            "--mlir-print-ir-after-failure",
        )
        assert (
            result.returncode != 0 and "kernel contract V4" in result.stderr
        ), result.stderr
        dump = result.stderr.split("IR Dump After", 1)[-1]
        assert dump.count("linalg.matmul") == 1 and dump.count("scf.for") == 1
    print(
        "PASS: K full/tail proofs, tensor recurrence, graph rebuild and ordered f32 masked accesses"
    )


if __name__ == "__main__":
    main()
