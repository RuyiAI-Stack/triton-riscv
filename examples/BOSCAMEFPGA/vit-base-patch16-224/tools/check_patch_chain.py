#!/usr/bin/env python3
"""Execute the existing patch adapter, with strict emitted-IR child kernels.

The unchanged generated C adapter runs natively, including its original bias
and NCHW copy. Its raw im2col and four numerical dependencies execute in the
strict simulator through descriptor callbacks. This is a mixed native-C/IR
operator check, not target assembly execution or a model test.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from export import ROOT, argument_shapes, describe, digest, tool
from oracles import DTYPES, output_names, quantize_rows
from simcheck import Machine, SimulationBlocked, check_artifacts, simulate_case

NAME = "patch_embed_196x768"
DEPENDENCIES = (
    "quantize_196x768",
    "quantize_768x768",
    "matmul_i8_196x768x768",
    "dequantize_196x768",
)


def descriptor(rank):
    class MemRef(ct.Structure):
        _fields_ = [
            ("allocated", ct.c_void_p),
            ("aligned", ct.c_void_p),
            ("offset", ct.c_int64),
            ("sizes", ct.c_int64 * rank),
            ("strides", ct.c_int64 * rank),
        ]

    return MemRef


MEMREF = {rank: descriptor(rank) for rank in (1, 2, 4)}


class Flat(ct.Structure):
    _fields_ = [
        ("allocated", ct.c_void_p),
        ("aligned", ct.c_void_p),
        ("offset", ct.c_int64),
        ("size", ct.c_int64),
        ("stride", ct.c_int64),
    ]


class Unranked(ct.Structure):
    _fields_ = [("rank", ct.c_int64), ("descriptor", ct.c_void_p)]


RAW = ct.CFUNCTYPE(None, *([ct.c_void_p] * 4 + [ct.c_int32] * 6))
STAGE = ct.CFUNCTYPE(None, ct.c_int, *([ct.c_void_p] * 4))
SHIM = r"""
static void (*raw_callback)(Unranked*,Unranked*,Unranked*,Unranked*,int,int,int,int,int,int);
static void (*stage_callback)(int,void*,void*,void*,void*);
void vit_patch_callbacks(void *raw,void *stage) { raw_callback=raw; stage_callback=stage; }
void _mlir_ciface_triton_patch_embed_196x768(Unranked*a,Unranked*b,Unranked*c,Unranked*d,int gx,int gy,int gz,int x,int y,int z) { raw_callback(a,b,c,d,gx,gy,gz,x,y,z); }
void _mlir_ciface_kernel_quantize_196x768(MemRef2*a,MemRef2*b,MemRef1*c) { stage_callback(0,a,b,c,0); }
void _mlir_ciface_kernel_quantize_768x768(MemRef2*a,MemRef2*b,MemRef1*c) { stage_callback(1,a,b,c,0); }
void _mlir_ciface_kernel_matmul_i8_196x768x768(MemRef2*a,MemRef2*b,MemRef2*c) { stage_callback(2,a,b,c,0); }
void _mlir_ciface_kernel_dequantize_196x768(MemRef2*a,MemRef1*b,MemRef1*c,MemRef2*d) { stage_callback(3,a,b,c,d); }
void vit_patch_poison(void) {
  void *p[]={patch_im2col,patch_im2col_q,patch_im2col_scale,patch_weight_q,patch_weight_scale,patch_acc,patch_gemm};
  const size_t n[]={sizeof(patch_im2col),sizeof(patch_im2col_q),sizeof(patch_im2col_scale),sizeof(patch_weight_q),sizeof(patch_weight_scale),sizeof(patch_acc),sizeof(patch_gemm)};
  for(int i=0;i<7;++i) for(size_t j=0;j<n[i];++j) ((unsigned char*)p[i])[j]=0x53;
}
"""


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def patches(image):
    # Independent NCHW patch indexing, never candidate im2col values.
    return np.array(
        [
            image[0, :, y : y + 16, x : x + 16].reshape(-1)
            for y in range(0, 224, 16)
            for x in range(0, 224, 16)
        ],
        np.float32,
    )


def fixtures(stress):
    rng = np.random.default_rng(107 if stress else 71)
    image = rng.normal(0, 0.1, (1, 3, 224, 224)).astype(np.float32)
    weight = rng.normal(0, 0.05, (768, 3, 16, 16)).astype(np.float32)
    bias = rng.normal(0, 0.02, (768,)).astype(np.float32)
    if stress:
        values = np.array([-127, -2.5, -0.5, -0.0, 0.0, 0.5, 2.5, 127], np.float32)
        image[:] = np.resize(values, image.shape) * np.float32(2**-8)
        weight[:] = np.resize(values[::-1], weight.shape) * np.float32(2**-10)
        weight[0] = 0
        image[:, :, :16, :16] = 0
    return [image, weight, bias, np.full((1, 768, 14, 14), -9132.0, np.float32)]


def check_patch_case(case, kernels):
    record = dict(
        schema=1,
        name=NAME,
        status="BLOCKED",
        programs=0,
        total_programs=0,
        outputs=0,
        runs=[],
        full_model_executed=False,
        execution="native generated C adapter with strict emitted-IR numerical children",
    )
    try:
        modules = {
            n: check_artifacts(describe(n), kernels) for n in (NAME, *DEPENDENCIES)
        }
        hashes = {n: digest(p) for n, p in modules.items()}
        adapter = Path(kernels) / NAME / "adapter.c"
        record.update(
            module_sha256=hashes[NAME],
            dependency_sha256={n: hashes[n] for n in DEPENDENCIES},
            kernel_module_sha256=digest(case["kernel_source"]),
            adapter_sha256=digest(adapter),
        )
        with tempfile.TemporaryDirectory(prefix="vit-patch-chain-") as temporary:
            temp = Path(temporary)
            loops = temp / "im2col.mlir"
            subprocess.run(
                [
                    tool("buddy-opt"),
                    str(modules[NAME]),
                    "--convert-linalg-to-loops",
                    "--lower-affine",
                    "--canonicalize",
                    "-o",
                    str(loops),
                ],
                check=True,
                capture_output=True,
            )
            machine = Machine(loops, case["symbol"])
            for bank in machine.spm:
                bank[:] = bytes([0xD3]) * len(bank)
            source = temp / "adapter.c"
            source.write_text(adapter.read_text() + SHIM)
            library = temp / "adapter.so"
            command = [
                "clang",
                "-shared",
                "-fPIC",
                "-O2",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-I" + str(ROOT),
                str(source),
                "-o",
                str(library),
            ]
            subprocess.run(command, check=True, capture_output=True)
            record["adapter_compile_command"] = command
            lib = ct.CDLL(str(library))
            lib.vit_patch_callbacks.argtypes = [RAW, STAGE]
            entry = getattr(lib, "_mlir_ciface_kernel_" + NAME)
            entry.argtypes = [ct.c_void_p] * 4
            machines = {}
            for index, order in enumerate(("forward", "tail-first", "reverse")):
                arrays = fixtures(index != 0)
                storage, descriptors, pointers = [], [], []
                for value in arrays:
                    allocation = ct.create_string_buffer(value.nbytes + 192)
                    address = (ct.addressof(allocation) + 127) & ~63
                    ct.memset(address - 64, 0xA5, 64)
                    ct.memset(address + value.nbytes, 0x5A, 64)
                    ct.memmove(address, value.ctypes.data, value.nbytes)
                    rank = value.ndim
                    strides = tuple(s // value.itemsize for s in value.strides)
                    descriptors.append(
                        MEMREF[rank](address, address, 0, value.shape, strides)
                    )
                    pointers.append(address)
                    storage.append(allocation)
                state = dict(
                    raw_calls=0, stage_calls=0, errors=[], stages=[], buffers=None
                )
                guard = bytes([0xC7]) * 1024

                @RAW
                def raw(*args):
                    if state["errors"]:
                        return
                    try:
                        point = tuple(args[7:])
                        assert tuple(args[4:7]) == (196, 1, 1)
                        assert point == (state["raw_calls"], 0, 0)
                        flat = [
                            ct.cast(
                                ct.cast(p, ct.POINTER(Unranked)).contents.descriptor,
                                ct.POINTER(Flat),
                            ).contents
                            for p in args[:4]
                        ]
                        sizes = [150528, 589824, 768, 150528]
                        assert all(
                            d.size == n and d.stride == 1 and d.offset == 0
                            for d, n in zip(flat, sizes)
                        )
                        if state["buffers"] is None:
                            state["buffers"] = {
                                f"%arg{i}": bytearray(
                                    ct.string_at(d.aligned, n * 4) + guard
                                )
                                for i, (d, n) in enumerate(zip(flat, sizes))
                            }
                            state["original"] = {
                                k: bytes(v) for k, v in state["buffers"].items()
                            }
                            state["access_start"] = len(machine.accesses)
                        machine.launch(
                            state["buffers"],
                            {
                                f"%arg{4 + i}": v
                                for i, v in enumerate([196, 1, 1, *point])
                            },
                        )
                        state["raw_calls"] += 1
                        if state["raw_calls"] == 196:
                            for i, n in enumerate(sizes):
                                data = state["buffers"][f"%arg{i}"]
                                assert data[-1024:] == guard
                                if i < 3:
                                    assert bytes(data) == state["original"][f"%arg{i}"]
                            for name, begin, end, action in machine.accesses[
                                state["access_start"] :
                            ]:
                                i = int(name[4:])
                                assert 0 <= begin < end <= sizes[i] * 4
                                assert action != "write" or i == 3
                            actual = np.frombuffer(
                                state["buffers"]["%arg3"], np.float32, 150528
                            ).reshape(196, 768)
                            np.testing.assert_array_equal(
                                actual.view(np.uint32),
                                patches(arrays[0]).view(np.uint32),
                            )
                            ct.memmove(
                                flat[3].aligned, actual.ctypes.data, actual.nbytes
                            )
                    except Exception as error:
                        state["errors"].append(repr(error))

                @STAGE
                def stage(which, *args):
                    if state["errors"]:
                        return
                    try:
                        assert (
                            state["raw_calls"] == 196 and which == state["stage_calls"]
                        )
                        dep = describe(DEPENDENCIES[which])
                        values, addresses = {}, {}
                        for arg, shape, pointer in zip(
                            dep["arguments"], argument_shapes(dep), args
                        ):
                            d = ct.cast(
                                pointer, ct.POINTER(MEMREF[arg["rank"]])
                            ).contents
                            assert tuple(d.sizes) == tuple(shape) and d.offset == 0
                            expected_strides = tuple(
                                int(np.prod(shape[i + 1 :])) for i in range(len(shape))
                            )
                            assert (
                                tuple(d.strides) == expected_strides
                                and d.aligned % 64 == 0
                            )
                            dtype = DTYPES[arg["dtype"]]
                            address = d.aligned
                            values[arg["name"]] = (
                                np.frombuffer(
                                    ct.string_at(
                                        address, int(np.prod(shape)) * dtype.itemsize
                                    ),
                                    dtype,
                                )
                                .copy()
                                .reshape(shape)
                            )
                            addresses[arg["name"]] = address
                        actual, evidence = simulate_case(
                            dep,
                            values,
                            kernels,
                            order=order,
                            progress=False,
                            machines=machines,
                        )
                        state["stages"].append(dict(case=dep["name"], **evidence))
                        for name in output_names(dep):
                            ct.memmove(
                                addresses[name],
                                actual[name].ctypes.data,
                                actual[name].nbytes,
                            )
                        state["stage_calls"] += 1
                    except Exception as error:
                        state["errors"].append(repr(error))

                lib.vit_patch_callbacks(raw, stage)
                if index == 0:
                    lib.vit_patch_poison()
                entry(*(ct.byref(d) for d in descriptors))
                if state["errors"]:
                    raise AssertionError(state["errors"])
                assert state["raw_calls"] == 196 and state["stage_calls"] == 4
                for value, address in zip(arrays, pointers):
                    assert ct.string_at(address - 64, 64) == bytes([0xA5]) * 64
                    assert (
                        ct.string_at(address + value.nbytes, 64) == bytes([0x5A]) * 64
                    )
                for value, address in zip(arrays[:3], pointers[:3]):
                    assert ct.string_at(address, value.nbytes) == value.tobytes()
                actual = (
                    np.frombuffer(
                        ct.string_at(pointers[3], arrays[3].nbytes), np.float32
                    )
                    .copy()
                    .reshape(1, 768, 14, 14)
                )
                aq, rs = quantize_rows(patches(arrays[0]))
                bq, cs = quantize_rows(arrays[1].reshape(768, 768))
                acc = (aq.astype(np.int64) @ bq.astype(np.int64).T).astype(np.int32)
                restored = (acc.astype(np.float32) * rs[:, None]) * cs[None, :]
                expected = (restored + arrays[2][None, :]).T.reshape(1, 768, 14, 14)
                np.testing.assert_array_equal(
                    actual.view(np.uint32), expected.view(np.uint32)
                )
                programs = 196 + sum(s["programs"] for s in state["stages"])
                record["runs"].append(
                    dict(
                        name="normal" if not index else "stress",
                        order=order,
                        programs=programs,
                        outputs=actual.size,
                        stages=state["stages"],
                        im2col_programs=196,
                        im2col_order="forward (actual adapter dispatch); child order only is varied",
                        output_sha256={"Out": sha(actual)},
                        reference_sha256={"Out": sha(expected)},
                        bit_exact_quantized_oracle=True,
                        guard_errors=0,
                        input_errors=0,
                    )
                )
                record["programs"] += programs
                record["outputs"] += actual.size
                print(f"{NAME}: round {index + 1}/3 PASS", flush=True)
            assert (
                record["runs"][1]["output_sha256"] == record["runs"][2]["output_sha256"]
            )
            assert hashes == {
                n: digest(check_artifacts(describe(n), kernels)) for n in modules
            }
            record.update(
                status="PASS",
                total_programs=record["programs"],
                comparison_rounds=3,
                max_abs=0.0,
            )
    except Exception as error:
        record.update(
            status="BLOCKED" if isinstance(error, SimulationBlocked) else "FAIL",
            reason=repr(error),
        )
    directory = Path(kernels) / NAME
    (directory / "simcheck-result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def collect_patch_case(case, kernels):
    record = json.loads((Path(kernels) / NAME / "simcheck-result.json").read_text())
    if record.get("status") != "PASS":
        return record
    assert record["name"] == NAME and record["module_sha256"] == digest(
        check_artifacts(case, kernels)
    )
    assert record["kernel_module_sha256"] == digest(case["kernel_source"])
    assert record["adapter_sha256"] == digest(Path(kernels) / NAME / "adapter.c")
    assert record["dependency_sha256"] == {
        n: digest(check_artifacts(describe(n), kernels)) for n in DEPENDENCIES
    }
    assert len(record["runs"]) == 3
    for run, order in zip(record["runs"], ("forward", "tail-first", "reverse")):
        assert (
            run["order"] == order
            and run["outputs"] == 150528
            and run["im2col_programs"] == 196
        )
        assert run.get("guard_errors") == run.get("input_errors") == 0
        assert run["output_sha256"] == run["reference_sha256"] and set(
            run["output_sha256"]
        ) == {"Out"}
        assert (
            len(run["output_sha256"]["Out"]) == 64 and run["bit_exact_quantized_oracle"]
        )
        assert [s["case"] for s in run["stages"]] == list(DEPENDENCIES)
        for stage, name in zip(run["stages"], DEPENDENCIES):
            assert stage["module_sha256"] == record["dependency_sha256"][name]
            assert stage["kernel_module_sha256"] == digest(
                describe(name)["kernel_source"]
            )
            assert (
                stage["programs"]
                == stage["total_programs"]
                == int(np.prod(describe(name)["grid"]))
            )
            shapes = argument_shapes(describe(name))
            expected_outputs = sum(
                int(np.prod(shape))
                for arg, shape in zip(describe(name)["arguments"], shapes)
                if arg["name"] in output_names(describe(name))
            )
            assert stage["outputs"] == expected_outputs
            if name.startswith("matmul_i8"):
                assert stage["argument_bytes"]["C"]["read"] == 0
        assert run["programs"] == 196 + sum(s["programs"] for s in run["stages"])
    assert record["outputs"] == 451584 and record["programs"] == record[
        "total_programs"
    ] == sum(r["programs"] for r in record["runs"])
    assert record["runs"][1]["output_sha256"] == record["runs"][2]["output_sha256"]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "build/patch-chain-check.json"
    )
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    record = (collect_patch_case if args.collect else check_patch_case)(
        describe(NAME), args.kernels
    )
    args.report.write_text(
        json.dumps(
            {"schema": 1, "status": record["status"], "cases": [record]}, indent=2
        )
        + "\n"
    )
    print(record["status"], record.get("reason", ""), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
