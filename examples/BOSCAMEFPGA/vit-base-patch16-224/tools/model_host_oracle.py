#!/usr/bin/env python3
"""Python side of the host whole-model numerical check.

Model adapters call typed per-case entries; the host library built by
tools/build_model_host.py forwards each call here.  Every call is evaluated
with the independent per-case reference in oracles.py (the same semantics used
by the strict simulator), which lets the real rewritten graph, W8A8 parameter
packing and mutable-workspace ABI run on x86 without an FPGA.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parents[1]
REPO = V.parents[2]
os.environ.setdefault("BUDDY_ROOT", str(REPO / "buddy-mlir"))
for path in (V, V / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cases  # noqa: E402
import oracles  # noqa: E402
import export  # noqa: E402

NP_DTYPES = {"f32": np.float32, "i8": np.int8, "i32": np.int32}

_DESCRIPTOR_CACHE = {}


def descriptor_type(rank):
    if rank not in _DESCRIPTOR_CACHE:

        class Descriptor(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.c_void_p),
                ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]

        _DESCRIPTOR_CACHE[rank] = Descriptor
    return _DESCRIPTOR_CACHE[rank]


def memref_array(pointer, shape, dtype):
    """Build a writable numpy view over one typed MemRef descriptor."""
    rank = len(shape)
    descriptor = ctypes.cast(pointer, ctypes.POINTER(descriptor_type(rank))).contents
    if not descriptor.aligned:
        raise RuntimeError("null aligned pointer for a kernel argument")
    itemsize = np.dtype(dtype).itemsize
    strides = [int(descriptor.strides[i]) for i in range(rank)]
    span = sum((int(shape[i]) - 1) * strides[i] for i in range(rank)) + 1
    needed = (int(descriptor.offset) + span) * itemsize
    buffer = (ctypes.c_char * needed).from_address(int(descriptor.aligned))
    return np.ndarray(
        tuple(int(x) for x in shape),
        dtype=np.dtype(dtype),
        buffer=buffer,
        offset=int(descriptor.offset) * itemsize,
        strides=tuple(stride * itemsize for stride in strides),
    )


class HostOracleDispatcher:
    CALLBACK = ctypes.CFUNCTYPE(
        None, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int
    )

    def __init__(self, replacement=None, host_library=None, echo=False):
        replacement = Path(replacement or (V / "build/replacement-w8a8"))
        if host_library is None:
            raise RuntimeError("host_library is required")
        self.library_path = Path(host_library)
        self.echo = echo
        records = json.loads((self.library_path.parent / "host-cases.json").read_text())
        self.records = records
        self.meta = {}
        for record in records:
            description = cases.describe(record["case"])
            shapes = [
                tuple(int(x) for x in shape)
                for shape in export.argument_shapes(description)
            ]
            if len(shapes) != len(record["ranks"]):
                raise RuntimeError("shape/rank mismatch for %s" % record["case"])
            names = [argument["name"] for argument in description["arguments"]]
            dtypes = [
                NP_DTYPES[argument["dtype"]] for argument in description["arguments"]
            ]
            self.meta[record["index"]] = {
                "case": description,
                "shapes": shapes,
                "names": names,
                "dtypes": dtypes,
            }
        self.calls = {}
        self.errors = []
        self.library = ctypes.CDLL(str(self.library_path))
        self.library.host_model_bind.argtypes = [self.CALLBACK]
        self.library.host_model_bind.restype = None
        self._callback = self.CALLBACK(self._dispatch)
        self.library.host_model_bind(self._callback)

    # ------------------------------------------------------------------
    def _dispatch(self, index, arguments, count):
        try:
            meta = self.meta[index]
            if count != len(meta["shapes"]):
                raise RuntimeError(
                    "argument count mismatch for %s" % meta["case"]["name"]
                )
            arrays = {}
            for i, (name, shape, dtype) in enumerate(
                zip(meta["names"], meta["shapes"], meta["dtypes"])
            ):
                arrays[name] = memref_array(arguments[i], shape, dtype)
            if meta["case"]["family"] == "patch_embed":
                outputs = self._patch_embed(arrays)
            else:
                outputs = oracles.reference(meta["case"], arrays)
            for name, value in outputs.items():
                np.copyto(
                    arrays[name],
                    np.asarray(value, dtype=arrays[name].dtype),
                    casting="unsafe",
                )
            self.calls[meta["case"]["name"]] = (
                self.calls.get(meta["case"]["name"], 0) + 1
            )
        except Exception:
            self.errors.append(traceback.format_exc())
            if self.echo:
                print("HOST ORACLE ERROR:", file=sys.stderr)
                print(self.errors[-1], file=sys.stderr)

    @staticmethod
    def _patch_embed(arrays):
        """Reproduce the graph-facing patch adapter: im2col, W8A8 projection,
        per-row/token dequantization, bias add and NCHW transpose."""
        image = np.ascontiguousarray(arrays["X"]).reshape(3, 224, 224)
        weight = np.ascontiguousarray(arrays["Weight"]).reshape(768, 768)
        bias = np.ascontiguousarray(arrays["Bias"]).reshape(768)
        patches = np.empty((196, 768), dtype=np.float32)
        for row in range(14):
            for col in range(14):
                block = image[:, row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16]
                patches[row * 14 + col] = block.reshape(-1)
        quantized, row_scale = oracles.quantize_rows(patches)
        weight_q, weight_scale = oracles.quantize_rows(weight)
        accumulator = (quantized.astype(np.int64) @ weight_q.astype(np.int64).T).astype(
            np.int32
        )
        value = (accumulator.astype(np.float32) * row_scale[:, None]) * weight_scale[
            None, :
        ]
        value = np.add(value, bias[None, :], dtype=np.float32)
        out = arrays["Out"]
        out[0] = value.T.reshape(768, 14, 14)
        return {"Out": out}

    def summary(self):
        total = sum(self.calls.values())
        return {
            "calls": total,
            "by_case": dict(sorted(self.calls.items())),
            "errors": self.errors[:3],
        }
