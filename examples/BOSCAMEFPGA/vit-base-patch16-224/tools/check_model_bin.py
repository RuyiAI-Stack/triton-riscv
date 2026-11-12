#!/usr/bin/env python3
"""Independently verify the whole-model ViT-B/16 NH/E6 flat image artifacts.

The tool recomputes every expectation from the graph replacement contract
(workspace-buffers.json, subgraph0.triton.mlir, triton-call-replacement.json)
and the linker inputs, then checks the produced artifacts: vit_model.elf,
vit_model.bin, manifest.json, ARG-MAP.json, params-layout.json, params.bin,
model_launcher.c, model_workspace.S and model_blobs.S.

Checks
  1. @vit argument contract: 734 argument roots in order, split into 273
     parameters / 1 image / 460 workspaces, each signature shape+dtype matching
     the parameter/input layout.
  2. params.bin packing: exactly the 273 parameter-layout entries, 64-byte
     aligned offsets, no overlaps, total size equals the payload sum plus
     padding, per-region sha256 and whole-file sha256.
  3. ARG-MAP.json + generated launcher: descriptors match the recomputed dense
     (row-major) strides, parameter/workspace classification, parameter offsets
     and sequential workspace offsets; blob/workspace assembly files.
  4. ELF structure and symbol table: ELF64 little-endian RISC-V, entries at the
     expected addresses, _mlir_ciface_vit plus all 1543 adapters and all 28
     kernel adapters defined, undefined symbols limited to the runtime
     allowlist.
  5. Flat binary: size equals the loaded PROGBITS extent from 0x80000000,
     sha256 matches the manifest, embedded parameter/input/reference blobs
     match their sources, and every NOLOAD region is absent from the image.
  6. Independent W8A8 parameter accounting: 127 used checkpoint tensors plus
     73 derived int8 weights and 73 derived scales = 273 parameters, with every
     weight_param_name/scale_param_name present in the layout.
  7. Run the Buddy NR ELF contract audit (check_nr_elf.py) on the ELF and
     record its result.

This is a structural/artifact audit only: no model execution, no board load,
no hardware run, and no claim about numerical or hardware correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

V = Path(__file__).resolve().parents[1]
REPO = V.parents[2]

IMAGE_BASE = 0x80000000
NH_ENTRY = 0x80000000
E6_ENTRY = 0x80010000
EXPECTED_VIT_ARGS = 734
EXPECTED_CIFACE_ARGS = 735
EXPECTED_PARAMS = 273
EXPECTED_WORKSPACES = 460
EXPECTED_IMAGE_ARGS = 1
EXPECTED_ADAPTERS = 1543
EXPECTED_DISTINCT_CASES = 28
EXPECTED_W8A8_LINEARS = 73
ALIGNMENT = 64
RESULT_SHAPE = [1, 1000]
RESULT_NBYTES = 4000

DTYPE_ITEMSIZE = {"f32": 4, "i8": 1, "i32": 4, "i64": 8}
LAYOUT_DTYPE = {
    "TensorDType.Float32": "f32",
    "TensorDType.Int8": "i8",
    "TensorDType.Int32": "i32",
    "TensorDType.Int64": "i64",
    "Float32": "f32",
    "Int8": "i8",
    "Int32": "i32",
    "Int64": "i64",
}
# Undefined symbols that the freestanding runtime may leave to the
# libc-like support layer.  __builtin_trap lowers to a trap family symbol.
ALLOWED_UNDEFINED = {
    "memcpy",
    "memset",
    "memmove",
    "malloc",
    "free",
    "calloc",
    "realloc",
    "memrefCopy",
    "abort",
}
TRAP_UNDEFINED_RE = re.compile(r"^(__builtin_trap|__trap|llvm\.trap|trap)")

REQUIRED_ARTIFACTS = (
    "vit_model.elf",
    "vit_model.bin",
    "manifest.json",
    "ARG-MAP.json",
    "params-layout.json",
    "params.bin",
    "model_launcher.c",
    "model_workspace.S",
    "model_blobs.S",
)

SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_NOBITS = 8
SHT_DYNSYM = 11
SHF_ALLOC = 0x2
STT_OBJECT = 1
STT_FUNC = 2

SCOPE = (
    "Structural artifact verification only. This tool does not execute the "
    "model, does not load or run anything on hardware, and makes no numerical "
    "or hardware-correctness claim; it verifies that the generated ELF/bin "
    "matches the recomputed ABI/layout/symbol expectations."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def product(values) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def dense_strides(shape) -> list:
    strides = []
    stride = 1
    for dim in reversed([int(x) for x in shape]):
        strides.insert(0, stride)
        stride *= dim
    return strides


def fmt_hex(value: int) -> str:
    return "0x%x" % value


def load_json(path: Path):
    return json.loads(Path(path).read_text())


class Check:
    """One audit check with PASS/FAIL state and collected reasons."""

    def __init__(self, cid: str, title: str):
        self.id = cid
        self.title = title
        self.reasons = []
        self.details = {}
        self.status = "PASS"

    def fail(self, reason: str) -> None:
        self.status = "FAIL"
        self.reasons.append(reason)

    def skip(self, reason: str) -> None:
        self.status = "SKIP"
        self.reasons.append(reason)

    def note(self, **details) -> None:
        self.details.update(details)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "reasons": self.reasons,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Minimal ELF64 parser (little-endian RISC-V)
# ---------------------------------------------------------------------------


class ElfFile:
    def __init__(self, path: Path):
        self.path = path
        self.error = None
        self.header = {}
        self.sections = []
        self.defined = {}
        self.undefined = set()
        self.sym_count = 0
        self.data = b""
        try:
            self.data = Path(path).read_bytes()
            self._parse()
        except Exception as error:  # pragma: no cover - defensive path
            self.error = "%s: %s" % (type(error).__name__, error)

    def _parse(self) -> None:
        data = self.data
        if len(data) < 64:
            raise ValueError("file shorter than an ELF header")
        if data[:4] != b"\x7fELF":
            raise ValueError("bad ELF magic")
        elf_class, elf_data = data[4], data[5]
        self.header = {
            "elf_class": elf_class,
            "elf_data": elf_data,
            "e_type": struct.unpack_from("<H", data, 16)[0],
            "e_machine": struct.unpack_from("<H", data, 18)[0],
            "e_entry": struct.unpack_from("<Q", data, 24)[0],
            "e_flags": struct.unpack_from("<I", data, 48)[0],
        }
        if elf_class != 2:
            raise ValueError("not an ELF64 file (EI_CLASS=%d)" % elf_class)
        if elf_data != 1:
            raise ValueError("not little-endian (EI_DATA=%d)" % elf_data)
        shoff = struct.unpack_from("<Q", data, 40)[0]
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
        if shentsize != 64:
            raise ValueError("unexpected section header size %d" % shentsize)
        if not shoff or not shnum or shoff + shnum * shentsize > len(data):
            raise ValueError("invalid section header table")
        raw = [
            struct.unpack_from("<IIQQQQIIQQ", data, shoff + index * shentsize)
            for index in range(shnum)
        ]
        if shstrndx >= shnum:
            raise ValueError("invalid section name string table index")
        shstr = raw[shstrndx]
        shstr_data = data[shstr[4] : shstr[4] + shstr[5]]

        def section_name(offset):
            if offset >= len(shstr_data):
                return ""
            return shstr_data[offset:].split(b"\x00", 1)[0].decode("utf-8", "replace")

        for index, entry in enumerate(raw):
            (name, kind, flags, addr, offset, size, link, info, addr_align, entsize) = (
                entry
            )
            self.sections.append(
                {
                    "index": index,
                    "name": section_name(name),
                    "type": kind,
                    "flags": flags,
                    "addr": addr,
                    "offset": offset,
                    "size": size,
                    "link": link,
                    "info": info,
                    "align": addr_align,
                    "entsize": entsize,
                }
            )
        self._parse_symbols(data)

    def _parse_symbols(self, data) -> None:
        candidates = [
            section
            for section in self.sections
            if section["type"] in (SHT_SYMTAB, SHT_DYNSYM) and section["size"]
        ]
        if not candidates:
            raise ValueError("ELF has no symbol table")
        for section in candidates:
            link = section["link"]
            if link >= len(self.sections):
                continue
            strtab = self.sections[link]
            strings = data[strtab["offset"] : strtab["offset"] + strtab["size"]]
            count = section["size"] // 24
            for index in range(count):
                offset = section["offset"] + index * 24
                (st_name, st_info, _st_other, st_shndx, st_value, st_size) = (
                    struct.unpack_from("<IBBHQQ", data, offset)
                )
                if not st_name:
                    continue
                name = strings[st_name:].split(b"\x00", 1)[0].decode("utf-8", "replace")
                self.sym_count += 1
                if st_shndx == 0:
                    self.undefined.add(name)
                    continue
                self.defined.setdefault(name, []).append(
                    {
                        "name": name,
                        "bind": st_info >> 4,
                        "type": st_info & 0xF,
                        "shndx": st_shndx,
                        "value": st_value,
                        "size": st_size,
                    }
                )

    def section(self, name: str):
        for section in self.sections:
            if section["name"] == name:
                return section
        return None

    def symbol(self, name: str):
        entries = self.defined.get(name)
        return entries[0] if entries else None

    def is_defined(self, name: str) -> bool:
        return name in self.defined

    def is_defined_function(self, name: str) -> bool:
        entries = self.defined.get(name)
        if not entries:
            return False
        return any(entry["type"] == STT_FUNC for entry in entries)

    def allocated_progbits(self):
        return [
            section
            for section in self.sections
            if section["type"] != SHT_NOBITS
            and section["flags"] & SHF_ALLOC
            and section["size"]
        ]

    def noload_sections(self):
        return [
            section
            for section in self.sections
            if section["type"] == SHT_NOBITS and section["flags"] & SHF_ALLOC
        ]


# ---------------------------------------------------------------------------
# MLIR @vit signature
# ---------------------------------------------------------------------------

_MLIR_ARG_RE = re.compile(
    r"^%(?P<name>[A-Za-z0-9_]+):\s*(?P<kind>tensor|memref)<(?P<type>[^<>]*)>"
)


def _split_top_level(text: str, separator: str = ",") -> list:
    parts = []
    depth_angle = 0
    depth_paren = 0
    current = []
    for char in text:
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        if char == separator and depth_angle == 0 and depth_paren == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_vit_signature(path: Path) -> list:
    text = Path(path).read_text()
    match = re.search(r"func\.func\s+@vit\(", text)
    if not match:
        raise ValueError("@vit is missing from %s" % Path(path).name)
    open_index = text.index("(", match.start())
    depth = 0
    close_index = None
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                close_index = index
                break
    if close_index is None:
        raise ValueError("unbalanced @vit signature parentheses")
    body = text[open_index + 1 : close_index]
    signature = []
    for position, raw in enumerate(_split_top_level(body)):
        raw = raw.strip()
        if not raw:
            raise ValueError("empty @vit argument at position %d" % position)
        match = _MLIR_ARG_RE.match(raw)
        if not match:
            raise ValueError(
                "cannot parse @vit argument %d: %r" % (position, raw[:120])
            )
        type_text = match.group("type").strip()
        tokens = type_text.split("x")
        dtype = tokens[-1]
        shape_tokens = tokens[:-1]
        if dtype not in DTYPE_ITEMSIZE:
            raise ValueError(
                "unsupported @vit dtype %r at position %d" % (dtype, position)
            )
        if any(token == "?" for token in shape_tokens):
            raise ValueError("dynamic @vit shape at position %d" % position)
        try:
            shape = [int(token) for token in shape_tokens]
        except ValueError as error:
            raise ValueError(
                "cannot parse @vit arg %d type %r (%s)" % (position, type_text, error)
            )
        signature.append(
            {
                "position": position,
                "kind": match.group("kind"),
                "name": match.group("name"),
                "shape": shape,
                "dtype": dtype,
            }
        )
    return signature


# ---------------------------------------------------------------------------
# Input resolution / loading
# ---------------------------------------------------------------------------


def first_existing(candidates):
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def resolve_input_paths(replacement: Path):
    return {
        "workspace_contract": first_existing(
            [
                replacement / "workspace-buffers.json",
                replacement / "replacement-w8a8" / "workspace-buffers.json",
            ]
        ),
        "subgraph": first_existing(
            [
                replacement / "subgraph0.triton.mlir",
                replacement / "replacement-w8a8" / "subgraph0.triton.mlir",
            ]
        ),
        "report": first_existing(
            [
                replacement / "triton-call-replacement.json",
                replacement / "replacement-w8a8" / "triton-call-replacement.json",
            ]
        ),
        "adapters_c": first_existing(
            [
                replacement / "vit_triton_adapters.c",
                replacement / "replacement-w8a8" / "vit_triton_adapters.c",
            ]
        ),
    }


def resolve_model_input(output: Path):
    """Locate the pixel-values/reference-logits blobs embedded by the build."""
    candidates = [
        output / "inputs" / "model-input",
        V / "build" / "model-input",
        output / "model-input",
    ]
    for directory in candidates:
        pixel = directory / "pixel-values.f32"
        reference = directory / "reference-logits.f32"
        if pixel.is_file() and reference.is_file():
            return pixel, reference
    return None, None


def build_context(args) -> dict:
    output = Path(args.output).resolve()
    replacement = (
        Path(args.replacement).resolve()
        if args.replacement
        else output / "inputs" / "replacement"
    )
    buddy = (
        Path(args.buddy_root).resolve()
        if args.buddy_root
        else Path(os.environ.get("BUDDY_ROOT", REPO / "buddy-mlir"))
    )
    context = {
        "output": output,
        "replacement": replacement,
        "buddy": buddy,
        "paths": {name: output / name for name in REQUIRED_ARTIFACTS},
        "input_paths": resolve_input_paths(replacement),
        "errors": [],
        "checks": [],
    }
    return context


def load_context(context: dict) -> None:
    """Load every artifact/input once, recording load errors."""

    artifacts = {}
    for name, path in context["paths"].items():
        entry = {"path": str(path), "exists": path.is_file()}
        if entry["exists"]:
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
        artifacts[name] = entry
    context["artifacts"] = artifacts

    def guarded(key, loader):
        try:
            context[key] = loader()
        except Exception as error:
            context[key] = None
            context["errors"].append("%s: %s: %s" % (key, type(error).__name__, error))

    def load_workspace_contract():
        path = context["input_paths"]["workspace_contract"]
        if path is None:
            raise FileNotFoundError(
                "workspace-buffers.json not found under %s" % context["replacement"]
            )
        return load_json(path)

    def load_report():
        path = context["input_paths"]["report"]
        if path is None:
            raise FileNotFoundError(
                "triton-call-replacement.json not found under %s"
                % context["replacement"]
            )
        return load_json(path)

    def load_signature():
        path = context["input_paths"]["subgraph"]
        if path is None:
            raise FileNotFoundError(
                "subgraph0.triton.mlir not found under %s" % context["replacement"]
            )
        return parse_vit_signature(path)

    guarded("workspace_contract", load_workspace_contract)
    guarded("report", load_report)
    guarded("signature", load_signature)

    if context["workspace_contract"] is not None:
        contract = context["workspace_contract"]
        context["roots"] = list(contract.get("argument_roots") or [])
        context["workspace_roots"] = list(contract.get("workspace_roots") or [])
        context["mwa"] = list(contract.get("mutable_workspace_arguments") or [])
        context["wrapper_abi"] = contract.get("wrapper_abi") or {}
    if context["report"] is not None:
        report = context["report"]
        context["parameter_layout"] = list(report.get("parameter_layout") or [])
        context["input_layout"] = list(report.get("input_layout") or [])
        context["distinct_cases"] = list(report.get("distinct_cases") or [])
        context["replaced"] = list(report.get("replaced") or [])
        context["w8a8_linears"] = list(report.get("w8a8_linears") or [])
        context["parameter_map"] = {
            entry.get("name"): entry for entry in context["parameter_layout"]
        }
        context["input_map"] = {
            entry.get("name"): entry for entry in context["input_layout"]
        }

    guarded("layout", lambda: load_json(context["paths"]["params-layout.json"]))
    guarded("arg_map", lambda: load_json(context["paths"]["ARG-MAP.json"]))
    guarded("manifest", lambda: load_json(context["paths"]["manifest.json"]))
    objects_path = context["output"] / "objects.json"
    guarded(
        "objects", lambda: load_json(objects_path) if objects_path.is_file() else None
    )

    def load_params():
        return context["paths"]["params.bin"].read_bytes()

    guarded("params_bytes", load_params)
    if context["params_bytes"] is not None:
        context["params_sha256"] = sha256_bytes(context["params_bytes"])
        context["params_size"] = len(context["params_bytes"])

    def load_binary():
        return context["paths"]["vit_model.bin"].read_bytes()

    guarded("bin_bytes", load_binary)
    if context["bin_bytes"] is not None:
        context["bin_sha256"] = sha256_bytes(context["bin_bytes"])
        context["bin_size"] = len(context["bin_bytes"])

    if context["paths"]["vit_model.elf"].is_file():
        context["elf"] = ElfFile(context["paths"]["vit_model.elf"])
    else:
        context["elf"] = None

    def load_text(name):
        path = context["paths"][name]
        if not path.is_file():
            raise FileNotFoundError(name)
        return path.read_text()

    guarded("launcher_text", lambda: load_text("model_launcher.c"))
    guarded("workspace_asm", lambda: load_text("model_workspace.S"))
    guarded("blobs_asm", lambda: load_text("model_blobs.S"))
    context["pixel_values"], context["reference_logits"] = resolve_model_input(
        context["output"]
    )


# ---------------------------------------------------------------------------
# Check 0: artifact presence + provenance of staged inputs
# ---------------------------------------------------------------------------


def check_artifacts(context: dict) -> Check:
    check = Check("artifacts", "required artifact files and staged-input provenance")
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not context["artifacts"].get(name, {}).get("exists")
    ]
    if missing:
        check.fail("missing required artifacts: %s" % ", ".join(missing))
    if context["errors"]:
        check.fail("input loading errors: %s" % "; ".join(context["errors"][:6]))

    # Staged inputs under build/model-bin/inputs must agree with the canonical
    # build/replacement-w8a8 files, so the flat bin is not built from stale copies.
    canonical = V / "build" / "replacement-w8a8"
    checks = (
        ("workspace_contract", "workspace-buffers.json"),
        ("subgraph", "subgraph0.triton.mlir"),
        ("report", "triton-call-replacement.json"),
        ("adapters_c", "vit_triton_adapters.c"),
    )
    stage_mismatch = []
    for key, name in checks:
        staged = context["input_paths"].get(key)
        reference = canonical / name
        if staged is None or not reference.is_file():
            continue
        if sha256_file(staged) != sha256_file(reference):
            stage_mismatch.append(name)
    if stage_mismatch:
        check.fail(
            "staged inputs differ from build/replacement-w8a8: %s"
            % ", ".join(stage_mismatch)
        )

    report = context.get("report") or {}
    contract_record = report.get("workspace_buffer_contract") or {}
    staged_contract = context["input_paths"].get("workspace_contract")
    if contract_record.get("sha256") and staged_contract is not None:
        if contract_record["sha256"] != sha256_file(staged_contract):
            check.fail("replacement report workspace_buffer_contract sha256 mismatch")
    primitive_record = report.get("attention_primitive_manifest") or {}
    if primitive_record.get("file") and primitive_record.get("sha256"):
        primitive_path = Path(primitive_record["file"])
        if (
            primitive_path.is_file()
            and sha256_file(primitive_path) != primitive_record["sha256"]
        ):
            check.fail(
                "replacement report attention_primitive_manifest sha256 mismatch"
            )

    check.note(
        present=[
            name
            for name in REQUIRED_ARTIFACTS
            if context["artifacts"].get(name, {}).get("exists")
        ],
        missing=missing,
        staged_inputs={
            key: str(context["input_paths"].get(key))
            for key in ("workspace_contract", "subgraph", "report")
        },
        staged_vs_canonical_mismatch=stage_mismatch,
    )
    return check


# ---------------------------------------------------------------------------
# Check 1: @vit argument contract
# ---------------------------------------------------------------------------


def check_argument_contract(context: dict) -> Check:
    check = Check(
        "argument-contract",
        "@vit argument order/shape/dtype versus the replacement workspace contract",
    )
    contract = context.get("workspace_contract")
    signature = context.get("signature")
    roots = context.get("roots") or []
    parameter_map = context.get("parameter_map") or {}
    input_map = context.get("input_map") or {}
    if contract is None or signature is None or context.get("report") is None:
        check.fail(
            "required contract inputs (workspace-buffers.json / subgraph0.triton.mlir / "
            "triton-call-replacement.json) unavailable"
        )
        return check

    if contract.get("schema") != 1:
        check.fail(
            "workspace-buffers.json schema is %r, expected 1"
            % (contract.get("schema"),)
        )
    if contract.get("status") != "PASS":
        check.fail(
            "workspace-buffers.json status is %r, expected PASS"
            % (contract.get("status"),)
        )
    if contract.get("function") != "vit":
        check.fail(
            "workspace-buffers.json function is %r, expected vit"
            % (contract.get("function"),)
        )
    if len(roots) != EXPECTED_VIT_ARGS:
        check.fail(
            "argument_roots has %d entries, expected %d"
            % (len(roots), EXPECTED_VIT_ARGS)
        )
    if len(set(roots)) != len(roots):
        check.fail("argument_roots contains duplicate names")
    if len(signature) != EXPECTED_VIT_ARGS:
        check.fail(
            "@vit signature has %d arguments, expected %d"
            % (len(signature), EXPECTED_VIT_ARGS)
        )
    if len(parameter_map) != EXPECTED_PARAMS:
        check.fail(
            "parameter_layout has %d names, expected %d"
            % (len(parameter_map), EXPECTED_PARAMS)
        )
    if len(input_map) != EXPECTED_WORKSPACES + EXPECTED_IMAGE_ARGS:
        check.fail(
            "input_layout has %d names, expected %d"
            % (len(input_map), EXPECTED_WORKSPACES + EXPECTED_IMAGE_ARGS)
        )

    # Classify the roots; the three classes must be exact and disjoint.
    workspace_set = set(context.get("workspace_roots") or [])
    param_roots = [root for root in roots if root in parameter_map]
    workspace_roots_in_order = [root for root in roots if root in workspace_set]
    image_roots = [
        root
        for root in roots
        if root not in parameter_map and root not in workspace_set
    ]
    if len(param_roots) != EXPECTED_PARAMS:
        check.fail(
            "classified %d parameter roots, expected %d"
            % (len(param_roots), EXPECTED_PARAMS)
        )
    if len(workspace_roots_in_order) != EXPECTED_WORKSPACES:
        check.fail(
            "classified %d workspace roots, expected %d"
            % (len(workspace_roots_in_order), EXPECTED_WORKSPACES)
        )
    if len(image_roots) != EXPECTED_IMAGE_ARGS:
        check.fail(
            "classified %d image roots, expected %d"
            % (len(image_roots), EXPECTED_IMAGE_ARGS)
        )
    if image_roots and image_roots != ["arg0_1"]:
        check.fail(
            "expected the single image root to be arg0_1, found %r" % (image_roots[:5],)
        )
    if roots and roots[0] != "arg0_1":
        check.fail("expected image root arg0_1 at position 0, found %r" % (roots[0],))

    # Cross-check the workspace contract's own order evidence.
    workspace_names = set(context.get("workspace_roots") or [])
    if len(workspace_names) != EXPECTED_WORKSPACES:
        check.fail(
            "workspace_roots has %d unique names, expected %d"
            % (len(workspace_names), EXPECTED_WORKSPACES)
        )
    if workspace_names != set(workspace_roots_in_order):
        differing = sorted(
            workspace_names.symmetric_difference(set(workspace_roots_in_order))
        )
        check.fail(
            "workspace_roots names differ from the workspace-classified argument_roots: %s"
            % (differing[:10],)
        )
    mwa = context.get("mwa") or []
    if len(mwa) != EXPECTED_WORKSPACES:
        check.fail(
            "mutable_workspace_arguments has %d entries, expected %d"
            % (len(mwa), EXPECTED_WORKSPACES)
        )
    expected_positions = [
        index for index, root in enumerate(roots) if root in workspace_names
    ]
    mwa_positions = []
    mwa_ok = True
    for entry in mwa:
        position = entry.get("argument")
        mwa_positions.append(position)
        if not isinstance(position, int) or position < 0 or position >= len(roots):
            check.fail(
                "mutable workspace argument position %r is out of range" % (position,)
            )
            mwa_ok = False
            continue
        if roots[position] != entry.get("root"):
            check.fail(
                "mutable workspace position %d maps to %r but the entry records %r"
                % (position, roots[position], entry.get("root"))
            )
            mwa_ok = False
    if mwa_ok and mwa_positions != expected_positions:
        check.fail("mutable_workspace_arguments is not in argument_roots order")
    wrapper = context.get("wrapper_abi") or {}
    if wrapper.get("status") != "PASS":
        check.fail("wrapper_abi status is %r, expected PASS" % (wrapper.get("status"),))
    if wrapper.get("callee") != "vit":
        check.fail("wrapper_abi callee is %r, expected vit" % (wrapper.get("callee"),))
    if wrapper.get("mutable_workspace_positions") != expected_positions:
        check.fail(
            "wrapper_abi.mutable_workspace_positions differs from the recomputed positions"
        )

    # Positional shape/dtype check against the MLIR signature.
    signature_mismatches = []
    for position, (root, argument) in enumerate(zip(roots, signature)):
        if argument.get("position") != position:
            signature_mismatches.append(
                "position %d records %r" % (position, argument.get("position"))
            )
            continue
        name = argument.get("name")
        if name not in ("arg%d" % position, str(position)):
            signature_mismatches.append(
                "signature argument %d is named %%%s" % (position, name)
            )
        if root in parameter_map:
            definition = parameter_map[root]
        elif root in input_map:
            definition = input_map[root]
        else:
            signature_mismatches.append("no layout entry for root %r" % (root,))
            continue
        expected_dtype = LAYOUT_DTYPE.get(str(definition.get("dtype")))
        if expected_dtype is None:
            signature_mismatches.append(
                "unsupported layout dtype %r for %s" % (definition.get("dtype"), root)
            )
            continue
        expected_shape = [int(dim) for dim in (definition.get("shape") or [])]
        if argument.get("shape") != expected_shape:
            signature_mismatches.append(
                "%s: @vit shape %s != layout shape %s"
                % (root, argument.get("shape"), expected_shape)
            )
        if argument.get("dtype") != expected_dtype:
            signature_mismatches.append(
                "%s: @vit dtype %s != layout dtype %s"
                % (root, argument.get("dtype"), expected_dtype)
            )
        if len(signature_mismatches) >= 20:
            break
    if signature_mismatches:
        check.fail(
            "%d positional shape/dtype mismatches; first: %s"
            % (len(signature_mismatches), "; ".join(signature_mismatches[:5]))
        )

    contract_path = context["input_paths"]["workspace_contract"]
    expected_contract = context["report"].get("workspace_buffer_contract") or {}
    if expected_contract.get("sha256") and contract_path is not None:
        actual = sha256_file(contract_path)
        if expected_contract["sha256"] != actual:
            check.fail(
                "workspace-buffers.json sha256 %s does not match the report record %s"
                % (actual, expected_contract["sha256"])
            )

    check.note(
        vit_args=len(roots),
        signature_args=len(signature),
        parameters=len(param_roots),
        image=len(image_roots),
        workspaces=len(workspace_roots_in_order),
        mutable_workspace_positions=len(expected_positions),
        parameter_layout=len(parameter_map),
        input_layout=len(input_map),
        contract_path=str(contract_path),
        signature_path=str(context["input_paths"]["subgraph"]),
    )
    return check


# ---------------------------------------------------------------------------
# Check 2: params.bin packing / params-layout.json
# ---------------------------------------------------------------------------


def check_params_layout(context: dict) -> Check:
    check = Check(
        "params-layout",
        "params.bin packing: 273 parameter entries, 64B alignment, no overlap, per-region sha256",
    )
    parameter_layout = context.get("parameter_layout") or []
    layout = context.get("layout")
    params = context.get("params_bytes")
    if layout is None or params is None or not parameter_layout:
        check.fail("params-layout.json / params.bin / parameter_layout unavailable")
        return check
    entries = layout.get("entries") or []
    if len(parameter_layout) != EXPECTED_PARAMS:
        check.fail(
            "parameter_layout has %d entries, expected %d"
            % (len(parameter_layout), EXPECTED_PARAMS)
        )
    if len(entries) != EXPECTED_PARAMS:
        check.fail(
            "params-layout entries has %d entries, expected %d"
            % (len(entries), EXPECTED_PARAMS)
        )
        return check
    if [entry.get("name") for entry in entries] != [
        entry.get("name") for entry in parameter_layout
    ]:
        check.fail("params-layout entry order does not follow parameter_layout order")

    expected = []
    offset = 0
    payload_total = 0
    problems = []
    for definition in parameter_layout:
        name = definition.get("name")
        dtype = LAYOUT_DTYPE.get(str(definition.get("dtype")))
        if dtype is None:
            problems.append(
                "%s has unsupported layout dtype %r" % (name, definition.get("dtype"))
            )
            continue
        shape = [int(dim) for dim in (definition.get("shape") or [])]
        elements = product(shape)
        nbytes = elements * DTYPE_ITEMSIZE[dtype]
        start = align_up(offset)
        if start % ALIGNMENT != 0:
            problems.append(
                "%s offset %d is not %d-byte aligned" % (name, start, ALIGNMENT)
            )
        if start < offset:
            problems.append("%s offset %d overlaps the previous region" % (name, start))
        expected.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "elements": elements,
                "nbytes": nbytes,
                "offset": start,
            }
        )
        offset = start + nbytes
        payload_total += nbytes
    total = align_up(offset)
    padding = total - payload_total
    if len(expected) != EXPECTED_PARAMS:
        check.fail(
            "could not derive %d parameter entries from parameter_layout"
            % EXPECTED_PARAMS
        )

    previous_end = 0
    for position, (got, want) in enumerate(zip(entries, expected)):
        if got.get("name") != want["name"]:
            problems.append(
                "entry %d name %r != parameter_layout name %r"
                % (position, got.get("name"), want["name"])
            )
            continue
        for field in ("shape", "dtype", "elements", "nbytes", "offset"):
            if got.get(field) != want[field]:
                problems.append(
                    "%s: %s %r != recomputed %r"
                    % (want["name"], field, got.get(field), want[field])
                )
        got_offset = got.get("offset")
        if not isinstance(got_offset, int) or got_offset % ALIGNMENT != 0:
            problems.append(
                "%s: offset %r is not %d-byte aligned"
                % (want["name"], got_offset, ALIGNMENT)
            )
        if isinstance(got_offset, int):
            if got_offset < previous_end:
                problems.append(
                    "%s: offset %d overlaps the previous region ending at %d"
                    % (want["name"], got_offset, previous_end)
                )
            if got_offset != align_up(previous_end):
                problems.append(
                    "%s: packed offset %d != padded previous end %d"
                    % (want["name"], got_offset, align_up(previous_end))
                )
            previous_end = got_offset + (got.get("nbytes") or 0)
        if len(problems) >= 20:
            break
    if problems:
        check.fail(
            "%d packing problems; first: %s" % (len(problems), "; ".join(problems[:6]))
        )

    if layout.get("bytes") != total:
        check.fail(
            "params-layout bytes %r != recomputed total %d"
            % (layout.get("bytes"), total)
        )
    if len(params) != total:
        check.fail("params.bin is %d bytes, expected %d" % (len(params), total))
    if payload_total + padding != total:
        check.fail(
            "payload %d + padding %d != total %d" % (payload_total, padding, total)
        )
    expected_sha = sha256_bytes(params)
    if layout.get("sha256") != expected_sha:
        check.fail(
            "params-layout sha256 %r != actual params.bin sha256 %s"
            % (layout.get("sha256"), expected_sha)
        )
    view = memoryview(params)
    bad_regions = []
    for entry in entries:
        offset_value = entry.get("offset")
        nbytes = entry.get("nbytes")
        if not isinstance(offset_value, int) or not isinstance(nbytes, int):
            bad_regions.append("%r has non-integer offset/size" % (entry.get("name"),))
            continue
        if offset_value < 0 or offset_value + nbytes > len(params):
            bad_regions.append("%s region is outside params.bin" % (entry.get("name"),))
            continue
        actual = sha256_bytes(view[offset_value : offset_value + nbytes])
        if actual != entry.get("sha256"):
            bad_regions.append("%s region sha256 mismatch" % (entry.get("name"),))
    if bad_regions:
        check.fail(
            "%d region checksum problems; first: %s"
            % (len(bad_regions), bad_regions[:5])
        )

    check.note(
        entries=len(entries),
        total_bytes=total,
        payload_bytes=payload_total,
        padding_bytes=padding,
        alignment=ALIGNMENT,
        params_sha256=expected_sha,
        paths={
            "params": str(context["paths"]["params.bin"]),
            "layout": str(context["paths"]["params-layout.json"]),
        },
    )
    return check


# ---------------------------------------------------------------------------
# Check 3: ARG-MAP.json + generated launcher / assembly blobs
# ---------------------------------------------------------------------------

_MEMREF_ARG_RE = re.compile(
    r"^\s*MemRef(?P<rank>\d+) a_(?P<index>\d+) = \{\(void \*\)0, (?P<pointer>.+?), 0, "
    r"\{(?P<sizes>[^{}]*)\}, \{(?P<strides>[^{}]*)\}\};\s*$",
    re.MULTILINE,
)
_RESULT_RE = re.compile(
    r"^\s*MemRef1 model_result = \{\(void \*\)0, \(void \*\)model_logits, 0, "
    r"\{1000\}, \{1\}\};\s*$",
    re.MULTILINE,
)
_EXTERN_RE = re.compile(
    r"^extern void _mlir_ciface_vit\((?P<args>.*)\);\s*$", re.MULTILINE
)
_CALL_RE = re.compile(r"^\s*_mlir_ciface_vit\((?P<args>.*)\);\s*$", re.MULTILINE)
_LOGITS_RE = re.compile(r"^static float model_logits\[1000\].*;\s*$", re.MULTILINE)
_PARAM_SOURCE_RE = re.compile(r"model_params_blob\s*\+\s*(\d+)")
_WORKSPACE_SOURCE_RE = re.compile(r"model_workspace_blob\s*\+\s*(\d+)")


def _int_list(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return [int(item) for item in re.split(r"[,\s]+", text) if item]


def _parse_incbin_blocks(text: str) -> list:
    blocks = []
    current = {"section": None, "globl": None, "incbin": None}
    for line in text.splitlines():
        stripped = line.strip()
        section = re.match(
            r'^\.section\s+(\S+)\s*,\s*"([a-zA-Z]+)"\s*,\s*@(\w+)', stripped
        )
        if section:
            if current["section"] or current["incbin"]:
                blocks.append(current)
                current = {"section": None, "globl": None, "incbin": None}
            current["section"] = section.group(1)
            current["flags"] = section.group(2)
            current["type"] = section.group(3)
            continue
        globl = re.match(r"^\.globl\s+(\S+)", stripped)
        if globl:
            current["globl"] = globl.group(1)
            continue
        incbin = re.match(r'^\.incbin\s+"([^"]+)"', stripped)
        if incbin:
            current["incbin"] = incbin.group(1)
    if current["section"] or current["incbin"]:
        blocks.append(current)
    return blocks


def check_argument_map(context: dict) -> Check:
    check = Check(
        "arg-map-launcher",
        "ARG-MAP/launcher descriptors: classification, offsets, dense strides, blob assembly",
    )
    arg_map = context.get("arg_map")
    roots = context.get("roots") or []
    signature = context.get("signature") or []
    parameter_map = context.get("parameter_map") or {}
    workspace_names = set(context.get("workspace_roots") or [])
    layout = context.get("layout") or {}
    layout_entries = {
        entry.get("name"): entry for entry in (layout.get("entries") or [])
    }
    launcher = context.get("launcher_text")
    workspace_asm = context.get("workspace_asm")
    blobs_asm = context.get("blobs_asm")
    if arg_map is None or launcher is None:
        check.fail("ARG-MAP.json / model_launcher.c unavailable")
        return check
    if len(roots) != EXPECTED_VIT_ARGS or len(signature) != EXPECTED_VIT_ARGS:
        check.fail(
            "cannot recompute ARG-MAP expectations without the 734-entry contract"
        )
        return check
    if len(layout_entries) != EXPECTED_PARAMS:
        check.fail("params-layout entries do not cover the 273 parameters")

    expected = []
    workspace_offset = 0
    workspace_index = 0
    workspace_offsets = {}
    for position, (root, argument) in enumerate(zip(roots, signature)):
        dtype = argument["dtype"]
        shape = [int(dim) for dim in argument["shape"]]
        nbytes = product(shape) * DTYPE_ITEMSIZE[dtype]
        descriptor = {
            "index": position,
            "name": root,
            "dtype": dtype,
            "shape": shape,
            "nbytes": nbytes,
        }
        if root in parameter_map:
            entry = layout_entries.get(root)
            if entry is None:
                check.fail("parameter %s is missing from params-layout.json" % root)
                continue
            descriptor["kind"] = "param"
            descriptor["param_offset"] = entry.get("offset")
            descriptor["param_bytes"] = entry.get("nbytes")
            descriptor["param_shape"] = entry.get("shape")
            descriptor["param_dtype"] = entry.get("dtype")
        elif root == "arg0_1":
            descriptor["kind"] = "input"
        elif root in workspace_names:
            start = align_up(workspace_offset)
            descriptor["kind"] = "workspace"
            descriptor["workspace"] = workspace_index
            descriptor["workspace_offset"] = start
            workspace_offsets[root] = start
            workspace_offset = start + nbytes
            workspace_index += 1
        else:
            check.fail("root %r has no recomputed classification" % (root,))
            continue
        expected.append(descriptor)
    workspace_bytes = align_up(workspace_offset)

    problems = []
    if arg_map.get("vit_args") != EXPECTED_VIT_ARGS:
        problems.append(
            "vit_args %r != %d" % (arg_map.get("vit_args"), EXPECTED_VIT_ARGS)
        )
    if arg_map.get("ciface_args") != EXPECTED_CIFACE_ARGS:
        problems.append(
            "ciface_args %r != %d" % (arg_map.get("ciface_args"), EXPECTED_CIFACE_ARGS)
        )
    result = arg_map.get("result") or {}
    if (
        result.get("shape") != RESULT_SHAPE
        or result.get("dtype") != "f32"
        or result.get("nbytes") != RESULT_NBYTES
    ):
        problems.append(
            "result descriptor %r != {shape %r, dtype f32, nbytes %d}"
            % (result, RESULT_SHAPE, RESULT_NBYTES)
        )
    if arg_map.get("params") != EXPECTED_PARAMS:
        problems.append(
            "params count %r != %d" % (arg_map.get("params"), EXPECTED_PARAMS)
        )
    if arg_map.get("workspaces") != EXPECTED_WORKSPACES:
        problems.append(
            "workspaces count %r != %d"
            % (arg_map.get("workspaces"), EXPECTED_WORKSPACES)
        )
    if arg_map.get("workspace_bytes") != workspace_bytes:
        problems.append(
            "workspace_bytes %r != recomputed %d"
            % (arg_map.get("workspace_bytes"), workspace_bytes)
        )
    args = arg_map.get("args") or []
    if len(args) != EXPECTED_VIT_ARGS:
        problems.append(
            "ARG-MAP args has %d entries, expected %d" % (len(args), EXPECTED_VIT_ARGS)
        )
    for got, want in zip(args, expected):
        index = want["index"]
        if got.get("index") != index:
            problems.append("arg %d index %r != %d" % (index, got.get("index"), index))
        if got.get("name") != want["name"]:
            problems.append(
                "arg %d name %r != %r" % (index, got.get("name"), want["name"])
            )
            continue
        if got.get("kind") != want["kind"]:
            problems.append(
                "%s kind %r != %r" % (want["name"], got.get("kind"), want["kind"])
            )
        if got.get("dtype") != want["dtype"]:
            problems.append(
                "%s dtype %r != %r" % (want["name"], got.get("dtype"), want["dtype"])
            )
        if got.get("shape") != want["shape"]:
            problems.append(
                "%s shape %r != %r" % (want["name"], got.get("shape"), want["shape"])
            )
        if got.get("nbytes") != want["nbytes"]:
            problems.append(
                "%s nbytes %r != %r" % (want["name"], got.get("nbytes"), want["nbytes"])
            )
        if want["kind"] == "param":
            source = got.get("source") or ""
            match = _PARAM_SOURCE_RE.search(source)
            if not match:
                problems.append(
                    "%s parameter source %r does not reference model_params_blob"
                    % (want["name"], source)
                )
            elif int(match.group(1)) != want["param_offset"]:
                problems.append(
                    "%s parameter offset %s != params-layout offset %r"
                    % (want["name"], match.group(1), want["param_offset"])
                )
            if (
                got.get("shape") != want["param_shape"]
                or got.get("dtype") != want["param_dtype"]
            ):
                problems.append(
                    "%s ARG-MAP descriptor disagrees with the params-layout entry"
                    % want["name"]
                )
            if got.get("nbytes") != want["param_bytes"]:
                problems.append(
                    "%s parameter nbytes %r != params-layout nbytes %r"
                    % (want["name"], got.get("nbytes"), want["param_bytes"])
                )
        elif want["kind"] == "input":
            if (got.get("source") or "") != "(void *)(uintptr_t)model_input_blob":
                problems.append(
                    "image argument source %r != model_input_blob"
                    % (got.get("source"),)
                )
        else:
            if got.get("workspace") != want["workspace"]:
                problems.append(
                    "%s workspace index %r != recomputed %d"
                    % (want["name"], got.get("workspace"), want["workspace"])
                )
            if got.get("workspace_offset") != want["workspace_offset"]:
                problems.append(
                    "%s workspace offset %r != recomputed %d"
                    % (
                        want["name"],
                        got.get("workspace_offset"),
                        want["workspace_offset"],
                    )
                )
    if problems:
        check.fail(
            "%d ARG-MAP descriptor problems; first: %s"
            % (len(problems), "; ".join(problems[:6]))
        )

    launcher_problems = []
    matches = list(_MEMREF_ARG_RE.finditer(launcher))
    if len(matches) != EXPECTED_VIT_ARGS:
        launcher_problems.append(
            "model_launcher.c has %d MemRef argument descriptors, expected %d"
            % (len(matches), EXPECTED_VIT_ARGS)
        )
    seen_indices = set()
    for match in matches:
        index = int(match.group("index"))
        rank = int(match.group("rank"))
        if index >= len(roots):
            launcher_problems.append(
                "launcher descriptor index %d out of range" % index
            )
            continue
        seen_indices.add(index)
        argument = signature[index]
        root = roots[index]
        shape = [int(dim) for dim in argument["shape"]]
        pointer = match.group("pointer").strip()
        sizes = _int_list(match.group("sizes"))
        strides = _int_list(match.group("strides"))
        if rank != len(shape):
            launcher_problems.append("a_%d rank %d != %d" % (index, rank, len(shape)))
        if sizes != shape:
            launcher_problems.append("a_%d sizes %r != %r" % (index, sizes, shape))
        if strides != dense_strides(shape):
            launcher_problems.append(
                "a_%d strides %r != dense %r" % (index, strides, dense_strides(shape))
            )
        if root in parameter_map:
            offset = layout_entries.get(root, {}).get("offset")
            source = _PARAM_SOURCE_RE.search(pointer)
            if not source or int(source.group(1)) != offset:
                launcher_problems.append(
                    "a_%d parameter pointer %r does not use params offset %r"
                    % (index, pointer, offset)
                )
        elif root == "arg0_1":
            if "model_input_blob" not in pointer:
                launcher_problems.append(
                    "a_%d image pointer %r does not use model_input_blob"
                    % (index, pointer)
                )
        else:
            expected_offset = workspace_offsets.get(root)
            source = _WORKSPACE_SOURCE_RE.search(pointer)
            if not source or int(source.group(1)) != expected_offset:
                launcher_problems.append(
                    "a_%d workspace pointer %r does not use workspace offset %r"
                    % (index, pointer, expected_offset)
                )
    if seen_indices != set(range(EXPECTED_VIT_ARGS)):
        launcher_problems.append("launcher descriptors do not cover a_0..a_733")
    if not _RESULT_RE.search(launcher):
        launcher_problems.append(
            "model_result descriptor line missing or not {1000}/{1}"
        )
    if not _LOGITS_RE.search(launcher):
        launcher_problems.append("model_logits[1000] declaration missing")
    extern = _EXTERN_RE.search(launcher)
    prototype = None
    if not extern:
        launcher_problems.append("extern _mlir_ciface_vit prototype missing")
    else:
        prototype = [
            item.strip()
            for item in _split_top_level(extern.group("args"))
            if item.strip()
        ]
        if len(prototype) != EXPECTED_CIFACE_ARGS:
            launcher_problems.append(
                "extern _mlir_ciface_vit has %d parameters, expected %d"
                % (len(prototype), EXPECTED_CIFACE_ARGS)
            )
        else:
            if prototype[0] != "MemRef1 *":
                launcher_problems.append(
                    "extern result parameter is %r, expected MemRef1 *" % prototype[0]
                )
            for position in range(EXPECTED_VIT_ARGS):
                want = "MemRef%d *" % len(signature[position]["shape"])
                if prototype[position + 1] != want:
                    launcher_problems.append(
                        "extern parameter %d is %r, expected %r"
                        % (position + 1, prototype[position + 1], want)
                    )
                    break
    call = _CALL_RE.search(launcher)
    if not call:
        launcher_problems.append("_mlir_ciface_vit call site missing")
    else:
        call_args = [
            item.strip()
            for item in _split_top_level(call.group("args"))
            if item.strip()
        ]
        wanted = ["&model_result"] + [
            "&a_%d" % index for index in range(EXPECTED_VIT_ARGS)
        ]
        if call_args != wanted:
            launcher_problems.append(
                "_mlir_ciface_vit call arguments do not match the generated descriptors"
            )
    if "int e6_main(void)" not in launcher:
        launcher_problems.append("e6_main definition missing from the launcher")
    if launcher_problems:
        check.fail(
            "%d launcher problems; first: %s"
            % (len(launcher_problems), "; ".join(launcher_problems[:6]))
        )

    asm_problems = []
    if workspace_asm is None:
        asm_problems.append("model_workspace.S missing")
    else:
        section = re.search(
            r'^\s*\.section\s+(\.model_workspace)\s*,\s*"([a-z]+)"\s*,\s*@nobits',
            workspace_asm,
            re.MULTILINE,
        )
        if not section:
            asm_problems.append(
                "model_workspace.S does not define a NOLOAD .model_workspace section"
            )
        space = re.search(r"^\s*\.space\s+(\d+)", workspace_asm, re.MULTILINE)
        size = re.search(
            r"^\s*\.size\s+model_workspace_blob\s*,\s*(\d+)",
            workspace_asm,
            re.MULTILINE,
        )
        if not space or int(space.group(1)) != workspace_bytes:
            asm_problems.append(
                "model_workspace.S .space %r != recomputed workspace_bytes %d"
                % (space.group(1) if space else None, workspace_bytes)
            )
        if not size or int(size.group(1)) != workspace_bytes:
            asm_problems.append(
                "model_workspace.S .size %r != recomputed workspace_bytes %d"
                % (size.group(1) if size else None, workspace_bytes)
            )
        if "model_workspace_blob:" not in workspace_asm:
            asm_problems.append(
                "model_workspace_blob symbol missing from model_workspace.S"
            )
    if blobs_asm is None:
        asm_problems.append("model_blobs.S missing")
    else:
        blocks = _parse_incbin_blocks(blobs_asm)
        expected_blocks = [
            (".model_params", "model_params_blob", context["paths"]["params.bin"]),
            (".model_input", "model_input_blob", context.get("pixel_values")),
            (
                ".model_reference",
                "model_reference_blob",
                context.get("reference_logits"),
            ),
        ]
        if len(blocks) != len(expected_blocks):
            asm_problems.append(
                "model_blobs.S has %d .incbin blocks, expected %d"
                % (len(blocks), len(expected_blocks))
            )
        for block, (section_name, symbol, source) in zip(blocks, expected_blocks):
            if block.get("section") != section_name:
                asm_problems.append(
                    "model_blobs.S block section %r != %r"
                    % (block.get("section"), section_name)
                )
            if block.get("globl") != symbol:
                asm_problems.append(
                    "model_blobs.S block symbol %r != %r" % (block.get("globl"), symbol)
                )
            if source is None:
                asm_problems.append(
                    "expected source blob for %s is unavailable" % symbol
                )
                continue
            incbin = block.get("incbin")
            if not incbin:
                asm_problems.append("%s .incbin path missing" % symbol)
                continue
            incbin_path = Path(incbin)
            if not incbin_path.is_file():
                asm_problems.append(
                    "%s .incbin path %r does not exist" % (symbol, incbin)
                )
                continue
            if sha256_file(incbin_path) != sha256_file(Path(source)):
                asm_problems.append(
                    "%s .incbin content does not match %s" % (symbol, Path(source).name)
                )
    if asm_problems:
        check.fail(
            "%d assembly problems; first: %s"
            % (len(asm_problems), "; ".join(asm_problems[:6]))
        )

    check.note(
        vit_args=len(args),
        workspace_bytes=workspace_bytes,
        launcher_descriptors=len(matches),
        extern_parameters=len(prototype) if prototype else None,
        pixel_values=str(context.get("pixel_values")),
        reference_logits=str(context.get("reference_logits")),
    )
    return check


# ---------------------------------------------------------------------------
# Check 4: ELF structure and symbols
# ---------------------------------------------------------------------------


def model_cache_layout_problems(elf: ElfFile) -> list:
    """Check the PMP cache-policy contract independently of linker ASSERTs."""
    problems = []
    required = (
        "__nh_e6_ddr_window_start",
        "__nh_e6_ddr_window_end",
        "__bss_start",
        "__bss_end",
        "__model_heap_end",
        "__e6_stack_bottom",
        "__e6_stack_top",
        "__nh_stack_bottom",
        "__nh_stack_top",
    )
    missing = [name for name in required if elf.symbol(name) is None]
    if missing:
        return ["cache layout symbols missing: " + ", ".join(missing)]
    values = {name: elf.symbol(name)["value"] for name in required}
    start = values["__nh_e6_ddr_window_start"]
    end = values["__nh_e6_ddr_window_end"]
    if not start or end <= start or (start | end) & 63:
        problems.append("PMP DDR window must be nonempty and 64-byte aligned")
    for name in (
        ".model_workspace",
        ".model_heap",
        ".model_operand_bss",
        ".bosc_ame_private_ddr",
    ):
        section = elf.section(name)
        if section is None:
            problems.append("operand NOLOAD section missing: " + name)
        elif (
            section["type"] != SHT_NOBITS
            or not section["size"]
            or section["addr"] < start
            or section["addr"] + section["size"] > end
        ):
            problems.append(name + " must be NOLOAD and wholly inside PMP DDR window")
    workspace = elf.section(".model_workspace")
    if workspace and workspace["addr"] != start:
        problems.append("PMP DDR window must begin at the actual workspace section")
    control = elf.section(".bss")
    if control is None or control["addr"] < end:
        problems.append("control BSS must follow PMP DDR window in stable entry 6")
    operand = elf.section(".model_operand_bss")
    bss_start = values["__bss_start"]
    bss_end = values["__bss_end"]
    if (
        operand is None
        or control is None
        or bss_start != operand["addr"]
        or bss_end != control["addr"] + control["size"]
        or bss_start < values["__model_heap_end"]
    ):
        problems.append(
            "startup clear must cover operand/control BSS and exclude model heap"
        )
    stacks = []
    for core in ("e6", "nh"):
        bottom = values["__" + core + "_stack_bottom"]
        top = values["__" + core + "_stack_top"]
        if bottom < end or top <= bottom or (bottom | top) & 15:
            problems.append(
                core + " stack must be aligned and wholly in stable entry 6"
            )
        if bss_end > bottom:
            problems.append("startup BSS clear overlaps " + core + " stack")
        stacks.append((bottom, top))
    if stacks[0][0] < stacks[1][1] and stacks[1][0] < stacks[0][1]:
        problems.append("E6 and NH stacks overlap")
    for section in elf.allocated_progbits():
        if section["addr"] + section["size"] > start:
            problems.append(
                "loaded image must precede PMP DDR window: " + section["name"]
            )
    if elf.is_defined_function("vit_model_trace_event"):
        # NH polls its trace word before E6 publishes results. Noncoherent
        # caches require whole-line ownership, including compiler padding.
        control_lines = set()
        for name in ("vit_op_trace_e6", "vit_op_trace_nh"):
            symbol = elf.symbol(name)
            if (
                symbol is None
                or symbol["type"] != STT_OBJECT
                or symbol["size"] != 64
                or symbol["value"] % 64
            ):
                problems.append(name + " must occupy one aligned 64-byte cache line")
            else:
                control_lines.add(symbol["value"] // 64)
        if len(control_lines) != 2:
            problems.append("E6 and NH trace controls must occupy distinct cache lines")
        for suffix in (
            "returned",
            "validation_executed",
            "checked",
            "errors",
            "cycles",
            "logits_fnv64",
            "top1",
            "max_abs_bits",
        ):
            name = "model_" + suffix
            symbol = elf.symbol(name)
            if symbol is None or not symbol["size"]:
                problems.append("result symbol missing or empty: " + name)
                continue
            first = symbol["value"] // 64
            last = (symbol["value"] + symbol["size"] - 1) // 64
            if any(first <= line <= last for line in control_lines):
                problems.append(name + " shares a cache line with a trace control")
    return problems


def check_elf_symbols(context: dict) -> Check:
    check = Check(
        "elf-symbols",
        "ELF64-RISCV structure, entry symbols, 1543 adapters, 28 kernel adapters, "
        "undefined allowlist",
    )
    elf = context.get("elf")
    if elf is None or elf.error:
        check.fail(
            "ELF unavailable or unparsable: %s" % ((elf.error if elf else "missing"),)
        )
        return check
    report = context.get("report")
    if report is None:
        check.fail("replacement report unavailable; cannot recompute required symbols")
        return check

    header = elf.header
    if header.get("elf_class") != 2:
        check.fail("ELF is not ELF64 (EI_CLASS=%r)" % (header.get("elf_class"),))
    if header.get("elf_data") != 1:
        check.fail("ELF is not little-endian (EI_DATA=%r)" % (header.get("elf_data"),))
    if header.get("e_machine") != 243:
        check.fail("ELF machine %r is not RISC-V (243)" % (header.get("e_machine"),))
    if header.get("e_type") != 2:
        check.fail("ELF type %r is not ET_EXEC (2)" % (header.get("e_type"),))
    if header.get("e_entry") != NH_ENTRY:
        check.fail(
            "ELF entry point %r is not %s" % (header.get("e_entry"), fmt_hex(NH_ENTRY))
        )
    if not elf.sections:
        check.fail("ELF has no section table")
        return check

    start = elf.symbol("_start")
    if start is None or start["type"] != STT_FUNC:
        check.fail("_start is not a defined function")
    elif start["value"] != NH_ENTRY:
        check.fail(
            "_start is at %s, expected %s"
            % (fmt_hex(start["value"]), fmt_hex(NH_ENTRY))
        )
    e6_start = elf.symbol("e6_start")
    if e6_start is None or e6_start["type"] != STT_FUNC:
        check.fail("e6_start is not a defined function")
    elif e6_start["value"] != E6_ENTRY:
        check.fail(
            "e6_start is at %s, expected %s"
            % (fmt_hex(e6_start["value"]), fmt_hex(E6_ENTRY))
        )
    if not elf.is_defined_function("_mlir_ciface_vit"):
        check.fail("_mlir_ciface_vit is not a defined function")
    if not elf.is_defined_function("e6_main"):
        check.fail("e6_main is not a defined function")
    for problem in model_cache_layout_problems(elf):
        check.fail(problem)

    nh_section = elf.section(".text.nh_start")
    if nh_section is None or nh_section["addr"] != NH_ENTRY:
        check.fail(".text.nh_start section is missing or not at %s" % fmt_hex(NH_ENTRY))
    e6_section = elf.section(".text.e6_start")
    if e6_section is None or e6_section["addr"] != E6_ENTRY:
        check.fail(".text.e6_start section is missing or not at %s" % fmt_hex(E6_ENTRY))

    cases = list(context.get("distinct_cases") or [])
    if len(cases) != EXPECTED_DISTINCT_CASES:
        check.fail(
            "distinct_cases has %d entries, expected %d"
            % (len(cases), EXPECTED_DISTINCT_CASES)
        )
    missing_cases = [
        case
        for case in cases
        if not elf.is_defined_function("_mlir_ciface_kernel_" + case)
    ]
    if missing_cases:
        check.fail(
            "%d distinct case kernel adapters missing from the ELF: %s"
            % (len(missing_cases), ", ".join(sorted(missing_cases)[:8]))
        )

    replaced = list(context.get("replaced") or [])
    if len(replaced) != EXPECTED_ADAPTERS:
        check.fail(
            "replacement report has %d replaced calls, expected %d"
            % (len(replaced), EXPECTED_ADAPTERS)
        )
    symbols = [record.get("symbol") for record in replaced]
    if len(set(symbols)) != len(symbols):
        check.fail("replacement report has duplicate adapter symbols")
    missing_adapters = [
        symbol
        for symbol in symbols
        if symbol and not elf.is_defined_function("_mlir_ciface_" + symbol)
    ]
    if missing_adapters:
        check.fail(
            "%d of %d adapter functions missing from the ELF: %s"
            % (len(missing_adapters), len(symbols), ", ".join(missing_adapters[:8]))
        )
    graph_calls = (report.get("graphs") or {}).get("forward") or {}
    if graph_calls.get("external_calls") not in (None, len(replaced)):
        check.fail(
            "report external_calls %r != replaced count %d"
            % (graph_calls.get("external_calls"), len(replaced))
        )

    bad_undefined = sorted(
        name
        for name in elf.undefined
        if name not in ALLOWED_UNDEFINED and not TRAP_UNDEFINED_RE.match(name)
    )
    if bad_undefined:
        check.fail(
            "non-allowlisted undefined symbols: %s" % (", ".join(bad_undefined[:20]),)
        )

    check.note(
        elf_class=header.get("elf_class"),
        elf_data=header.get("elf_data"),
        machine=header.get("e_machine"),
        e_type=header.get("e_type"),
        entry=fmt_hex(header.get("e_entry") or 0),
        _start=fmt_hex(start["value"]) if start else None,
        e6_start=fmt_hex(e6_start["value"]) if e6_start else None,
        symbol_count=elf.sym_count,
        distinct_cases=len(cases),
        kernel_adapters_required=len(cases),
        adapters_required=len(symbols),
        undefined=sorted(elf.undefined),
        undefined_allowlisted=len(elf.undefined) - len(bad_undefined),
    )
    return check


# ---------------------------------------------------------------------------
# Check 5: flat binary / manifest / NOLOAD regions
# ---------------------------------------------------------------------------


def check_bin_image(context: dict) -> Check:
    check = Check(
        "bin-image",
        "flat bin size vs loaded PROGBITS extent, manifest hashes, NOLOAD regions absent",
    )
    manifest = context.get("manifest")
    binary = context.get("bin_bytes")
    elf = context.get("elf")
    params = context.get("params_bytes")
    if manifest is None or binary is None or elf is None or elf.error:
        check.fail("manifest.json / vit_model.bin / vit_model.elf unavailable")
        return check
    problems = []

    if manifest.get("schema") != 1:
        problems.append("manifest schema %r != 1" % (manifest.get("schema"),))
    if manifest.get("vit_args") != EXPECTED_VIT_ARGS:
        problems.append(
            "manifest vit_args %r != %d" % (manifest.get("vit_args"), EXPECTED_VIT_ARGS)
        )
    if manifest.get("ciface_args") != EXPECTED_CIFACE_ARGS:
        problems.append(
            "manifest ciface_args %r != %d"
            % (manifest.get("ciface_args"), EXPECTED_CIFACE_ARGS)
        )
    if params is not None:
        if manifest.get("params_bytes") != len(params):
            problems.append(
                "manifest params_bytes %r != %d"
                % (manifest.get("params_bytes"), len(params))
            )
        if manifest.get("params_sha256") != context.get("params_sha256"):
            problems.append("manifest params_sha256 does not match params.bin")
    if manifest.get("bin_bytes") != len(binary):
        problems.append(
            "manifest bin_bytes %r != %d" % (manifest.get("bin_bytes"), len(binary))
        )
    if manifest.get("bin_sha256") != context.get("bin_sha256"):
        problems.append("manifest bin_sha256 does not match vit_model.bin")
    elf_stat = context["artifacts"]["vit_model.elf"]
    if manifest.get("elf_bytes") != elf_stat.get("bytes"):
        problems.append(
            "manifest elf_bytes %r != %d"
            % (manifest.get("elf_bytes"), elf_stat.get("bytes"))
        )
    if manifest.get("elf_sha256") != elf_stat.get("sha256"):
        problems.append("manifest elf_sha256 does not match vit_model.elf")
    manifest_undefined = manifest.get("undefined_symbols")
    if manifest_undefined is not None and sorted(manifest_undefined) != sorted(
        elf.undefined
    ):
        problems.append(
            "manifest undefined_symbols does not match the ELF symbol table"
        )
    objects = context.get("objects")
    if isinstance(objects, list) and manifest.get("objects") != len(objects):
        problems.append(
            "manifest objects %r != objects.json length %d"
            % (manifest.get("objects"), len(objects))
        )

    loaded = elf.allocated_progbits()
    expected_size = None
    loaded_end = None
    if not loaded:
        problems.append("ELF has no allocated PROGBITS sections")
    else:
        lowest = min(section["addr"] for section in loaded)
        loaded_end = max(section["addr"] + section["size"] for section in loaded)
        expected_size = loaded_end - IMAGE_BASE
        if lowest != IMAGE_BASE:
            problems.append(
                "lowest loaded PROGBITS address is %s, expected %s"
                % (fmt_hex(lowest), fmt_hex(IMAGE_BASE))
            )
        if expected_size != len(binary):
            problems.append(
                "vit_model.bin is %d bytes but the loaded PROGBITS extent is %d bytes"
                % (len(binary), expected_size)
            )
        for section in loaded:
            start = section["addr"] - IMAGE_BASE
            inside = section["offset"]
            if start < 0 or start + section["size"] > len(binary):
                problems.append(
                    "section %s lies outside vit_model.bin"
                    % (section["name"] or "<unnamed>")
                )
                continue
            if (
                binary[start : start + section["size"]]
                != elf.data[inside : inside + section["size"]]
            ):
                problems.append(
                    "vit_model.bin content differs from ELF section %s"
                    % (section["name"] or "<unnamed>")
                )

    noload = elf.noload_sections()
    noload_names = {section["name"]: section for section in noload}
    for required in (".model_workspace", ".model_heap", ".bss"):
        if required not in noload_names:
            problems.append("NOLOAD section %s is missing from the ELF" % required)
    for section in noload:
        offset = section["addr"] - IMAGE_BASE
        if offset < len(binary):
            problems.append(
                "NOLOAD section %s starts at bin offset %d, inside vit_model.bin"
                % (section["name"], offset)
            )
    workspace_section = noload_names.get(".model_workspace")
    if workspace_section is not None:
        arg_map = context.get("arg_map") or {}
        expected_workspace = arg_map.get("workspace_bytes")
        if (
            expected_workspace is not None
            and workspace_section["size"] != expected_workspace
        ):
            problems.append(
                ".model_workspace size %d != ARG-MAP workspace_bytes %r"
                % (workspace_section["size"], expected_workspace)
            )
        if loaded_end is not None and workspace_section["addr"] < loaded_end:
            problems.append(
                ".model_workspace starts before the end of the loaded image"
            )

    def embedded(section_name, source_bytes, source_label):
        if source_bytes is None:
            problems.append(
                "cannot verify %s: source %s unavailable" % (section_name, source_label)
            )
            return
        section = elf.section(section_name)
        if section is None:
            problems.append("ELF section %s is missing" % section_name)
            return
        payload = len(source_bytes)
        padded = align_up(payload, ALIGNMENT)
        # The linker script rounds each embedded blob section up to its
        # alignment, so the section may carry zero padding after the payload.
        if section["size"] not in (payload, padded):
            problems.append(
                "ELF section %s is %d bytes, source %s is %d bytes (padded %d)"
                % (section_name, section["size"], source_label, payload, padded)
            )
            return
        in_elf = elf.data[section["offset"] : section["offset"] + section["size"]]
        if in_elf[:payload] != source_bytes:
            problems.append(
                "ELF section %s content differs from %s" % (section_name, source_label)
            )
        elif any(in_elf[payload:]):
            problems.append(
                "ELF section %s alignment padding is not zero" % section_name
            )
        start = section["addr"] - IMAGE_BASE
        if 0 <= start and start + section["size"] <= len(binary):
            image = binary[start : start + section["size"]]
            if image[:payload] != source_bytes:
                problems.append(
                    "vit_model.bin content at %s differs from %s"
                    % (section_name, source_label)
                )
            elif any(image[payload:]):
                problems.append(
                    "vit_model.bin padding at %s is not zero" % section_name
                )
        else:
            problems.append("%s lies outside vit_model.bin" % section_name)

    embedded(".model_params", params, "params.bin")
    pixel_values = context.get("pixel_values")
    reference_logits = context.get("reference_logits")
    embedded(
        ".model_input",
        pixel_values.read_bytes() if pixel_values else None,
        str(pixel_values),
    )
    embedded(
        ".model_reference",
        reference_logits.read_bytes() if reference_logits else None,
        str(reference_logits),
    )

    if problems:
        check.fail(
            "%d bin/manifest problems; first: %s"
            % (len(problems), "; ".join(problems[:6]))
        )

    check.note(
        bin_bytes=len(binary),
        bin_sha256=context.get("bin_sha256"),
        loaded_progbits_extent=expected_size,
        image_base=fmt_hex(IMAGE_BASE),
        loaded_end=fmt_hex(loaded_end) if loaded_end is not None else None,
        noload_sections={
            section["name"]: fmt_hex(section["addr"]) for section in noload
        },
        manifest_audit_status=(manifest.get("audit") or {}).get("status")
        if isinstance(manifest.get("audit"), dict)
        else None,
    )
    return check


# ---------------------------------------------------------------------------
# Check 6: W8A8 derived parameter accounting
# ---------------------------------------------------------------------------


def check_w8a8_parameters(context: dict) -> Check:
    check = Check(
        "w8a8-params",
        "W8A8 parameter accounting: 73 int8 weights + 73 scales derived from 73 replaced linears",
    )
    report = context.get("report")
    parameter_layout = context.get("parameter_layout") or []
    parameter_map = context.get("parameter_map") or {}
    param_roots = [
        root for root in (context.get("roots") or []) if root in parameter_map
    ]
    arg_map = context.get("arg_map") or {}
    arg_kinds = {
        arg.get("name"): arg.get("kind") for arg in (arg_map.get("args") or [])
    }
    if report is None or not parameter_layout:
        check.fail("replacement report / parameter_layout unavailable")
        return check
    linears = list(context.get("w8a8_linears") or [])
    problems = []
    if len(linears) != EXPECTED_W8A8_LINEARS:
        problems.append(
            "w8a8_linears has %d entries, expected %d"
            % (len(linears), EXPECTED_W8A8_LINEARS)
        )
    if report.get("w8a8_linear_count") != EXPECTED_W8A8_LINEARS:
        problems.append(
            "w8a8_linear_count %r != %d"
            % (report.get("w8a8_linear_count"), EXPECTED_W8A8_LINEARS)
        )
    if report.get("w8a8_enabled") is not True:
        problems.append("w8a8_enabled is not true")

    layout_map = {entry.get("name"): entry for entry in parameter_layout}
    original = set(report.get("original_parameters") or [])
    layout_names = set(layout_map)
    weight_names = []
    scale_names = []
    weight_sources = []
    for record in linears:
        weight_name = record.get("weight_param_name")
        scale_name = record.get("scale_param_name")
        weight_source = record.get("weight_param")
        weight_names.append(weight_name)
        scale_names.append(scale_name)
        weight_sources.append(weight_source)
        for label, name in (
            ("weight_param_name", weight_name),
            ("scale_param_name", scale_name),
        ):
            if not name:
                problems.append(
                    "%s is missing for linear %r" % (label, record.get("node"))
                )
                continue
            entry = layout_map.get(name)
            if entry is None:
                problems.append("%s %r is absent from parameter_layout" % (label, name))
                continue
            dtype = LAYOUT_DTYPE.get(str(entry.get("dtype")))
            if label == "weight_param_name" and dtype != "i8":
                problems.append(
                    "derived weight %r has dtype %r, expected int8"
                    % (name, entry.get("dtype"))
                )
            if label == "scale_param_name" and dtype != "f32":
                problems.append(
                    "derived scale %r has dtype %r, expected float32"
                    % (name, entry.get("dtype"))
                )
            if name not in param_roots:
                problems.append("%r is not an @vit parameter root" % (name,))
            if arg_kinds.get(name) != "param":
                problems.append(
                    "%r is not classified as a parameter in ARG-MAP" % (name,)
                )
        if not weight_source:
            problems.append(
                "weight_param is missing for linear %r" % (record.get("node"),)
            )
        elif weight_source not in original:
            problems.append(
                "weight_param %r is not an original imported parameter" % weight_source
            )

    derived = set(weight_names) | set(scale_names)
    if len(derived) != 2 * EXPECTED_W8A8_LINEARS:
        problems.append(
            "expected %d distinct derived parameter names, found %d"
            % (2 * EXPECTED_W8A8_LINEARS, len(derived))
        )
    if len(set(weight_names)) != len(weight_names):
        problems.append("duplicate weight_param_name values")
    if len(set(scale_names)) != len(scale_names):
        problems.append("duplicate scale_param_name values")
    linear_derived = layout_names - original
    if linear_derived != derived:
        problems.append(
            "parameter_layout-minus-original (%d names) does not equal the W8A8 derived names "
            "(%d names)" % (len(linear_derived), len(derived))
        )
    if len(layout_names) != EXPECTED_PARAMS:
        problems.append(
            "parameter_layout has %d entries, expected %d"
            % (len(layout_names), EXPECTED_PARAMS)
        )
    pruned_originals = original - layout_names
    if set(weight_sources) != pruned_originals:
        problems.append(
            "pruned original parameters (%d) do not equal the replaced linear weights (%d)"
            % (len(pruned_originals), len(set(weight_sources)))
        )
    used_originals = original & layout_names
    if len(used_originals) + 2 * EXPECTED_W8A8_LINEARS != len(layout_names):
        problems.append(
            "used originals %d + derived %d != parameter count %d"
            % (len(used_originals), 2 * EXPECTED_W8A8_LINEARS, len(layout_names))
        )
    for record, weight_name, scale_name in zip(linears, weight_names, scale_names):
        shape = record.get("linear_shape") or []
        if len(shape) != 3:
            problems.append(
                "linear %r has malformed linear_shape %r" % (record.get("node"), shape)
            )
            continue
        _rows, n, k = (int(value) for value in shape)
        weight_entry = layout_map.get(weight_name) or {}
        scale_entry = layout_map.get(scale_name) or {}
        if [int(value) for value in (weight_entry.get("shape") or [])] != [n, k]:
            problems.append(
                "weight %r shape %r != [%d, %d]"
                % (weight_name, weight_entry.get("shape"), n, k)
            )
        if [int(value) for value in (scale_entry.get("shape") or [])] != [n]:
            problems.append(
                "scale %r shape %r != [%d]" % (scale_name, scale_entry.get("shape"), n)
            )
    if problems:
        check.fail(
            "%d W8A8 accounting problems; first: %s"
            % (len(problems), "; ".join(problems[:6]))
        )

    check.note(
        w8a8_linears=len(linears),
        derived_weights=len(weight_names),
        derived_scales=len(scale_names),
        derived_total=len(derived),
        used_originals=len(used_originals),
        pruned_originals=len(pruned_originals),
        parameter_layout=len(layout_names),
        original_parameters=len(original),
        sample_weight=weight_names[0] if weight_names else None,
        sample_scale=scale_names[0] if scale_names else None,
    )
    return check


# ---------------------------------------------------------------------------
# Check 7: independent NR ELF audit (buddy check_nr_elf.py)
# ---------------------------------------------------------------------------


def check_elf_audit(context: dict, ctx_args) -> Check:
    check = Check(
        "elf-audit",
        "Buddy NR ELF audit (check_nr_elf.py) on the complete linked image",
    )
    elf_path = context["paths"]["vit_model.elf"]
    if not elf_path.is_file():
        check.fail("vit_model.elf is missing; cannot run the NR ELF audit")
        return check
    if getattr(ctx_args, "skip_audit", False):
        check.skip("NR ELF audit skipped by --skip-audit")
        return check
    buddy = context["buddy"]
    checker = buddy / "examples" / "FPGA-BOSCAME" / "tools" / "check_nr_elf.py"
    if not checker.is_file():
        check.fail("check_nr_elf.py not found at %s" % checker)
        return check
    objdump = ctx_args.objdump
    if not objdump:
        candidate = buddy / "llvm" / "build" / "bin" / "llvm-objdump"
        objdump = (
            str(candidate) if candidate.is_file() else shutil.which("llvm-objdump")
        )
    if not objdump:
        check.fail("no llvm-objdump available for the NR ELF audit")
        return check
    temp_dir = tempfile.mkdtemp(prefix="vit-bin-audit-")
    audit_path = Path(temp_dir) / "elf-audit.json"
    command = [
        sys.executable,
        str(checker),
        str(elf_path),
        "--objdump",
        str(objdump),
        "--output",
        str(audit_path),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=ctx_args.audit_timeout,
        )
    except Exception as error:
        check.fail(
            "failed to run check_nr_elf.py: %s: %s" % (type(error).__name__, error)
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return check
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    audit = None
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text())
        except Exception as error:
            check.fail(
                "cannot parse the audit report: %s: %s" % (type(error).__name__, error)
            )
    shutil.rmtree(temp_dir, ignore_errors=True)

    problems = []
    if audit is None:
        problems.append(
            "check_nr_elf.py produced no audit JSON (rc=%d)" % result.returncode
        )
    else:
        if audit.get("status") != "PASS":
            errors = audit.get("errors") or []
            problems.append(
                "audit status %r with %d errors: %s"
                % (
                    audit.get("status"),
                    len(errors),
                    "; ".join(str(item) for item in errors[:10]),
                )
            )
        expected_elf_sha = context["artifacts"]["vit_model.elf"].get("sha256")
        if audit.get("elf_sha256") != expected_elf_sha:
            problems.append("audit elf_sha256 does not match vit_model.elf")
        manifest = context.get("manifest") or {}
        manifest_audit = manifest.get("audit") or {}
        manifest_status = (
            manifest_audit.get("status") if isinstance(manifest_audit, dict) else None
        )
        if manifest_status != audit.get("status"):
            problems.append(
                "manifest records audit status %r but the independent audit says %r"
                % (manifest_status, audit.get("status"))
            )
        if isinstance(manifest_audit, dict) and manifest_audit.get(
            "elf_sha256"
        ) not in (
            None,
            expected_elf_sha,
        ):
            problems.append("manifest audit elf_sha256 does not match vit_model.elf")
    if result.returncode != 0 and audit is not None and audit.get("status") == "PASS":
        problems.append(
            "check_nr_elf.py exited %d although its report says PASS"
            % result.returncode
        )
    if problems:
        check.fail(
            "%d audit problems; first: %s" % (len(problems), "; ".join(problems[:6]))
        )

    rvv = (audit or {}).get("rvv_instructions") or {}
    manifest = context.get("manifest") or {}
    manifest_audit = (
        manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
    )
    check.note(
        checker=str(checker),
        objdump=str(objdump),
        exit_code=result.returncode,
        audit_status=audit.get("status") if audit else None,
        audit_elf_sha256=audit.get("elf_sha256") if audit else None,
        instructions=audit.get("instructions") if audit else None,
        ame_instructions=audit.get("ame_instructions") if audit else None,
        rvv_instructions=sum(rvv.values()),
        executable_sections=audit.get("executable_sections") if audit else None,
        errors=(audit.get("errors") if audit else None),
        stdout=stdout,
        stderr=stderr[-2000:] if stderr else "",
        manifest_audit_status=manifest_audit.get("status"),
    )
    return check


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECK_FUNCTIONS = (
    check_artifacts,
    check_argument_contract,
    check_params_layout,
    check_argument_map,
    check_elf_symbols,
    check_bin_image,
    check_w8a8_parameters,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independently verify the whole-model ViT-B/16 NH/E6 flat image artifacts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V / "build" / "model-bin",
        help="directory holding the produced flat-image artifacts (default: V/build/model-bin)",
    )
    parser.add_argument(
        "--replacement",
        type=Path,
        default=None,
        help="staged replacement directory (default: <output>/inputs/replacement)",
    )
    parser.add_argument(
        "--buddy-root",
        default=None,
        help="buddy-mlir checkout (default: $BUDDY_ROOT or <repo>/buddy-mlir)",
    )
    parser.add_argument(
        "--objdump", default=None, help="llvm-objdump used by the NR ELF audit"
    )
    parser.add_argument(
        "--audit-timeout",
        type=float,
        default=600.0,
        help="NR ELF audit timeout in seconds",
    )
    parser.add_argument(
        "--skip-audit", action="store_true", help="debugging only: skip check 7"
    )
    parser.add_argument(
        "--verify-json",
        type=Path,
        default=None,
        help="VERIFY.json destination (default: <output>/VERIFY.json)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    context = build_context(args)
    output = context["output"]
    verify_path = Path(args.verify_json) if args.verify_json else output / "VERIFY.json"
    started = time.time()
    load_context(context)
    checks = []
    for function in CHECK_FUNCTIONS:
        try:
            checks.append(function(context))
        except Exception:
            failed = Check(function.__name__, "check crashed")
            failed.fail("unexpected exception:\n%s" % traceback.format_exc())
            checks.append(failed)
    try:
        checks.append(check_elf_audit(context, args))
    except Exception:
        failed = Check("elf-audit", "check crashed")
        failed.fail("unexpected exception:\n%s" % traceback.format_exc())
        checks.append(failed)

    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    reasons = []
    for check in checks:
        reasons.extend("%s: %s" % (check.id, reason) for reason in check.reasons)

    artifacts = {}
    for name, entry in sorted(context["artifacts"].items()):
        artifacts[name] = {
            "path": entry["path"],
            "exists": entry["exists"],
            "bytes": entry.get("bytes"),
            "sha256": entry.get("sha256"),
        }

    elf = context.get("elf")
    audit_details = checks[-1].details or {}
    key_numbers = {
        "vit_args": len(context.get("roots") or []),
        "params": len(context.get("parameter_map") or {}),
        "workspaces": len(context.get("workspace_roots") or []),
        "params_bytes": context.get("params_size"),
        "params_sha256": context.get("params_sha256"),
        "bin_bytes": context.get("bin_size"),
        "bin_sha256": context.get("bin_sha256"),
        "elf_bytes": context["artifacts"].get("vit_model.elf", {}).get("bytes"),
        "elf_sha256": context["artifacts"].get("vit_model.elf", {}).get("sha256"),
        "workspace_bytes": (context.get("arg_map") or {}).get("workspace_bytes"),
        "adapters_required": len(context.get("replaced") or []),
        "distinct_cases": len(context.get("distinct_cases") or []),
        "w8a8_linears": len(context.get("w8a8_linears") or []),
        "undefined_symbols": (
            sorted(elf.undefined) if elf is not None and not elf.error else None
        ),
        "audit_status": audit_details.get("audit_status"),
        "audit_instructions": audit_details.get("instructions"),
        "audit_ame_instructions": audit_details.get("ame_instructions"),
        "audit_rvv_instructions": audit_details.get("rvv_instructions"),
    }
    report = {
        "schema": 1,
        "tool": "examples/BOSCAMEFPGA/vit-base-patch16-224/tools/check_model_bin.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - started, 3),
        "status": status,
        "reasons": reasons,
        "checks": [check.as_dict() for check in checks],
        "key_numbers": key_numbers,
        "artifacts": artifacts,
        "inputs": {
            key: (str(value) if value is not None else None)
            for key, value in (context.get("input_paths") or {}).items()
        },
        "scope": SCOPE,
    }
    verify_path.parent.mkdir(parents=True, exist_ok=True)
    verify_path.write_text(json.dumps(report, indent=2) + "\n")

    for check in checks:
        print("[%s] %s: %s" % (check.status, check.id, check.title))
        for reason in check.reasons:
            print("    - %s" % (reason,))
    print()
    print("VERIFY: %s" % status)
    print("  artifacts: %s" % verify_path)
    print(
        "  bin: %s bytes sha256 %s"
        % (key_numbers["bin_bytes"], key_numbers["bin_sha256"])
    )
    print(
        "  elf: %s bytes sha256 %s"
        % (key_numbers["elf_bytes"], key_numbers["elf_sha256"])
    )
    print(
        "  vit args: %s (%s params, %s workspaces); workspace bytes %s"
        % (
            key_numbers["vit_args"],
            key_numbers["params"],
            key_numbers["workspaces"],
            key_numbers["workspace_bytes"],
        )
    )
    print("  undefined symbols: %r" % (key_numbers["undefined_symbols"],))
    print("  NR ELF audit: %r" % (key_numbers["audit_status"],))
    if reasons:
        print("  reasons: %d" % len(reasons))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
