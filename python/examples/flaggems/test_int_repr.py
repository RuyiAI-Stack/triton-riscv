import importlib

import pytest
import torch

from .int_repr import int_repr

# torch reference: aten::int_repr / aten::int_repr.out, both CPU-callable.
# torch.quantize_per_tensor emits a deprecation UserWarning on this build;
# it is unrelated to the operator under test.
pytestmark = [
    pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor"),
    pytest.mark.filterwarnings("ignore:An output with one or more elements"),
]

torch_int_repr = torch.ops.aten.int_repr

_Q_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
QUANTIZED_DTYPES = list(_Q_TO_INT)

SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (64, 48),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim (the strided kernel's rank bound)
]


def quantized_input(shape, qdtype, scale=1.0, zero_point=0):
    payload = torch.randn(shape, dtype=torch.float32)
    return torch.quantize_per_tensor(payload, scale, zero_point, qdtype)


def sentinel_buffer(shape, out_dtype):
    return torch.full(shape, torch.iinfo(out_dtype).max, dtype=out_dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr(shape, qdtype):
    inp = quantized_input(shape, qdtype)
    before = torch_int_repr(inp)

    ref = torch_int_repr(inp)
    res = int_repr(inp)

    assert res.dtype == _Q_TO_INT[qdtype]
    assert res.shape == ref.shape
    assert torch.equal(res, ref)
    # Reading must not modify the quantized storage.
    assert torch.equal(torch_int_repr(inp), before)


@pytest.mark.parametrize(
    "qdtype,layout",
    [
        (qdtype, layout)
        for qdtype in QUANTIZED_DTYPES
        for layout in ("transposed", "offset_slice", "strided_last", "expanded")
    ],
)
def test_int_repr_reads_layout(qdtype, layout):
    inp = quantized_input((6, 7, 8), qdtype)
    if layout == "transposed":
        inp = inp.transpose(0, 1)
    elif layout == "offset_slice":
        inp = inp[1:]
    elif layout == "strided_last":
        inp = inp[..., ::2]
    elif layout == "expanded":
        inp = inp[:1].expand(4, *inp.shape[1:])

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_reads_channels_last(qdtype):
    # channels_last is a 4-D memory format; the storage stays dense so the
    # copy is one flat span while the result keeps the input's strides.
    inp = quantized_input((4, 6, 8, 10), qdtype).to(memory_format=torch.channels_last)
    assert not inp.is_contiguous()

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert res.stride() == inp.stride()
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_result_is_independent_of_input_storage(qdtype):
    inp = quantized_input((2, 19, 7), qdtype)
    expected = torch_int_repr(inp)

    res = int_repr(inp)
    assert torch.equal(res, expected)

    # Filling the result must leave the quantized input untouched.
    res.fill_(3)
    assert torch.equal(torch_int_repr(inp), expected)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out(shape, qdtype):
    inp = quantized_input(shape, qdtype)
    out_dtype = _Q_TO_INT[qdtype]
    before = torch_int_repr(inp)

    ref = torch_int_repr.out(inp, out=torch.empty(shape, dtype=out_dtype))
    buf = sentinel_buffer(shape, out_dtype)
    res = int_repr(inp, out=buf)

    assert res is buf
    assert torch.equal(res, ref)
    assert torch.equal(torch_int_repr(inp), before)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out_resizes_buffer(qdtype):
    inp = quantized_input((2, 19, 7), qdtype)
    out_dtype = _Q_TO_INT[qdtype]
    before = torch_int_repr(inp)

    ref = torch_int_repr.out(inp, out=torch.empty((3,), dtype=out_dtype))
    buf = sentinel_buffer((3,), out_dtype)
    res = int_repr(inp, out=buf)

    assert res is buf
    assert res.shape == (2, 19, 7)
    assert torch.equal(res, ref)
    assert torch.equal(torch_int_repr(inp), before)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out_writes_through_strided_buffer(qdtype):
    inp = quantized_input((4, 5, 6), qdtype)
    out_dtype = _Q_TO_INT[qdtype]
    gap = (4, 5, 12)
    sentinel = torch.iinfo(out_dtype).max
    before = torch_int_repr(inp)

    ref_pad = torch.full(gap, sentinel, dtype=out_dtype)
    ref = torch_int_repr.out(inp, out=ref_pad[:, :, ::2])
    pad = torch.full(gap, sentinel, dtype=out_dtype)
    buf = pad[:, :, ::2]
    res = int_repr(inp, out=buf)

    assert res is buf
    assert tuple(buf.stride()) == (60, 12, 2)
    assert torch.equal(res, ref)
    # Only the strided view is written; the parent's padding stays intact.
    assert torch.equal(pad, ref_pad)
    assert torch.equal(torch_int_repr(inp), before)


@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (2, 0, 19, 7)])
@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_degenerate_shape(shape, qdtype):
    inp = quantized_input(shape, qdtype)

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert res.shape == ref.shape
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out_degenerate_shape(qdtype):
    inp = quantized_input((0, 3), qdtype)
    out_dtype = _Q_TO_INT[qdtype]

    ref = torch_int_repr.out(inp, out=torch.empty((0, 3), dtype=out_dtype))
    buf = sentinel_buffer((0, 3), out_dtype)
    res = int_repr(inp, out=buf)

    assert res is buf
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("shape, axis", [((4, 5), 0), ((4, 5), 1), ((3, 7, 5), 2)])
def test_int_repr_per_channel_quantization(qdtype, shape, axis):
    info = torch.iinfo(_Q_TO_INT[qdtype])
    span = info.max - info.min
    channels = shape[axis]
    scales = [2.0**-10, 2.0**-6, 0.5, 1.0, 4.0, 64.0]
    scales = [scales[i % len(scales)] for i in range(channels)]
    zero_points = [
        int(info.min + (span * i) // max(channels - 1, 1)) for i in range(channels)
    ]
    payload = torch.randn(shape, dtype=torch.float32)
    inp = torch.quantize_per_channel(
        payload,
        torch.tensor(scales, dtype=torch.float64),
        torch.tensor(zero_points, dtype=torch.int64),
        axis,
        qdtype,
    )
    before = torch_int_repr(inp)

    ref = torch_int_repr(inp)
    res = int_repr(inp)

    assert torch.equal(res, ref)
    assert torch.equal(torch_int_repr(inp), before)


@pytest.mark.parametrize(
    "qdtype, scale, fraction",
    [
        (qdtype, scale, fraction)
        for qdtype in QUANTIZED_DTYPES
        for scale, fraction in [
            (2.0**-20, 0.0),
            (2.0**20, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]
    ],
)
def test_int_repr_extreme_qparams(qdtype, scale, fraction):
    info = torch.iinfo(_Q_TO_INT[qdtype])
    zero_point = int(round(info.min + fraction * (info.max - info.min)))
    inp = quantized_input((6, 7, 8), qdtype, scale, zero_point)

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("rule", ["saturate_high", "saturate_low", "zero"])
def test_int_repr_storage_boundaries(qdtype, rule):
    info = torch.iinfo(_Q_TO_INT[qdtype])
    value = {"saturate_high": info.max, "saturate_low": info.min, "zero": 0.0}[rule]
    payload = torch.full((6, 7, 8), float(value), dtype=torch.float32)
    inp = torch.quantize_per_tensor(payload, 1.0, 0, qdtype)

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert torch.equal(res, ref)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
@pytest.mark.parametrize("scenario", ["nan", "inf", "mixed"])
def test_int_repr_quantized_special_values(qdtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    payload = torch.tensor(payloads[scenario], dtype=torch.float32)
    inp = torch.quantize_per_tensor(payload, 1.0, 0, qdtype)

    ref = torch_int_repr(inp)
    res = int_repr(inp)
    assert torch.equal(res, ref)


@pytest.mark.parametrize(
    "plain_dtype",
    [torch.int8, torch.uint8, torch.float32, torch.float16, torch.int32, torch.bool],
)
def test_int_repr_rejects_non_quantized_dtype(plain_dtype):
    inp = torch.zeros((2, 19, 7), dtype=plain_dtype)
    with pytest.raises(RuntimeError):
        int_repr(inp)


def test_int_repr_rejects_non_tensor_argument():
    with pytest.raises(TypeError):
        int_repr([1, 2, 3])


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_out_rejects_mismatched_dtype(qdtype):
    inp = quantized_input((2, 19, 7), qdtype)
    buf = torch.empty((2, 19, 7), dtype=torch.int16)
    with pytest.raises(RuntimeError):
        int_repr(inp, out=buf)


@pytest.mark.parametrize("qdtype", QUANTIZED_DTYPES)
def test_int_repr_flat_triton_path(qdtype, monkeypatch):
    mod = importlib.import_module("flaggems.int_repr")

    launches = []
    real_run = mod._int_repr_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._int_repr_flat_kernel, "run", spy_run)
    inp = quantized_input((4, 64), qdtype)
    res = mod.int_repr(inp)
    assert len(launches) >= 1
    assert torch.equal(res, torch_int_repr(inp))


def test_int_repr_strided_triton_path(monkeypatch):
    # A zero-stride expand cannot take the flat path and must drive the
    # rank-decomposing kernel.
    mod = importlib.import_module("flaggems.int_repr")

    launches = []
    real_run = mod._int_repr_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._int_repr_strided_kernel, "run", spy_run)
    base = quantized_input((1, 33), torch.quint8)
    inp = base.expand(8, 33)
    res = mod.int_repr(inp)
    assert len(launches) == 1
    assert torch.equal(res, torch_int_repr(inp))
