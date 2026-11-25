"""Validation utilities shared by the B12--B14 differential tests."""

import torch
from triton.runtime.jit import JITFunction


def check_case(monkeypatch, runner, name, variant):
    launches = []
    original = JITFunction.run

    def record(kernel, *args, **kwargs):
        launches.append(kernel.__name__)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", record)
    actual, expected = runner(name, variant)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4, equal_nan=True)
    assert launches or (name == "embedding_sparse_backward" and variant == 0), (
        f"{name} did not execute a Triton kernel"
    )
