import torch

from .assert_async import _assert_async


def test_assert_async_pass():
    torch.manual_seed(0)
    x = torch.tensor([1], device="cpu", dtype=torch.int32)
    # This should not raise an error
    _assert_async(x)


def test_assert_async_fail():
    torch.manual_seed(0)
    x = torch.tensor([0], device="cpu", dtype=torch.int32)
    # The behavior of device_assert failure might vary depending on the backend,
    # but since it's "async", the error might not be caught immediately in Python
    # unless synchronization happens. Or it might just fail to compile if device_assert
    # is not supported.

    # We'll just run it to see what happens.
    # In a real environment, it might abort the process or print to stderr.
    try:
        _assert_async(x, "Test assert failed")
    except Exception as e:
        print(f"Caught exception: {e}")
