from .detach_copy import _copy_values


def _nested_from_padded_and_nested_example(padded, example):
    _ = example
    return _copy_values(padded)
