from .detach_copy import _copy_values


def _nested_view_from_jagged_copy(values, offsets):
    return _copy_values(values), _copy_values(offsets)
