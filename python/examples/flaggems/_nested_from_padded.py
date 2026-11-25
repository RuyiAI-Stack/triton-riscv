from .detach_copy import _copy_values


def _nested_from_padded(padded, offsets):
    return _copy_values(padded.reshape(-1)[: int(offsets[-1])])
