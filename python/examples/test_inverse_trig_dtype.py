from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import triton.language as tl


@pytest.mark.parametrize("name", ["asin", "acos", "atan"])
@pytest.mark.parametrize("member", [False, True], ids=["function", "member"])
@pytest.mark.parametrize(
    "dtype", [tl.float16, tl.bfloat16, tl.int32, tl.float32, tl.float64]
)
def test_inverse_trig_dtype(name, member, dtype):
    # No IR is needed: the dtype guard must run before the builder is called.
    handle = object()
    x = tl.tensor(handle, dtype)
    create = Mock(return_value=handle)
    builder = SimpleNamespace(**{f"create_{name}": create})
    semantic = SimpleNamespace(builder=builder, to_tensor=lambda value: value)
    fn = getattr(x, name) if member else getattr(tl.math, name)
    args = () if member else (x,)

    if dtype in (tl.float32, tl.float64):
        result = fn(*args, _semantic=semantic)
        assert result.dtype == dtype
        assert result.handle is handle
        create.assert_called_once_with(handle)
    else:
        with pytest.raises(ValueError, match="Expected dtype"):
            fn(*args, _semantic=semantic)
        create.assert_not_called()
