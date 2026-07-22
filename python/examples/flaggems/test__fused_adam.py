import pytest
import torch

from ._fused_adam import _fused_adam, _fused_adam_


def _adam_state():
    param = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32, device="cpu")
    grad = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32, device="cpu")
    return {
        "params": [param],
        "grads": [grad],
        "exp_avgs": [torch.zeros_like(param)],
        "exp_avg_sqs": [torch.zeros_like(param)],
        "max_exp_avg_sqs": [torch.zeros_like(param)],
        "state_steps": [torch.tensor(1.0, device="cpu")],
    }


def _clone_adam_state(state):
    return {name: [value.clone() for value in values] for name, values in state.items()}


def _assert_adam_state_close(actual, expected, amsgrad):
    for name in ("params", "exp_avgs", "exp_avg_sqs"):
        for actual_tensor, expected_tensor in zip(actual[name], expected[name]):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4
            )
    if amsgrad:
        for actual_tensor, expected_tensor in zip(
            actual["max_exp_avg_sqs"], expected["max_exp_avg_sqs"]
        ):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4
            )


def test_fused_adam_single_step():
    actual = _adam_state()
    expected = _clone_adam_state(actual)

    _fused_adam(
        actual["params"],
        actual["grads"],
        actual["exp_avgs"],
        actual["exp_avg_sqs"],
        [],
        actual["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )
    torch._fused_adam_(
        expected["params"],
        expected["grads"],
        expected["exp_avgs"],
        expected["exp_avg_sqs"],
        [],
        expected["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )

    _assert_adam_state_close(actual, expected, amsgrad=False)


def test_fused_adam_inplace_entry_returns_none():
    actual = _adam_state()
    expected = _clone_adam_state(actual)

    for step, grad in enumerate(
        (torch.tensor([0.1, -0.2, 0.3]), torch.tensor([-0.4, 0.5, -0.6])),
        start=1,
    ):
        actual["grads"][0].copy_(grad)
        expected["grads"][0].copy_(grad)
        actual["state_steps"][0].fill_(step)
        expected["state_steps"][0].fill_(step)
        ret = _fused_adam_(
            actual["params"],
            actual["grads"],
            actual["exp_avgs"],
            actual["exp_avg_sqs"],
            actual["max_exp_avg_sqs"],
            actual["state_steps"],
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=True,
            maximize=False,
        )
        torch._fused_adam_(
            expected["params"],
            expected["grads"],
            expected["exp_avgs"],
            expected["exp_avg_sqs"],
            expected["max_exp_avg_sqs"],
            expected["state_steps"],
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=True,
            maximize=False,
        )

    assert ret is None
    _assert_adam_state_close(actual, expected, amsgrad=True)


@pytest.mark.parametrize(
    "weight_decay, maximize, use_grad_scale",
    [(0.1, False, False), (0.0, True, False), (0.0, False, True)],
)
def test_fused_adam_options_match_torch(weight_decay, maximize, use_grad_scale):
    actual = _adam_state()
    expected = _clone_adam_state(actual)
    grad_scale = torch.tensor(2.0, dtype=torch.float32, device="cpu")
    grad_scale = grad_scale if use_grad_scale else None

    _fused_adam(
        actual["params"],
        actual["grads"],
        actual["exp_avgs"],
        actual["exp_avg_sqs"],
        [],
        actual["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=weight_decay,
        eps=1e-8,
        amsgrad=False,
        maximize=maximize,
        grad_scale=grad_scale,
    )
    torch._fused_adam_(
        expected["params"],
        expected["grads"],
        expected["exp_avgs"],
        expected["exp_avg_sqs"],
        [],
        expected["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=weight_decay,
        eps=1e-8,
        amsgrad=False,
        maximize=maximize,
        grad_scale=grad_scale,
    )

    _assert_adam_state_close(actual, expected, amsgrad=False)


def test_fused_adam_found_inf_skips_update():
    actual = _adam_state()
    expected = _clone_adam_state(actual)
    found_inf = torch.ones((), dtype=torch.float32, device="cpu")

    ret = _fused_adam_(
        actual["params"],
        actual["grads"],
        actual["exp_avgs"],
        actual["exp_avg_sqs"],
        [],
        actual["state_steps"],
        found_inf=found_inf,
    )
    torch._fused_adam_(
        expected["params"],
        expected["grads"],
        expected["exp_avgs"],
        expected["exp_avg_sqs"],
        [],
        expected["state_steps"],
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        found_inf=found_inf,
    )

    assert ret is None
    _assert_adam_state_close(actual, expected, amsgrad=False)


def test_fused_adam_empty_and_multiblock_parameters_match_torch():
    sizes = (0, 4099)
    actual = {
        "params": [
            torch.linspace(-1.0, 1.0, size, dtype=torch.float32, device="cpu")
            for size in sizes
        ],
        "grads": [
            torch.linspace(1.0, -1.0, size, dtype=torch.float32, device="cpu")
            for size in sizes
        ],
        "exp_avgs": [
            torch.zeros(size, dtype=torch.float32, device="cpu") for size in sizes
        ],
        "exp_avg_sqs": [
            torch.zeros(size, dtype=torch.float32, device="cpu") for size in sizes
        ],
        "max_exp_avg_sqs": [],
        "state_steps": [torch.tensor(2.0, device="cpu") for _ in sizes],
    }
    expected = _clone_adam_state(actual)

    _fused_adam_(
        actual["params"],
        actual["grads"],
        actual["exp_avgs"],
        actual["exp_avg_sqs"],
        [],
        actual["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.1,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )
    torch._fused_adam_(
        expected["params"],
        expected["grads"],
        expected["exp_avgs"],
        expected["exp_avg_sqs"],
        [],
        expected["state_steps"],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.1,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )

    _assert_adam_state_close(actual, expected, amsgrad=False)
