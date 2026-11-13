#!/usr/bin/env python3
"""Check graph coverage and exact XLNet semantics with a random one-layer HF fixture."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory  # noqa: E402 - local modules require the example path
from model_graph import operator_names  # noqa: E402 - local modules require the example path


def main():
    names = {c["name"] for c in inventory()}
    missing = set(operator_names()) - names
    if missing:
        raise AssertionError("missing graph kernels: " + str(missing))
    import torch
    from model_reference import forward
    from transformers import XLNetConfig, XLNetLMHeadModel

    torch.set_num_threads(2)
    torch.manual_seed(71)
    cfg = XLNetConfig(**json.loads((ROOT / "model/config.json").read_text()))
    # Preserve model widths/heads/activation/eps/vocabulary; shorten only repeated layers in this fixture.
    cfg.n_layer = 1
    model = XLNetLMHeadModel(cfg).eval()
    ids = torch.arange(128).reshape(1, 128) * 7
    records = []
    with torch.no_grad():
        for use_segments in (False, True):
            mask = torch.ones(1, 128, dtype=torch.long)
            mask[:, :11] = 0
            seg = torch.zeros_like(ids)
            seg[:, 64:] = 1
            seg[:, -1] = 2
            segments = seg if use_segments else None
            actual = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=segments,
                use_mems=False,
                output_hidden_states=True,
            )
            hidden, logits = forward(model, ids, mask, segments)
            torch.testing.assert_close(
                hidden, actual.hidden_states[-1], atol=6e-6, rtol=6e-6
            )
            torch.testing.assert_close(logits, actual.logits, atol=1e-5, rtol=1e-5)
            records.append(
                {
                    "segments": use_segments,
                    "hidden_max_abs": float(
                        (hidden - actual.hidden_states[-1]).abs().max()
                    ),
                    "logits_max_abs": float((logits - actual.logits).abs().max()),
                }
            )
    report = {
        "status": "PASS",
        "operator_count": len(names),
        "routed_operators": len(operator_names()),
        "fixture_layers": 1,
        "checkpoint_loaded": False,
        "emitted_execution": False,
        "tests": records,
    }
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build/modelcheck-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
