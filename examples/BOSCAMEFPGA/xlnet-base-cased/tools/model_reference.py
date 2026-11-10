"""Independent host FP32 XLNet content-stream decomposition.

Used only to check graph semantics against Transformers, never as device output.
"""

import math

import torch
import torch.nn.functional as F


def forward(model, input_ids, attention_mask=None, token_type_ids=None, lm_head=True):
    if input_ids.shape != (1, 128):
        raise ValueError("fixed specialization requires input_ids[1,128]")
    h = model.transformer.word_embedding(input_ids).squeeze(0)
    seq, width = h.shape
    pos = torch.arange(seq, -seq, -1, dtype=h.dtype, device=h.device)
    inv_freq = 1 / (
        10000 ** (torch.arange(0, width, 2, dtype=h.dtype, device=h.device) / width)
    )
    sinusoid = pos[:, None] * inv_freq[None, :]
    relative = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
    if attention_mask is None:
        blocked = torch.zeros((seq, seq), dtype=torch.bool, device=h.device)
    else:
        blocked = (attention_mask.reshape(1, seq) == 0).expand(seq, seq) & ~torch.eye(
            seq, dtype=torch.bool, device=h.device
        )
    seg = (
        None
        if token_type_ids is None
        else token_type_ids.reshape(seq, 1) != token_type_ids.reshape(1, seq)
    )
    for layer in model.transformer.layer:
        a = layer.rel_attn

        # Checkpoint q/k/v/r are [input, head, dim]; the physical i8 right matrix is their flatten transpose.
        def project(x, w):
            return (
                (x @ w.reshape(width, width))
                .reshape(x.shape[0], 12, 64)
                .permute(1, 0, 2)
            )

        q, k, v, r = (
            project(h, a.q),
            project(h, a.k),
            project(h, a.v),
            project(relative, a.r),
        )
        ac = (q + a.r_w_bias[:, None, :]) @ k.transpose(1, 2)
        bd = (q + a.r_r_bias[:, None, :]) @ r.transpose(1, 2)
        bd = bd.reshape(12, 2 * seq, seq)[:, 1:, :].reshape(12, seq, 2 * seq - 1)[
            :, :, :seq
        ]
        if seg is None:
            ef = 0
        else:
            seg_weight = a.seg_embed.permute(1, 0, 2)
            ef2 = (q + a.r_s_bias[:, None, :]) @ seg_weight.transpose(1, 2)
            ef = torch.gather(ef2, 2, seg[None].expand(12, seq, seq).long())
        score = ((ac + bd) + ef) / math.sqrt(64)
        score = score - blocked[None].to(h.dtype) * 1e30
        probability = score.softmax(dim=-1)
        context = (probability @ v).permute(1, 0, 2).reshape(seq, width)
        # o[output,head,dim] already has physical [N,K] orientation.
        attn_output = context @ a.o.reshape(width, width).T
        h = F.layer_norm(
            h + attn_output,
            (width,),
            a.layer_norm.weight,
            a.layer_norm.bias,
            a.layer_norm.eps,
        )
        ff = layer.ff
        z = F.gelu(F.linear(h, ff.layer_1.weight, ff.layer_1.bias), approximate="none")
        z = F.linear(z, ff.layer_2.weight, ff.layer_2.bias)
        h = F.layer_norm(
            h + z, (width,), ff.layer_norm.weight, ff.layer_norm.bias, ff.layer_norm.eps
        )
    hidden = h.unsqueeze(0)
    return (
        (hidden, F.linear(hidden, model.lm_loss.weight, model.lm_loss.bias))
        if lm_head
        else hidden
    )
