"""Scalar Philox-4x32-10 and Box--Muller oracle, independent of Triton."""

import math
import struct

import torch


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def normal(n, seed, offset, mean, std):
    mask = (1 << 32) - 1
    values = []
    for counter in range(offset, offset + (n + 3) // 4):
        a, b, c, d = counter & mask, counter >> 32, 0, 0
        k0, k1 = seed & mask, seed >> 32
        for _ in range(10):
            p0 = 0xD2511F53 * a
            p1 = 0xCD9E8D57 * c
            a, b, c, d = (
                ((p1 >> 32) ^ b ^ k0) & mask,
                p1 & mask,
                ((p0 >> 32) ^ d ^ k1) & mask,
                p0 & mask,
            )
            k0 = (k0 + 0x9E3779B9) & mask
            k1 = (k1 + 0xBB67AE85) & mask
        u = [_f32(_f32(v) * 2**-32 + 2**-33) for v in (a, b, c, d)]
        for i in (0, 2):
            radius = math.sqrt(-2 * math.log(u[i]))
            theta = 2 * math.pi * u[i + 1]
            values.extend(
                (
                    mean + std * radius * math.cos(theta),
                    mean + std * radius * math.sin(theta),
                )
            )
    return torch.tensor(values[:n], dtype=torch.float32)
