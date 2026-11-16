# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused elementwise ops of the layer-by-layer (prefill) path, in Triton: one pass over memory each.

* ``quantize_fp8(x, scale)``: ModelOpt static per-tensor activation quantization, ``e4m3(clamp(x / scale, +-448))``.
* ``gelu_tanh_mul_fp8(gate_up, scale)``: ``gelu_tanh(gate) * up`` rounded to bf16 (as the unfused form), then
  quantized for the FP8 down projection.
* ``qk_norm_rope(x, weight, cos, sin, positions)``: per-head RMSNorm with weight, then NeoX rotate-half RoPE from
  fp32 cos / sin tables, on a ``[T, H, D]`` view whose rows may be strided (the q or k slice of a fused qkv).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

FP8_MAX = 448.0


@triton.jit
def _quantize_kernel(x_ptr, out_ptr, scale_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.clamp(x / tl.load(scale_ptr), -448.0, 448.0)
    tl.store(out_ptr + offs, y.to(tl.float8e4nv), mask=mask)


@triton.jit
def _gelu_mul_quant_kernel(gu_ptr, out_ptr, scale_ptr, inter, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    g = tl.load(gu_ptr + row * 2 * inter + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * inter + inter + cols, mask=mask, other=0.0).to(tl.float32)
    # gelu(x, approximate="tanh") = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    t = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    a = (0.5 * g * (1.0 + t) * u).to(tl.bfloat16).to(tl.float32)
    y = tl.clamp(a / tl.load(scale_ptr), -448.0, 448.0)
    tl.store(out_ptr + row * inter + cols, y.to(tl.float8e4nv), mask=mask)


@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    out_ptr,
    stride_t,
    stride_h,
    num_heads,
    eps,
    HEAD_DIM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    half: tl.constexpr = HEAD_DIM // 2
    d = tl.arange(0, half)
    base = x_ptr + t * stride_t + h * stride_h
    x1 = tl.load(base + d).to(tl.float32)
    x2 = tl.load(base + half + d).to(tl.float32)
    r = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / HEAD_DIM + eps)
    # The normalized value is rounded to bf16 before RoPE, as the unfused form does.
    y1 = (x1 * r * tl.load(w_ptr + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * r * tl.load(w_ptr + half + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    p = tl.load(pos_ptr + t).to(tl.int64)
    c = tl.load(cos_ptr + p * half + d)
    s = tl.load(sin_ptr + p * half + d)
    o = out_ptr + (t * num_heads + h) * HEAD_DIM
    tl.store(o + d, (y1 * c - y2 * s).to(tl.bfloat16))
    tl.store(o + half + d, (y2 * c + y1 * s).to(tl.bfloat16))


class QuantizeFp8(OpWrapper):
    """``e4m3(clamp(x / scale, -448, 448))`` of a contiguous bf16 tensor; ``scale`` a 1-element fp32 tensor."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (Cell("prefill activations", {"shape": (4096, 1536)}),)

    def __call__(self, x, scale):
        x = x.contiguous()
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        n = x.numel()
        if n:
            _quantize_kernel[(triton.cdiv(n, 4096),)](x, out, scale, n, BLOCK=4096)
        return out

    def reference(self, x, scale):
        return (x.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


class GeluTanhMulFp8(OpWrapper):
    """``e4m3(clamp(bf16(gelu_tanh(gate) * up) / scale))`` of a ``[T, 2 * inter]`` gate|up tensor."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell("prefill MLP, intermediate 6144", {"shape": (4096, 12288)}),
        Cell("prefill double-wide MLP, intermediate 12288", {"shape": (4096, 24576)}),
    )

    def __call__(self, gate_up, scale):
        gate_up = gate_up.contiguous()
        rows, width = gate_up.shape
        inter = width // 2
        out = torch.empty(rows, inter, dtype=torch.float8_e4m3fn, device=gate_up.device)
        if rows:
            _gelu_mul_quant_kernel[(rows, triton.cdiv(inter, 2048))](
                gate_up, out, scale, inter, BLOCK=2048
            )
        return out

    def reference(self, gate_up, scale):
        inter = gate_up.shape[-1] // 2
        a = (
            torch.nn.functional.gelu(gate_up[:, :inter].float(), approximate="tanh")
            * gate_up[:, inter:].float()
        )
        return (
            (a.to(torch.bfloat16).float() / scale.float())
            .clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
        )


class QkNormRope(OpWrapper):
    """Per-head RMSNorm (``weight [D]``) then RoPE at ``positions [T]`` of ``x [T, H, D]`` (last dim contiguous);
    returns a contiguous bf16 ``[T, H, D]``. ``cos`` / ``sin``: fp32 ``[max_positions, D / 2]``."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell("sliding q: 8 heads x 256 in a fused qkv row", {"heads": 8, "head_dim": 256}),
        Cell("full k: 1 head x 512 in a fused qkv row", {"heads": 1, "head_dim": 512}),
    )

    def __call__(self, x, weight, cos, sin, positions, eps):
        num_tokens, num_heads, head_dim = x.shape
        assert x.stride(-1) == 1
        out = torch.empty(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=x.device)
        if num_tokens:
            _qk_norm_rope_kernel[(num_tokens, num_heads)](
                x, weight, cos, sin, positions, out, x.stride(0), x.stride(1), num_heads, eps,
                HEAD_DIM=head_dim, num_warps=4,
            )  # fmt: skip
        return out

    def reference(self, x, weight, cos, sin, positions, eps):
        xf = x.float()
        y = (
            (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float())
            .to(torch.bfloat16)
            .float()
        )
        c, s = cos[positions.long()].unsqueeze(1), sin[positions.long()].unsqueeze(1)
        half = x.shape[-1] // 2
        y1, y2 = y[..., :half], y[..., half:]
        return torch.cat([y1 * c - y2 * s, y2 * c + y1 * s], -1).to(torch.bfloat16)


quantize_fp8 = QuantizeFp8()
gelu_tanh_mul_fp8 = GeluTanhMulFp8()
qk_norm_rope = QkNormRope()
