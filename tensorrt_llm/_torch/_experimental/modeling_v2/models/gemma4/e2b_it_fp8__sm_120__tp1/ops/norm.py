# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 RMSNorm with an optional residual add after it, in one Triton kernel.

``out = residual + bf16(x * rsqrt(mean(x^2) + eps) * weight)`` over the last dimension, bf16 in and out, fp32 inside.
Gemma-4 normalizes *after* each sublayer and then adds the residual (the "sandwich" norm), so the add follows the
normalization; the normalized value is rounded to bf16 before the add, as the unfused form does. ``weight`` None is
the weightless norm (V). Leading dimensions are flattened into rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    res_ptr,
    out_ptr,
    n_cols,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    if HAS_WEIGHT:
        y = y * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y.to(tl.bfloat16)
    if HAS_RESIDUAL:
        y = (
            tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            + y.to(tl.float32)
        ).to(tl.bfloat16)
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


class RmsNorm(OpWrapper):
    """See the module docstring."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(
            "residual stream with residual add",
            {"rows": 48, "cols": 1536, "weight": True, "residual": True},
        ),
        Cell(
            "prefill residual stream",
            {"rows": 4096, "cols": 1536, "weight": True, "residual": False},
        ),
        Cell(
            "q/k heads, head_dim 512", {"rows": 64, "cols": 512, "weight": True, "residual": False}
        ),
        Cell(
            "weightless V norm, head_dim 256",
            {"rows": 4096, "cols": 256, "weight": False, "residual": False},
        ),
        Cell(
            "assistant width 256 with residual",
            {"rows": 8, "cols": 256, "weight": True, "residual": True},
        ),
    )

    def __call__(self, x, weight, residual, eps):
        n_cols = x.shape[-1]
        x = x.contiguous()
        out = torch.empty_like(x, dtype=torch.bfloat16)
        rows = x.numel() // n_cols
        if rows == 0:
            return out
        _rms_norm_kernel[(rows,)](
            x,
            weight if weight is not None else x,
            residual.contiguous() if residual is not None else x,
            out,
            n_cols,
            eps,
            HAS_WEIGHT=weight is not None,
            HAS_RESIDUAL=residual is not None,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=4 if n_cols <= 512 else 8,
        )
        return out

    def reference(self, x, weight, residual, eps):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            y = y * weight.float()
        y = y.to(torch.bfloat16)
        if residual is not None:
            y = (residual.float() + y.float()).to(torch.bfloat16)
        return y


rms_norm = RmsNorm()
