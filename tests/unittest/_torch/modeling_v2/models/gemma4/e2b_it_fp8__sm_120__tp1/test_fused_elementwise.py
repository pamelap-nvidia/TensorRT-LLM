# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's fused Triton elementwise ops (RMSNorm + residual, FP8 quantization, gelu-tanh-mul + FP8, q/k norm +
RoPE) against their PyTorch references."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 E4M3 Triton kernels need a Hopper or newer GPU",
)


@pytest.mark.parametrize(
    "shape, weight, residual",
    [
        ((48, 1536), True, True),
        ((4096, 1536), True, False),
        ((64, 8, 512), True, False),
        ((300, 256), False, False),
        ((8, 256), True, True),
    ],
)
def test_rms_norm_matches_reference(shape, weight, residual):
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops.norm import (
        rms_norm,
    )

    gen = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn(*shape, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    w = (
        (torch.rand(shape[-1], device="cuda", generator=gen) + 0.5).to(torch.bfloat16)
        if weight
        else None
    )
    r = torch.randn(*shape, device="cuda", generator=gen).to(torch.bfloat16) if residual else None
    out = rms_norm(x, w, r, 1e-6)
    ref = rms_norm.reference(x, w, r, 1e-6)
    bound = 2 * torch.finfo(torch.bfloat16).eps * ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() <= bound


def _elementwise():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops import (
        elementwise,
    )

    return elementwise


def _fp8_codes_close(a, b, max_mismatch=1e-3):
    """FP8 codes equal except for a few rounding-boundary elements (the kernels compute in a different order)."""
    mismatch = (a.view(torch.uint8) != b.view(torch.uint8)).float().mean().item()
    assert mismatch <= max_mismatch, mismatch
    assert (
        a.float() - b.float()
    ).abs().max().item() <= 32.0  # one E4M3 step at the top of the range


@pytest.mark.parametrize("shape", [(4096, 1536), (7, 2048)])
def test_quantize_fp8_matches_reference(shape):
    ew = _elementwise()
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(*shape, device="cuda", generator=gen) * 4).to(torch.bfloat16)
    scale = torch.tensor([0.011], device="cuda")
    _fp8_codes_close(ew.quantize_fp8(x, scale), ew.quantize_fp8.reference(x, scale))


@pytest.mark.parametrize("shape", [(256, 12288), (33, 24576)])
def test_gelu_tanh_mul_fp8_matches_reference(shape):
    ew = _elementwise()
    gen = torch.Generator(device="cuda").manual_seed(1)
    gate_up = (torch.randn(*shape, device="cuda", generator=gen) * 2).to(torch.bfloat16)
    scale = torch.tensor([0.02], device="cuda")
    _fp8_codes_close(
        ew.gelu_tanh_mul_fp8(gate_up, scale), ew.gelu_tanh_mul_fp8.reference(gate_up, scale), 5e-3
    )


@pytest.mark.parametrize(
    "heads, head_dim, row_width", [(8, 256, 2560), (1, 512, 5120), (8, 512, 4096)]
)
def test_qk_norm_rope_matches_reference(heads, head_dim, row_width):
    ew = _elementwise()
    gen = torch.Generator(device="cuda").manual_seed(2)
    tokens = 300
    rows = torch.randn(tokens, row_width, device="cuda", generator=gen).to(torch.bfloat16)
    x = rows[:, : heads * head_dim].view(
        tokens, heads, head_dim
    )  # a strided slice of a fused qkv row
    weight = (torch.rand(head_dim, device="cuda", generator=gen) + 0.5).to(torch.bfloat16)
    ang = torch.arange(6000, device="cuda", dtype=torch.float32)[:, None] * torch.rand(
        head_dim // 2, device="cuda"
    )
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    positions = torch.randint(0, 6000, (tokens,), device="cuda", generator=gen, dtype=torch.int32)
    out = ew.qk_norm_rope(x, weight, cos, sin, positions, 1e-6)
    ref = ew.qk_norm_rope.reference(x, weight, cos, sin, positions, 1e-6)
    assert (out.float() - ref.float()).abs().max().item() <= 2e-2
