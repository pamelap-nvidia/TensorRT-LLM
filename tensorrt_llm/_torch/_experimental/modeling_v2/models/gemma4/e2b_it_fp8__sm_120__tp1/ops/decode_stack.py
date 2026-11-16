# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gemma-4 E2B decoder stack in one launch: PLE projection, 35 decoder layers and the final RMSNorm.

For ``R <= 8`` requests of ``Q <= 8`` new tokens each (``T = R * Q``), at positions ``cache_lens[r] + j``:

    hidden_states     [T, 1536] bf16   embed_tokens(ids) * bf16(sqrt(1536))
    ple_token_embeds  [T, 35, 256] bf16 embed_tokens_per_layer(ids) * 16
    -> out            [T, 1536] bf16   final RMSNorm of the last layer (the LM head is not part of the op)

and the new K/V rows of the 15 KV-producing layers are written in place, FP8, into the engine's paged caches.

Layer kinds (every fifth layer full attention; layers 15-34 are KV-shared and have a double-wide MLP):

    A  layers 0-3, 5-8, 10-13      sliding, head_dim 256, fused FP8 qkv
    B  layers 4, 9, 14             full,    head_dim 512, fused FP8 qkv
    C  layers 15-18, ..., 30-33    sliding, BF16 q_proj, reads the KV of layer 13
    D  layers 19, 24, 29, 34       full,    BF16 q_proj, reads the KV of layer 14

FP8 linears are ModelOpt per-tensor static: ``x_q = e4m3(clamp(x / in_scale))``, ``y = bf16(x_q @ W_q^T * in_scale
* w_scale)``; the fused qkv / gate_up carry one (max) scale pair (see weights.py). Attention: scale 1 after q/k
RMSNorm, V RMSNorm without weight, sliding window ``0 <= q - k < 512``, proportional RoPE on full layers (64 of
256 frequency pairs, theta 1e6), default RoPE theta 1e4 on sliding layers.

KV: ``kv_caches[l]`` is ``get_buffers(l, kv_layout="HND")`` of layer ``l < 15``, ``[pages, 2, 1, page_size, D]``
FP8; ``pt_s`` / ``pt_f`` are the flat page tables of the sliding and full pools and ``pt_indptr[r]`` the offset of
request r's pages in both. Lengths are read on the device only: the op is CUDA-graph capturable and replays with
new ``cache_lens``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN = 1536
NUM_LAYERS = 35
PLE_DIM = 256
NUM_HEADS = 8
HEAD_DIM_SLIDING = 256
HEAD_DIM_FULL = 512
SLIDING_WINDOW = 512
EPS = 1e-6
FP8_MAX = 448.0
ROPE_THETA_SLIDING = 10000.0
ROPE_THETA_FULL = 1000000.0
ROPE_ANGLES_FULL = 64
LAYERS_A = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)
LAYERS_B = (4, 9, 14)
LAYERS_C = (15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33)
LAYERS_D = (19, 24, 29, 34)
KV_SOURCE_SLIDING = 13
KV_SOURCE_FULL = 14
MAX_REQUESTS = 8
MAX_TOKENS_PER_REQUEST = 8

#: The native op's weight list, in order (shapes and dtypes are checked by the op).
WEIGHT_NAMES = (
    "ple_model_proj_w", "ple_proj_norm_w", "layer_norm_w", "layer_scalar", "ple_gate_w", "ple_proj_w",
    "final_norm_w",
    "a_qkv_w", "a_o_w", "a_gu_w", "a_down_w", "a_qk_norm_w", "a_scales",
    "b_qkv_w", "b_o_w", "b_gu_w", "b_down_w", "b_qk_norm_w", "b_scales",
    "c_q_w", "c_o_w", "c_gu_w", "c_down_w", "c_q_norm_w", "c_scales",
    "d_q_w", "d_o_w", "d_gu_w", "d_down_w", "d_q_norm_w", "d_scales",
)  # fmt: skip


def layer_kind(i: int) -> tuple[str, int]:
    """(kind, index within the kind) of decoder layer ``i``."""
    if i < 15:
        return ("A", LAYERS_A.index(i)) if i % 5 != 4 else ("B", LAYERS_B.index(i))
    return ("C", LAYERS_C.index(i)) if i % 5 != 4 else ("D", LAYERS_D.index(i))


def kv_meta(kv_caches: list[torch.Tensor]) -> torch.Tensor:
    """CPU int64 ``[3, 15]``: per KV-producing layer the address of page 0's K rows, of its V rows, and the page
    stride in bytes, as the kernel takes them."""
    meta = torch.empty(3, 15, dtype=torch.int64)
    for layer, cache in enumerate(kv_caches):
        meta[0, layer] = cache[0, 0].data_ptr()
        meta[1, layer] = cache[0, 1].data_ptr()
        meta[2, layer] = cache.stride(0) * cache.element_size()
    return meta


class DecodeStack(OpWrapper):
    """The decoder-stack megakernel (Kernel Factory 382d6e41d928, paged KV). See the module docstring."""

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell("one request, Q=5 (MTP k=4 verify)", {"requests": 1, "tokens": 5}),
        Cell("one request, Q=1 (no speculation)", {"requests": 1, "tokens": 1}),
        Cell("8 requests, Q=6 (MTP k=5 at the batch limit)", {"requests": 8, "tokens": 6}),
        Cell("8 requests, Q=8 (largest verify block)", {"requests": 8, "tokens": 8}),
        Cell("3 requests, Q=4, short and page-aligned contexts", {"requests": 3, "tokens": 4}),
    )
    note = "page size must equal the compiled KV_PAGE_SHIFT (32); one KV head; FP8 KV with scale 1"

    def __call__(
        self,
        hidden_states,
        ple_token_embeds,
        cache_lens,
        kv_caches,
        pt_s,
        pt_f,
        pt_indptr,
        weights,
        out,
    ):
        from .. import native_ops

        assert kv_caches[0].shape[3] == 32, "the kernel is compiled for 32-token pages"
        native_ops.decode_stack(
            hidden_states,
            ple_token_embeds,
            cache_lens,
            kv_meta(kv_caches),
            pt_s,
            pt_f,
            pt_indptr,
            list(weights),
            out,
        )
        return out

    def reference(
        self,
        hidden_states,
        ple_token_embeds,
        cache_lens,
        kv_caches,
        pt_s,
        pt_f,
        pt_indptr,
        weights,
        out,
    ):
        """fp32 math with the op's rounding points; reads and writes the paged caches. ``out`` is filled too."""
        w = dict(zip(WEIGHT_NAMES, weights))
        num_tokens = hidden_states.shape[0]
        num_requests = cache_lens.shape[0]
        q_len = num_tokens // num_requests
        dev = hidden_states.device
        lens = cache_lens.long()
        pos = (lens[:, None] + torch.arange(q_len, device=dev)[None]).reshape(-1)
        req = torch.arange(num_requests, device=dev).repeat_interleave(q_len)
        page_size = kv_caches[0].shape[3]

        def rows(layer, kind, keys_pos, keys_req):
            table = pt_s if layer % 5 != 4 else pt_f
            page = table[pt_indptr[keys_req].long() + keys_pos // page_size].long()
            return kv_caches[layer][page, kind, 0, keys_pos % page_size]

        proj = (_lin16(hidden_states, w["ple_model_proj_w"]) * (HIDDEN**-0.5)).reshape(
            num_tokens, NUM_LAYERS, PLE_DIM
        )
        pli = (_rms(proj, w["ple_proj_norm_w"]) + ple_token_embeds) * (2.0**-0.5)
        h = hidden_states
        groups = {
            "A": (
                "a_qkv_w",
                "a_o_w",
                "a_gu_w",
                "a_down_w",
                "a_qk_norm_w",
                "a_scales",
                HEAD_DIM_SLIDING,
            ),
            "B": (
                "b_qkv_w",
                "b_o_w",
                "b_gu_w",
                "b_down_w",
                "b_qk_norm_w",
                "b_scales",
                HEAD_DIM_FULL,
            ),
            "C": (
                "c_q_w",
                "c_o_w",
                "c_gu_w",
                "c_down_w",
                "c_q_norm_w",
                "c_scales",
                HEAD_DIM_SLIDING,
            ),
            "D": ("d_q_w", "d_o_w", "d_gu_w", "d_down_w", "d_q_norm_w", "d_scales", HEAD_DIM_FULL),
        }
        for i in range(NUM_LAYERS):
            kind, j = layer_kind(i)
            qkv_n, o_n, gu_n, down_n, norm_n, sc_n, hd = groups[kind]
            sliding = hd == HEAD_DIM_SLIDING
            sc = w[sc_n][j]
            nw = w["layer_norm_w"][i]
            x = _rms(h, nw[0])
            if kind in "AB":
                qkv = _lin8(x, w[qkv_n][j], sc[0, 0], sc[0, 1])
                q = qkv[:, : NUM_HEADS * hd].reshape(num_tokens, NUM_HEADS, hd)
                k = qkv[:, NUM_HEADS * hd : NUM_HEADS * hd + hd].reshape(num_tokens, 1, hd)
                v = _rms(qkv[:, NUM_HEADS * hd + hd :])
                q = rope(_rms(q, w[norm_n][j, 0]), pos, hd)
                k = rope(_rms(k, w[norm_n][j, 1]), pos, hd)[:, 0]
                table = pt_s if sliding else pt_f
                page = table[pt_indptr[req].long() + pos // page_size].long()
                kv_caches[i][page, 0, 0, pos % page_size] = k.float().to(torch.float8_e4m3fn)
                kv_caches[i][page, 1, 0, pos % page_size] = v.float().to(torch.float8_e4m3fn)
                kv_layer = i
            else:
                q = rope(
                    _rms(_lin16(x, w[qkv_n][j]).reshape(num_tokens, NUM_HEADS, hd), w[norm_n][j]),
                    pos,
                    hd,
                )
                kv_layer = KV_SOURCE_SLIDING if sliding else KV_SOURCE_FULL
            attn = torch.empty(num_tokens, NUM_HEADS * hd, dtype=torch.bfloat16, device=dev)
            for t in range(num_tokens):
                p, r = int(pos[t]), int(req[t])
                lo = max(p - SLIDING_WINDOW + 1, 0) if sliding else 0
                keys = torch.arange(lo, p + 1, device=dev)
                kr = torch.full_like(keys, r)
                kc = rows(kv_layer, 0, keys, kr).float()
                vc = rows(kv_layer, 1, keys, kr).float()
                prob = torch.softmax(q[t].float() @ kc.t(), -1)
                attn[t] = (prob @ vc).reshape(-1).to(torch.bfloat16)
            o_row = 1 if kind in "AB" else 0
            h = h + _rms(_lin8(attn, w[o_n][j], sc[o_row, 0], sc[o_row, 1]), nw[1])
            gu = _lin8(_rms(h, nw[2]), w[gu_n][j], sc[o_row + 1, 0], sc[o_row + 1, 1])
            inter = gu.shape[-1] // 2
            act = (F.gelu(gu[:, :inter].float(), approximate="tanh") * gu[:, inter:].float()).to(
                torch.bfloat16
            )
            h = h + _rms(_lin8(act, w[down_n][j], sc[o_row + 2, 0], sc[o_row + 2, 1]), nw[3])
            g = (
                F.gelu(_lin16(h, w["ple_gate_w"][i]).float(), approximate="tanh")
                * pli[:, i].float()
            ).to(torch.bfloat16)
            h = h + _rms(_lin16(g, w["ple_proj_w"][i]), nw[4])
            h = (h.float() * w["layer_scalar"][i].float()).to(torch.bfloat16)
        out.copy_(_rms(h, w["final_norm_w"]))
        return out


decode_stack = DecodeStack()


def _rms(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if weight is not None:
        y = y * weight.float()
    return y.to(torch.bfloat16)


def _lin8(x, w8, w_scale, in_scale):
    xq = (x.float() / in_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (xq.float() @ w8.float().t() * (in_scale * w_scale)).to(torch.bfloat16)


def _lin16(x, w):
    return (x.float() @ w.float().t()).to(torch.bfloat16)


def rope_inv_freq(head_dim: int, device=None) -> torch.Tensor:
    """Gemma-4 RoPE frequencies of a layer with ``head_dim``: default (theta 1e4) on sliding layers, proportional
    (theta 1e6, the first 64 of 256 frequency pairs rotate) on full layers."""
    if head_dim == HEAD_DIM_SLIDING:
        return 1.0 / (
            ROPE_THETA_SLIDING
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
    inv = torch.zeros(head_dim // 2, device=device, dtype=torch.float32)
    inv[:ROPE_ANGLES_FULL] = 1.0 / (
        ROPE_THETA_FULL
        ** (torch.arange(0, 2 * ROPE_ANGLES_FULL, 2, device=device, dtype=torch.float32) / head_dim)
    )
    return inv


def rope(x: torch.Tensor, positions: torch.Tensor, head_dim: int) -> torch.Tensor:
    """NeoX rotate-half RoPE of ``x [T, heads, head_dim]`` at ``positions [T]`` in fp32, rounded to bf16."""
    half = head_dim // 2
    ang = positions.float()[:, None] * rope_inv_freq(head_dim, x.device)[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    xf = x.float()
    rot = torch.cat([-xf[..., half:], xf[..., :half]], -1)
    return (xf * cos + rot * sin).to(torch.bfloat16)


def head_dim_of(layer: int) -> int:
    return HEAD_DIM_FULL if layer % 5 == 4 else HEAD_DIM_SLIDING
