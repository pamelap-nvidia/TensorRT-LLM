# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged FP8 KV cache ops of the Gemma-4 E2B target: append and grouped-query attention, in Triton.

Gemma-4 has one KV head per layer. A layer's cache is the engine's buffer ``kv_cache_manager.get_buffers(layer,
kv_layout="HND")``: ``[pages, 2 (K, V), 1, page_size, head_dim]`` FP8 E4M3 (scale 1), possibly a strided view
into a pool that interleaves several layers per page. A request's pages are ``page_table[kv_start[r] + i]``; token
``t`` of the request lives in page ``t // page_size`` at slot ``t % page_size``.

Attention masks by an *anchor* per query token: the query sees keys ``k`` with ``lo <= k <= anchor``, where
``lo = anchor - window + 1`` on sliding-window layers (``window`` keys, Hugging Face's ``0 <= q - k < window``)
and 0 otherwise. A prefill or verify token anchors at its own position; a draft query of the Q-only assistant,
which reads the target's committed KV of length ``kv_len``, anchors at ``kv_len - 1``. The softmax scale is 1
(Gemma-4 normalizes q and k).

Query tokens are processed in blocks that never span two requests: ``blocks[b] = (first token, end token)``,
every token of a block belonging to the same request. Shapes depend only on the block count, so a fixed block
table (one token per block, as the draft chain uses) is CUDA-graph capturable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _append_kernel(
    k_ptr,
    v_ptr,
    cache_ptr,
    page_table_ptr,
    kv_start_ptr,
    token_req_ptr,
    positions_ptr,
    num_tokens,
    stride_page,
    stride_kv,
    stride_tok,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= num_tokens:
        return
    req = tl.load(token_req_ptr + t)
    pos = tl.load(positions_ptr + t)
    if pos < 0:
        return
    page = tl.load(page_table_ptr + tl.load(kv_start_ptr + req) + pos // PAGE_SIZE).to(tl.int64)
    base = page * stride_page + (pos % PAGE_SIZE) * stride_tok
    d = tl.arange(0, HEAD_DIM)
    k = tl.load(k_ptr + t * HEAD_DIM + d).to(tl.float32)
    v = tl.load(v_ptr + t * HEAD_DIM + d).to(tl.float32)
    tl.store(cache_ptr + base + d, k.to(tl.float8e4nv))
    tl.store(cache_ptr + base + stride_kv + d, v.to(tl.float8e4nv))


@triton.jit
def _attention_kernel(
    q_ptr,
    out_ptr,
    cache_ptr,
    page_table_ptr,
    kv_start_ptr,
    token_req_ptr,
    anchor_ptr,
    blocks_ptr,
    part_o_ptr,
    part_ml_ptr,
    num_rows,
    stride_page,
    stride_kv,
    stride_tok,
    WINDOW: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    split = tl.program_id(1)
    tok0 = tl.load(blocks_ptr + 2 * b)
    tok1 = tl.load(blocks_ptr + 2 * b + 1)
    rows = tl.arange(0, BLOCK_M)
    tok = tok0 + rows // NUM_HEADS
    head = rows % NUM_HEADS
    row_ok = tok < tok1
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + (tok[:, None] * NUM_HEADS + head[:, None]) * HEAD_DIM + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    req = tl.load(token_req_ptr + tok0)
    kv_start = tl.load(kv_start_ptr + req)
    anchor = tl.load(anchor_ptr + tok, mask=row_ok, other=-1)
    if WINDOW > 0:
        lo = tl.maximum(anchor - WINDOW + 1, 0)
    else:
        lo = tl.zeros([BLOCK_M], dtype=tl.int32)
    lo = tl.where(row_ok, lo, 2147483647)
    k_begin = (tl.min(lo, axis=0) // BLOCK_N) * BLOCK_N
    k_end = tl.max(anchor, axis=0) + 1
    if NUM_SPLITS > 1:
        # Flash-decoding: this program covers one of NUM_SPLITS equal, BLOCK_N-aligned slices of the key range.
        chunk = tl.cdiv(tl.cdiv(tl.maximum(k_end - k_begin, 0), NUM_SPLITS), BLOCK_N) * BLOCK_N
        k_begin = k_begin + split * chunk
        k_end = tl.minimum(k_end, k_begin + chunk)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    n = tl.arange(0, BLOCK_N)
    for k0 in range(k_begin, k_end, BLOCK_N):
        kk = k0 + n
        k_ok = kk < k_end
        page = tl.load(page_table_ptr + kv_start + kk // PAGE_SIZE, mask=k_ok, other=0).to(tl.int64)
        addr = page * stride_page + (kk % PAGE_SIZE) * stride_tok
        k = tl.load(cache_ptr + addr[:, None] + d[None, :], mask=k_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
        visible = (kk[None, :] >= lo[:, None]) & (kk[None, :] <= anchor[:, None])
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            cache_ptr + stride_kv + addr[:, None] + d[None, :], mask=k_ok[:, None], other=0.0
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    row = tok * NUM_HEADS + head
    if NUM_SPLITS == 1:
        tl.store(
            out_ptr + row[:, None] * HEAD_DIM + d[None, :],
            out.to(tl.bfloat16),
            mask=row_ok[:, None],
        )
    else:
        part = split * num_rows + row
        tl.store(part_o_ptr + part[:, None] * HEAD_DIM + d[None, :], out, mask=row_ok[:, None])
        tl.store(part_ml_ptr + 2 * part, m_i, mask=row_ok)
        tl.store(part_ml_ptr + 2 * part + 1, l_i, mask=row_ok)


@triton.jit
def _combine_kernel(
    part_o_ptr, part_ml_ptr, out_ptr, num_rows, HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    s = tl.arange(0, NUM_SPLITS)
    m = tl.load(part_ml_ptr + 2 * (s * num_rows + row))
    denom = tl.load(part_ml_ptr + 2 * (s * num_rows + row) + 1)
    m_max = tl.max(m, axis=0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.where(denom > 0.0, tl.exp(m - m_max) * denom, 0.0)
    d = tl.arange(0, HEAD_DIM)
    o = tl.load(part_o_ptr + (s[:, None] * num_rows + row) * HEAD_DIM + d[None, :])
    total = tl.sum(w, axis=0)
    out = tl.sum(o * w[:, None], axis=0) / tl.where(total > 0.0, total, 1.0)
    tl.store(out_ptr + row * HEAD_DIM + d, out.to(tl.bfloat16))


def _cache_strides(cache: torch.Tensor) -> tuple[int, int, int]:
    assert cache.dtype == torch.float8_e4m3fn and cache.dim() == 5, (cache.dtype, cache.shape)
    assert cache.shape[1] == 2 and cache.shape[2] == 1 and cache.stride(-1) == 1, (
        cache.shape,
        cache.stride(),
    )
    return cache.stride(0), cache.stride(1), cache.stride(3)


class PagedKvAppend(OpWrapper):
    """Write one token's K and V row per query token into the layer's paged FP8 cache (E4M3, scale 1).

    ``k`` / ``v``: ``[T, head_dim]`` bf16 (one KV head). ``token_req[t]`` / ``positions[t]``: the token's request
    and absolute position; a negative position skips the token (padding rows). Rows are rounded to E4M3 with
    saturation off, as the engine's FP8 KV append does.
    """

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell("sliding layer, head_dim 256, page 32", {"head_dim": 256, "page_size": 32}),
        Cell("full layer, head_dim 512, page 32", {"head_dim": 512, "page_size": 32}),
    )

    def __call__(self, k, v, cache, page_table, kv_start, token_req, positions):
        stride_page, stride_kv, stride_tok = _cache_strides(cache)
        num_tokens, head_dim = k.shape
        if num_tokens == 0:
            return
        _append_kernel[(num_tokens,)](
            k,
            v,
            cache,
            page_table,
            kv_start,
            token_req,
            positions,
            num_tokens,
            stride_page,
            stride_kv,
            stride_tok,
            PAGE_SIZE=cache.shape[3],
            HEAD_DIM=head_dim,
        )

    def reference(self, k, v, cache, page_table, kv_start, token_req, positions):
        page_size = cache.shape[3]
        for t in range(k.shape[0]):
            pos = int(positions[t])
            if pos < 0:
                continue
            page = int(page_table[int(kv_start[int(token_req[t])]) + pos // page_size])
            cache[page, 0, 0, pos % page_size] = k[t].float().to(torch.float8_e4m3fn)
            cache[page, 1, 0, pos % page_size] = v[t].float().to(torch.float8_e4m3fn)


class PagedAttention(OpWrapper):
    """Grouped-query attention of ``q [T, H, D]`` over one KV head in a paged FP8 cache; returns ``[T, H, D]`` bf16.

    See the module docstring for the anchor mask and the block table. ``window`` 0 is full attention.
    """

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(
            "sliding layer prefill: 8 heads, D 256, window 512",
            {"heads": 8, "head_dim": 256, "window": 512},
        ),
        Cell("full layer prefill: 8 heads, D 512", {"heads": 8, "head_dim": 512, "window": 0}),
        Cell(
            "assistant draft: 4 heads, D 256, window 512, one token per block",
            {"heads": 4, "head_dim": 256, "window": 512},
        ),
        Cell(
            "assistant draft: 4 heads, D 512, one token per block",
            {"heads": 4, "head_dim": 512, "window": 0},
        ),
        Cell(
            "assistant draft, split-K over 16 programs",
            {"heads": 4, "head_dim": 512, "window": 0, "splits": 16},
        ),
    )
    note = (
        "blocks must not span requests; a row whose anchor is negative returns zeros. num_splits > 1 (a power of"
        " two) splits each block's key range over that many programs and combines them (flash-decoding), for the"
        " few-row draft queries over long caches"
    )

    @staticmethod
    def tile(num_heads: int, head_dim: int, tokens_per_block: int) -> tuple[int, int]:
        """(BLOCK_M, BLOCK_N) for a block of ``tokens_per_block`` tokens."""
        block_m = max(16, triton.next_power_of_2(num_heads * tokens_per_block))
        block_n = 64 if head_dim <= 256 else 32
        return block_m, block_n

    @staticmethod
    def tokens_per_block(num_heads: int, head_dim: int) -> int:
        """Prefill block size in tokens: 64 query rows for D 256, 32 for D 512."""
        return max(1, (64 if head_dim <= 256 else 32) // num_heads)

    def __call__(
        self,
        q,
        cache,
        page_table,
        kv_start,
        token_req,
        anchor,
        blocks,
        tokens_per_block,
        window,
        num_splits=1,
    ):
        stride_page, stride_kv, stride_tok = _cache_strides(cache)
        num_tokens, num_heads, head_dim = q.shape
        out = torch.empty_like(q)
        if blocks.shape[0] == 0:
            return out
        num_rows = num_tokens * num_heads
        if num_splits > 1:
            part_o = torch.empty(
                num_splits, num_rows, head_dim, dtype=torch.float32, device=q.device
            )
            part_ml = torch.empty(num_splits, num_rows, 2, dtype=torch.float32, device=q.device)
        else:
            part_o = part_ml = out
        block_m, block_n = self.tile(num_heads, head_dim, tokens_per_block)
        _attention_kernel[(blocks.shape[0], num_splits)](
            q,
            out,
            cache,
            page_table,
            kv_start,
            token_req,
            anchor,
            blocks,
            part_o,
            part_ml,
            num_rows,
            stride_page,
            stride_kv,
            stride_tok,
            WINDOW=window,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            PAGE_SIZE=cache.shape[3],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_SPLITS=num_splits,
            num_warps=4 if head_dim <= 256 else 8,
            num_stages=2,
        )
        if num_splits > 1:
            _combine_kernel[(num_rows,)](
                part_o,
                part_ml,
                out,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                num_warps=4,
            )
        return out

    def reference(
        self,
        q,
        cache,
        page_table,
        kv_start,
        token_req,
        anchor,
        blocks,
        tokens_per_block,
        window,
        num_splits=1,
    ):
        """fp32 attention over the dequantized cache; ``blocks``, ``tokens_per_block``, ``num_splits`` are ignored."""
        page_size = cache.shape[3]
        res = torch.zeros_like(q)
        for t in range(q.shape[0]):
            a = int(anchor[t])
            if a < 0:
                continue
            lo = max(a - window + 1, 0) if window > 0 else 0
            start = int(kv_start[int(token_req[t])])
            keys = torch.arange(lo, a + 1, device=q.device)
            pages = page_table[start + keys // page_size].long()
            kc = cache[pages, 0, 0, keys % page_size].float()
            vc = cache[pages, 1, 0, keys % page_size].float()
            s = q[t].float() @ kc.t()
            res[t] = (torch.softmax(s, -1) @ vc).to(q.dtype)
        return res


paged_kv_append = PagedKvAppend()
paged_attention = PagedAttention()


def request_blocks(seq_lens: list[int], tokens_per_block: int, device) -> torch.Tensor:
    """Host-built block table ``[num_blocks, 2]`` (first token, end token) for requests of ``seq_lens`` tokens laid
    out back to back: every request split into blocks of at most ``tokens_per_block`` tokens."""
    blocks = []
    start = 0
    for n in seq_lens:
        for b0 in range(start, start + n, tokens_per_block):
            blocks.append((b0, min(b0 + tokens_per_block, start + n)))
        start += n
    return torch.tensor(blocks, dtype=torch.int32).reshape(-1, 2).to(device, non_blocking=True)
