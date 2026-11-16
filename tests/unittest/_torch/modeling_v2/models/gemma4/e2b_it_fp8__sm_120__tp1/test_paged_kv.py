# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's Triton paged-KV append and attention against their PyTorch references, on every cell."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 E4M3 Triton kernels need a Hopper or newer GPU",
)

PAGE_SIZE = 32


def _ops():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops import (
        paged_kv,
    )

    return paged_kv


def _cache(num_pages, head_dim, layers=3, layer=1, seed=0):
    """A layer view ``[pages, 2, 1, page_size, D]`` into a pool that interleaves ``layers`` layers per page."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pool = (
        torch.randn(num_pages, layers, 2, PAGE_SIZE, head_dim, device="cuda", generator=gen) * 0.5
    ).to(torch.float8_e4m3fn)
    return pool, pool[:, layer].unsqueeze(2)


def _tables(seq_kv_lens, num_pages, seed=0):
    gen = torch.Generator().manual_seed(seed)
    pages = [-(-n // PAGE_SIZE) for n in seq_kv_lens]
    perm = torch.randperm(num_pages, generator=gen)[: sum(pages)].to(torch.int32)
    starts = torch.tensor([0] + pages[:-1], dtype=torch.int32).cumsum(0).to(torch.int32)
    return perm.cuda(), starts.cuda()


@pytest.mark.parametrize("head_dim", [256, 512])
def test_append_writes_only_the_new_rows(head_dim):
    pk = _ops()
    kv_lens = [70, 33, 128]
    q_lens = [5, 33, 1]
    pool, cache = _cache(16, head_dim)
    table, starts = _tables(kv_lens, 16)
    token_req = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), torch.tensor(q_lens)
    ).cuda()
    positions = (
        torch.cat([torch.arange(n - q, n) for n, q in zip(kv_lens, q_lens)]).to(torch.int32).cuda()
    )
    positions[2] = -1  # a padding row is skipped
    k = torch.randn(len(positions), head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = pool.clone()
    pk.paged_kv_append.reference(
        k, v, expected[:, 1].unsqueeze(2), table, starts, token_req, positions
    )
    pk.paged_kv_append(k, v, cache, table, starts, token_req, positions)
    assert torch.equal(pool.view(torch.uint8), expected.view(torch.uint8))


ATTENTION_CASES = [
    # heads, head_dim, window, per-request (kv_len, q_len), tokens per block (None: the prefill default), splits
    (8, 256, 512, [(1300, 1300), (40, 7)], None, 1),
    (8, 512, 0, [(700, 700), (2100, 6)], None, 1),
    (8, 256, 512, [(4100, 6), (513, 6), (5, 5)], None, 1),
    (4, 256, 512, [(2000, 1), (511, 1), (512, 1), (3, 1)], 1, 1),
    (4, 512, 0, [(2000, 1), (64, 1), (1, 1)], 1, 1),
    (4, 256, 512, [(4100, 1), (511, 1), (512, 1), (3, 1)], 1, 4),
    (4, 512, 0, [(4600, 1), (64, 1), (1, 1), (33, 1)], 1, 16),
]


@pytest.mark.parametrize("heads, head_dim, window, seqs, tpb, splits", ATTENTION_CASES)
def test_attention_matches_reference(heads, head_dim, window, seqs, tpb, splits):
    pk = _ops()
    kv_lens = [n for n, _ in seqs]
    q_lens = [q for _, q in seqs]
    num_pages = sum(-(-n // PAGE_SIZE) for n in kv_lens) + 4
    _, cache = _cache(num_pages, head_dim, seed=1)
    table, starts = _tables(kv_lens, num_pages, seed=2)
    token_req = torch.repeat_interleave(
        torch.arange(len(seqs), dtype=torch.int32), torch.tensor(q_lens)
    ).cuda()
    if tpb == 1:
        # Assistant draft rows: one query per request, anchored at the last committed key.
        anchors = torch.tensor([n - 1 for n in kv_lens], dtype=torch.int32).cuda()
    else:
        anchors = torch.cat([torch.arange(n - q, n) for n, q in seqs]).to(torch.int32).cuda()
    if tpb is None:
        tpb = pk.paged_attention.tokens_per_block(heads, head_dim)
    blocks = pk.request_blocks(q_lens, tpb, "cuda")
    gen = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(len(anchors), heads, head_dim, device="cuda", generator=gen).to(torch.bfloat16)
    out = pk.paged_attention(
        q, cache, table, starts, token_req, anchors, blocks, tpb, window, splits
    )
    ref = pk.paged_attention.reference(
        q, cache, table, starts, token_req, anchors, blocks, tpb, window
    )
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 2e-2, err
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 5e-3, rel


def test_block_table_keeps_requests_apart():
    pk = _ops()
    blocks = pk.request_blocks([3, 9, 1], 4, "cpu").tolist()
    assert blocks == [[0, 3], [3, 7], [7, 11], [11, 12], [12, 13]]
