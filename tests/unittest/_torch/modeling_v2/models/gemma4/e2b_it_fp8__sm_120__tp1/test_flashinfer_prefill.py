# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's batch prefill over the paged FP8 cache (the target's prompt attention of layers 0-14) against the
fp32 reference of the Triton kernel, with the same 512-key window, on strided layer views of interleaved pools."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="the target's FlashInfer prefill is used on sm120",
)

PAGE_SIZE = 32


def _ops():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops import (
        flashinfer_prefill,
        paged_kv,
    )

    return flashinfer_prefill, paged_kv


CASES = [
    # head_dim, window, per-request (cached tokens, prompt rows)
    (256, 512, [(0, 1300), (0, 40), (0, 1)]),
    (256, 512, [(0, 4096)]),
    (512, 0, [(0, 700), (0, 2100), (0, 33)]),
    (512, 0, [(0, 4096)]),
    (256, 512, [(64, 600)]),  # prompt after cached tokens: queries are the last rows of the keys
]


@pytest.mark.parametrize("head_dim, window, reqs", CASES)
def test_flashinfer_prefill_matches_reference(head_dim, window, reqs):
    fp, pk = _ops()
    kv_lens = [c + q for c, q in reqs]
    q_lens = [q for _, q in reqs]
    pages = [-(-n // PAGE_SIZE) for n in kv_lens]
    num_pages = sum(pages) + 5
    gen = torch.Generator(device="cuda").manual_seed(0)
    # A pool interleaving 3 layers per page (K ~ 0.12, V ~ 1, as the real cache); layer 1's strided view.
    pool = torch.empty(
        num_pages, 3, 2, PAGE_SIZE, head_dim, device="cuda", dtype=torch.float8_e4m3fn
    )
    pool[:, :, 0] = (
        torch.randn(num_pages, 3, PAGE_SIZE, head_dim, device="cuda", generator=gen) * 0.12
    ).to(torch.float8_e4m3fn)
    pool[:, :, 1] = torch.randn(num_pages, 3, PAGE_SIZE, head_dim, device="cuda", generator=gen).to(
        torch.float8_e4m3fn
    )
    cache = pool[:, 1].unsqueeze(2)
    table = torch.randperm(num_pages, generator=torch.Generator().manual_seed(1))[: sum(pages)]
    table = table.to("cuda", torch.int32)
    starts = torch.tensor([0] + pages[:-1]).cumsum(0).to("cuda", torch.int32)
    token_req = torch.repeat_interleave(
        torch.arange(len(reqs), dtype=torch.int32), torch.tensor(q_lens)
    ).cuda()
    positions = torch.cat([torch.arange(c, c + q) for c, q in reqs]).to("cuda", torch.int32)
    q = torch.randn(len(positions), 8, head_dim, device="cuda", generator=gen)
    q = (q / q.norm(dim=-1, keepdim=True) * head_dim**0.5).to(torch.bfloat16)

    plan = fp.PrefillPlan(8, head_dim, window, PAGE_SIZE, torch.device("cuda"))
    plan.plan(q_lens, kv_lens, pages, table)
    out = fp.flashinfer_paged_prefill(plan, q, cache)
    # The reference is O(T) python loops: check a sample of rows of long prompts.
    rows = torch.unique(
        torch.cat([torch.arange(0, len(positions), 97), torch.tensor([len(positions) - 1])])
    )
    rows = rows.cuda()
    ref = fp.flashinfer_paged_prefill.reference(plan, q, cache)[rows]
    err = (out[rows].float() - ref.float()).abs().max().item()
    rel = ((out[rows].float() - ref.float()).norm() / ref.float().norm()).item()
    assert err < 2e-2 and rel < 5e-3, (err, rel)
    # And the same as the Triton kernel the generation rows keep.
    tpb = pk.paged_attention.tokens_per_block(8, head_dim)
    tri = pk.paged_attention(
        q,
        cache,
        table,
        starts,
        token_req,
        positions,
        pk.request_blocks(q_lens, tpb, "cuda"),
        tpb,
        window,
    )
    assert ((out.float() - tri.float()).norm() / tri.float().norm()).item() < 5e-3
