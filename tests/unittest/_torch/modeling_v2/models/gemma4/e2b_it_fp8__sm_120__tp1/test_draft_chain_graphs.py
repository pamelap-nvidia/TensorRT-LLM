# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DraftChainGraphs``: the graphed draft chain returns exactly what the eager chain returns, follows new inputs on
every replay, and keeps one graph per key and input shape.

The chain stands in for the assistant's: per step a projection to 4 query heads, the target's Triton paged
attention over a paged FP8 cache (one query per request, flash-decoding splits), an output projection and a
greedy token, which feeds the next step's embedding.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 E4M3 Triton kernels need a Hopper or newer GPU",
)

PAGE_SIZE = 32
HEADS, HEAD_DIM, WIDTH, VOCAB, STEPS = 4, 512, 256, 1000, 3


def _modules():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1 import (
        modeling,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops import (
        paged_kv,
    )

    return modeling, paged_kv


def _setup(kv_lens, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-n // PAGE_SIZE) for n in kv_lens]
    num_pages = sum(pages) + 2
    cache = (
        torch.randn(num_pages, 2, 1, PAGE_SIZE, HEAD_DIM, device="cuda", generator=gen) * 0.5
    ).to(torch.float8_e4m3fn)
    table = torch.randperm(num_pages, device="cuda", generator=gen)[: sum(pages)].to(torch.int32)
    w = {
        "embed": torch.randn(VOCAB, WIDTH, device="cuda", generator=gen).to(torch.bfloat16),
        "q": (torch.randn(HEADS * HEAD_DIM, WIDTH, device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        ),
        "o": (torch.randn(WIDTH, HEADS * HEAD_DIM, device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        ),
        "head": torch.randn(VOCAB, WIDTH, device="cuda", generator=gen).to(torch.bfloat16),
    }
    return cache, table, pages, w


def _inputs(kv_lens, pages, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = len(kv_lens)
    starts = torch.tensor([0] + pages[:-1], dtype=torch.int32).cumsum(0).to(torch.int32).cuda()
    return dict(
        tokens=torch.randint(0, VOCAB, (b,), device="cuda", generator=gen, dtype=torch.int32),
        hidden=torch.randn(b, WIDTH, device="cuda", generator=gen).to(torch.bfloat16),
        starts=starts,
        anchors=torch.tensor(kv_lens, dtype=torch.int32).cuda() - 1 - seed % 3,
    )


def _chain(paged_kv, cache, table, w):
    def chain(tokens, hidden, starts, anchors):
        b = tokens.shape[0]
        rows = torch.arange(b, dtype=torch.int32, device=tokens.device)
        blocks = torch.stack([rows, rows + 1], 1)
        drafts = []
        for _ in range(STEPS):
            x = w["embed"][tokens.long()] + hidden
            q = (x @ w["q"].t()).view(b, HEADS, HEAD_DIM)
            attn = paged_kv.paged_attention(
                q, cache, table, starts, rows, anchors, blocks, 1, 0, 16
            )
            hidden = attn.view(b, -1) @ w["o"].t()
            tokens = (hidden @ w["head"].t()).argmax(-1).to(torch.int32)
            drafts.append(tokens)
        return torch.stack(drafts, 1)

    return chain


@pytest.mark.parametrize(
    "kv_lens", [[3000], [1200, 77, 4100]], ids=["one_request", "three_requests"]
)
def test_graph_matches_eager_and_follows_inputs(kv_lens):
    m, pk = _modules()
    cache, table, pages, w = _setup(kv_lens)
    chain = _chain(pk, cache, table, w)
    graphs = m.DraftChainGraphs()
    key = (len(kv_lens), STEPS, table.data_ptr(), cache.data_ptr())
    outs = []
    for seed in range(3):  # capture, then two replays with new inputs
        inputs = _inputs(kv_lens, pages, seed)
        eager = chain(**inputs)
        graphed = graphs(chain, inputs, key)
        assert torch.equal(graphed, eager), seed
        outs.append(graphed)
    assert len(graphs._graphs) == 1
    # The returned drafts are not the graph's buffer: an earlier result survives later replays.
    assert torch.equal(outs[0], chain(**_inputs(kv_lens, pages, 0)))


def test_one_graph_per_key_and_shape():
    m, pk = _modules()
    cache, table, pages, w = _setup([500, 600, 700])
    chain = _chain(pk, cache, table, w)
    graphs = m.DraftChainGraphs()
    for b in (1, 3, 1):
        inputs = _inputs([500, 600, 700][:b], pages[:b], seed=b)
        assert torch.equal(graphs(chain, inputs, ("k",)), chain(**inputs))
    assert len(graphs._graphs) == 2
