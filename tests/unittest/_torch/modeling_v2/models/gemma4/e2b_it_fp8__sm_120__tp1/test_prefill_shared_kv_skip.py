# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The layer-by-layer prefill with the KV-shared layers (15-34) skipped on the rows no one reads.

Random weights with the checkpoint's shapes and FP8 statistics (``_kf_reference.generate_inputs``), two paged
pools (sliding: 12 layers, full: 3) with random page tables, a mixed step of context requests and one generation
request. With and without the skip, from the same cache contents: every cache byte must be the same (only layers
0-14 write KV), and the kept rows -- each context's last row and every generation row -- must agree within the
FP8 chaotic-noise level.
"""

import _kf_reference as kf
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 GEMMs and FP8 E4M3 Triton kernels need a Hopper or newer GPU",
)

PAGE_SIZE = 32


def _modeling():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1 import (
        modeling,
    )

    return modeling


def test_shared_kv_rows():
    m = _modeling()
    # Two contexts (5 and 3 prompt rows), two generation requests of 2 rows each.
    assert m.shared_kv_rows([5, 3, 2, 2], 2) == ([4, 7, 8, 9, 10, 11], [1, 1, 2, 2])
    assert m.shared_kv_rows([4, 6], 2) == ([3, 9], [1, 1])
    # Nothing to skip: no contexts, or single-row contexts.
    assert m.shared_kv_rows([3, 3], 0) is None
    assert m.shared_kv_rows([1, 1, 4], 2) is None


class _Pools:
    """The 15 KV-producing layers' ``[pages, 2, 1, page_size, D]`` views into two pools that interleave layers."""

    def __init__(self, num_pages, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        # Past KV with the real magnitudes (as ``_kf_reference``): K ~ the k_norm weights (~0.12), V unit RMS.
        kv_scale = torch.tensor([0.12, 1.0], device="cuda").view(1, 1, 2, 1, 1)
        self.pools = {
            kind: (
                torch.randn(num_pages, layers, 2, PAGE_SIZE, hd, device="cuda", generator=gen)
                * kv_scale
            ).to(torch.float8_e4m3fn)
            for kind, layers, hd in (("s", 12, 256), ("f", 3, 512))
        }
        self.views, js, jf = [], 0, 0
        for layer in range(15):
            if layer % 5 == 4:
                self.views.append(self.pools["f"][:, jf].unsqueeze(2))
                jf += 1
            else:
                self.views.append(self.pools["s"][:, js].unsqueeze(2))
                js += 1

    def snapshot(self):
        return {k: v.clone() for k, v in self.pools.items()}

    def restore(self, snap):
        for k, v in self.pools.items():
            v.copy_(snap[k])


@pytest.mark.parametrize(
    "ctx_lens, gen_kv_lens, q_len",
    [([700], [], 0), ([300, 77, 1], [500], 3), ([1200, 40], [90, 2000], 6)],
    ids=["one_context", "mixed_three_contexts", "mixed_two_and_two"],
)
def test_skip_keeps_cache_and_read_rows(ctx_lens, gen_kv_lens, q_len):
    m = _modeling()
    device = torch.device("cuda")
    inputs = kf.generate_inputs(
        {"num_requests": 8, "tokens_per_request": 8, "max_ctx": 4096}, device, 5
    )
    w = {name: inputs[name].contiguous() for name in m.WEIGHT_NAMES}
    rope = m._Rope(4096, device)

    seq_lens = list(ctx_lens) + [q_len] * len(gen_kv_lens)
    kv_lens = list(ctx_lens) + list(gen_kv_lens)
    num_tokens = sum(seq_lens)
    positions = torch.cat(
        [torch.arange(n) for n in ctx_lens] + [torch.arange(n - q_len, n) for n in gen_kv_lens]
    ).to(device, torch.int32)
    reps = -(-num_tokens // inputs["hidden_states"].shape[0])
    hidden = inputs["hidden_states"].repeat(reps, 1)[:num_tokens].contiguous()
    ple_tokens = inputs["ple_token_embeds"].repeat(reps, 1, 1)[:num_tokens].contiguous()

    pages = [-(-n // PAGE_SIZE) for n in kv_lens]
    num_pages = sum(pages) + 7
    gen = torch.Generator().manual_seed(1)
    tables = {
        True: torch.randperm(num_pages, generator=gen)[: sum(pages)].to(device, torch.int32),
        False: torch.randperm(num_pages, generator=gen)[: sum(pages)].to(device, torch.int32),
    }
    starts = torch.tensor([0] + pages[:-1]).cumsum(0).to(device, torch.int32)
    pools = _Pools(num_pages)
    before = pools.snapshot()

    def run(skip):
        pools.restore(before)
        out = m.prefill_layers(
            w, rope, pools.views, tables, starts, seq_lens, len(ctx_lens), hidden, ple_tokens,
            positions, skip_shared_kv_layers=skip,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out, pools.snapshot()

    full, full_pools = run(False)
    skipped, skipped_pools = run(True)

    for kind in full_pools:
        assert torch.equal(
            full_pools[kind].view(torch.uint8), skipped_pools[kind].view(torch.uint8)
        )
        assert not torch.equal(full_pools[kind].view(torch.uint8), before[kind].view(torch.uint8))

    rows, _ = m.shared_kv_rows(seq_lens, len(ctx_lens)) or (list(range(num_tokens)), None)
    rows = torch.tensor(rows, device=device)
    a, b = full[rows].float(), skipped[rows].float()
    assert torch.isfinite(a).all()
    rel_l2 = ((a - b).norm() / a.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    # Layers 15-34 see the same inputs on the kept rows; only the GEMMs' row count changes, and cuBLAS picks its
    # kernel (accumulation order) by shape, so a row moves by ~1 ulp. FP8 static activation quantization makes
    # random-weight stacks chaotic at that level (see ``_kf_reference.check_outputs``): a statistical check, with
    # 20 instead of 35 layers of drift.
    assert rel_l2 <= 0.15 and cos.mean() >= 0.98 and cos.min() >= 0.95, (
        rel_l2,
        cos.mean(),
        cos.min(),
    )
    dropped = torch.ones(num_tokens, dtype=torch.bool, device=device)
    dropped[rows] = False
    assert not skipped[dropped].any()
