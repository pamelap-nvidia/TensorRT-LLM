# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt attention of the KV-producing layers with FlashInfer's batch prefill over the paged FP8 cache.

Called directly (``flashinfer.BatchPrefillWithPagedKVCacheWrapper``, FA2 backend), not through an attention
backend. It serves only the context requests of a step: their KV lengths (cached tokens + prompt chunk) are known
on the host, so each step plans with the exact lengths and the kernel runs at the planned lengths -- the SM120
FlashInfer decode issues (runtime KV lengths shorter than planned) cannot occur. Generation rows of mixed steps
and every draft / verify query keep the Triton kernel (``paged_kv.paged_attention``).

The plan (host work and a small upload) is state, so it lives in a ``PrefillPlan`` (wrapper + 128 MiB workspace)
the target keeps per layer geometry (sliding: head_dim 256, 512-key window; full: head_dim 512, causal) and query
layout (every prompt row; each context's last row), planned once per step and run by every layer that uses it;
``FlashInferPagedPrefill`` is the stateless call.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .paged_kv import paged_attention

_WORKSPACE_BYTES = 128 << 20


class PrefillPlan:
    """One FlashInfer batch-prefill wrapper (and its workspace) for one layer geometry, re-planned per step."""

    def __init__(
        self, num_heads: int, head_dim: int, window: int, page_size: int, device: torch.device
    ):
        import flashinfer

        self.num_heads, self.head_dim, self.window, self.page_size = (
            num_heads,
            head_dim,
            window,
            page_size,
        )
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            kv_layout="HND",
            backend="fa2",
        )

    def plan(
        self,
        q_lens: list[int],
        kv_lens: list[int],
        page_counts: list[int],
        page_table: torch.Tensor,
    ) -> None:
        """Plan for requests of ``q_lens`` query rows (the last rows of ``kv_lens`` keys) whose pages are the
        consecutive runs of ``page_counts`` entries of ``page_table`` (device). Lengths are host integers."""
        assert all(-(-n // self.page_size) == p for n, p in zip(kv_lens, page_counts)), (
            kv_lens,
            page_counts,
        )
        self.lengths = (list(q_lens), list(kv_lens), list(page_counts))
        self.page_table = page_table
        qo_indptr = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        kv_indptr = torch.tensor([0] + page_counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        last_page = torch.tensor(
            [n - (p - 1) * self.page_size for n, p in zip(kv_lens, page_counts)], dtype=torch.int32
        )
        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            page_table,
            last_page,
            self.num_heads,
            1,
            self.head_dim,
            self.page_size,
            causal=True,
            # Hugging Face's window keeps keys with 0 <= q - k < window; FlashInfer's window_left counts the keys
            # left of the query, so window - 1.
            window_left=self.window - 1 if self.window > 0 else -1,
            sm_scale=1.0,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )


class FlashInferPagedPrefill(OpWrapper):
    """Causal (optionally sliding-window) GQA of ``q [T, H, D]`` bf16 over one KV head in a paged FP8 cache
    ``[pages, 2, 1, page_size, D]`` (HND, possibly a strided view of a pool that interleaves layers), with the
    request layout of ``plan`` (a planned ``PrefillPlan``): request ``r``'s query rows are the last ``q_lens[r]``
    positions of its ``kv_lens[r]`` keys. Softmax scale 1 (Gemma-4 normalizes q and k). Returns ``[T, H, D]`` bf16.
    """

    ARCHS = frozenset({Arch.SM_120})
    CELLS = (
        Cell(
            "sliding layer prefill: 8 heads, D 256, window 512",
            {"heads": 8, "head_dim": 256, "window": 512},
        ),
        Cell("full layer prefill: 8 heads, D 512", {"heads": 8, "head_dim": 512, "window": 0}),
    )
    note = "the plan must have been made with this step's exact lengths and the cache's page table"

    def __call__(
        self,
        plan: PrefillPlan,
        q: torch.Tensor,
        cache: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache.dtype == torch.float8_e4m3fn and tuple(cache.shape[1:3]) == (2, 1), (
            cache.dtype,
            cache.shape,
        )
        assert (
            tuple(q.shape[1:]) == (plan.num_heads, plan.head_dim)
            and cache.shape[-1] == plan.head_dim
        )
        return plan.wrapper.run(q, (cache[:, 0], cache[:, 1]), out=out)

    def reference(self, plan, q, cache, out=None):
        """fp32 attention over the dequantized cache with the planned layout (the Triton kernel's reference, each
        row anchored at its own position)."""
        q_lens, kv_lens, page_counts = plan.lengths
        device = q.device
        token_req = torch.repeat_interleave(
            torch.arange(len(q_lens), dtype=torch.int32), torch.tensor(q_lens)
        ).to(device)
        positions = torch.cat([torch.arange(n - m, n) for m, n in zip(q_lens, kv_lens)]).to(
            device, torch.int32
        )
        starts = (torch.tensor([0] + page_counts[:-1]).cumsum(0)).to(device, torch.int32)
        res = paged_attention.reference(
            q, cache, plan.page_table, starts, token_req, positions, None, 1, plan.window
        )
        if out is not None:
            out.copy_(res)
        return res


flashinfer_paged_prefill = FlashInferPagedPrefill()
