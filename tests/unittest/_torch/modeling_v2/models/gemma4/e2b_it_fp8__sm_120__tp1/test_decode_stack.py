# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decoder-stack megakernel against the Kernel Factory reference, on the engine's paged FP8 KV layout.

Random weights with the real checkpoint's shapes, dtypes and FP8 statistics (``_kf_reference.generate_inputs``,
ModelOpt-style calibrated input scales), random per-request page tables in two pools that interleave their layers
per page, as the KV cache manager does. Checked with the reference's statistical checker (FP8 static activation
quantization makes the 35-layer output chaotic at the 1-ulp level, so element-wise tolerances cannot be used):
hidden-state relative L2 / per-token cosine, the new K/V rows per layer, and every other cache byte untouched.
"""

import _kf_reference as kf
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="the megakernel is compiled for sm120",
)

PAGE_SIZE = 32


def _ops():
    from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4.e2b_it_fp8__sm_120__tp1.ops import (
        decode_stack as ds,
    )

    return ds


class _PagedCaches:
    """KF contiguous caches ``[layers, R, L, D]`` copied into two paged pools (sliding: 12 layers, full: 3)."""

    def __init__(self, inputs, num_requests, max_ctx, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.pages_per_request = max_ctx // PAGE_SIZE
        total = num_requests * self.pages_per_request + 5
        self.pools, self.tables = {}, {}
        for kind, (layers, hd) in {"s": (12, 256), "f": (3, 512)}.items():
            # [pages, layers, K/V, page_size, D]: the layers of a pool interleaved per page
            pool = torch.randint(
                0, 255, (total, layers, 2, PAGE_SIZE, hd), dtype=torch.uint8, generator=gen
            )
            self.pools[kind] = pool.cuda().view(torch.float8_e4m3fn)
            perm = torch.randperm(total, generator=gen)[: num_requests * self.pages_per_request]
            self.tables[kind] = perm.to(torch.int32).cuda()
        self.indptr = torch.arange(
            0,
            num_requests * self.pages_per_request + 1,
            self.pages_per_request,
            dtype=torch.int32,
            device="cuda",
        )
        # Per KV-producing layer 0..14 the layer's [pages, 2, 1, page_size, D] view.
        self.layer_views, js, jf = [], 0, 0
        for layer in range(15):
            if layer % 5 == 4:
                self.layer_views.append(self.pools["f"][:, jf].unsqueeze(2))
                jf += 1
            else:
                self.layer_views.append(self.pools["s"][:, js].unsqueeze(2))
                js += 1
        self._scatter(inputs, num_requests)

    def _layer_pages(self, layer, r):
        table = self.tables["f" if layer % 5 == 4 else "s"]
        return table[r * self.pages_per_request : (r + 1) * self.pages_per_request].long()

    def _kv_source(self, inputs, layer):
        j = layer // 5 if layer % 5 == 4 else layer - layer // 5
        if layer % 5 == 4:
            return inputs["k_cache_full"][j], inputs["v_cache_full"][j]
        return inputs["k_cache_swa"][j], inputs["v_cache_swa"][j]

    def _scatter(self, inputs, num_requests):
        for layer in range(15):
            k, v = self._kv_source(inputs, layer)
            view = self.layer_views[layer]
            for r in range(num_requests):
                pages = self._layer_pages(layer, r)
                view[pages, 0, 0] = k[r].reshape(len(pages), PAGE_SIZE, -1)
                view[pages, 1, 0] = v[r].reshape(len(pages), PAGE_SIZE, -1)

    def gather(self, num_requests):
        """Back to the KF contiguous layout."""
        out = {}
        for name, layers in (("swa", [i for i in range(15) if i % 5 != 4]), ("full", [4, 9, 14])):
            ks, vs = [], []
            for layer in layers:
                view = self.layer_views[layer]
                kr, vr = [], []
                for r in range(num_requests):
                    pages = self._layer_pages(layer, r)
                    kr.append(view[pages, 0, 0].reshape(-1, view.shape[-1]))
                    vr.append(view[pages, 1, 0].reshape(-1, view.shape[-1]))
                ks.append(torch.stack(kr))
                vs.append(torch.stack(vr))
            out[f"k_cache_{name}"], out[f"v_cache_{name}"] = torch.stack(ks), torch.stack(vs)
        return out

    def snapshot(self):
        return {k: v.clone() for k, v in self.pools.items()}

    def restore(self, snap):
        for k in self.pools:
            self.pools[k].copy_(snap[k])

    def written_bytes(self, cache_lens, q_len):
        """Mask of the pool bytes the step may write: the new rows of every layer of every request."""
        masks = {
            k: torch.zeros(v.shape[:4], dtype=torch.bool, device="cuda")
            for k, v in self.pools.items()
        }
        js, jf = 0, 0
        for layer in range(15):
            kind, j = ("f", jf) if layer % 5 == 4 else ("s", js)
            jf, js = (jf + 1, js) if kind == "f" else (jf, js + 1)
            for r, c in enumerate(cache_lens):
                pages = self._layer_pages(layer, r)
                for t in range(c, c + q_len):
                    masks[kind][pages[t // PAGE_SIZE], j, :, t % PAGE_SIZE] = True
        return masks


def _weights(inputs, ds):
    return [inputs[name].contiguous() for name in ds.WEIGHT_NAMES]


def _run_op(ds, inputs, caches, cache_lens, out):
    ds.decode_stack(
        inputs["hidden_states"],
        inputs["ple_token_embeds"],
        cache_lens,
        caches.layer_views,
        caches.tables["s"],
        caches.tables["f"],
        caches.indptr,
        _weights(inputs, ds),
        out,
    )
    return out


CASES = [
    (1, 5, [4180]),
    (1, 1, [4100]),
    (2, 5, [30, 600]),
    (4, 8, [1, 511, 512, 4000]),
    (8, 6, [4180 + 37 * i + (i * 13) % PAGE_SIZE for i in range(8)]),
    (3, 4, [1000, 2048, 3333]),
]


@pytest.mark.parametrize("num_requests, q_len, ctx", CASES, ids=[f"R{r}Q{q}" for r, q, _ in CASES])
def test_decode_stack_matches_reference(num_requests, q_len, ctx):
    ds = _ops()
    max_ctx = 4608
    inputs = kf.generate_inputs(
        {"num_requests": num_requests, "tokens_per_request": q_len, "max_ctx": max_ctx},
        torch.device("cuda"),
        7,
        ctx_lens=ctx,
    )
    caches = _PagedCaches(inputs, num_requests, max_ctx)
    before = caches.snapshot()
    cache_lens = inputs["cache_lens"]
    out = torch.empty(num_requests * q_len, 1536, dtype=torch.bfloat16, device="cuda")
    _run_op(ds, inputs, caches, cache_lens, out)
    torch.cuda.synchronize()

    ref_caches = {
        k: inputs[k].clone() for k in ("k_cache_swa", "v_cache_swa", "k_cache_full", "v_cache_full")
    }
    args = [ref_caches.get(n, inputs.get(n)) for n in kf.INPUT_NAMES]
    ref_out = kf.run(*args)
    user = {"out_hidden": out, **caches.gather(num_requests)}
    res = kf.check_outputs(
        {"out_hidden": ref_out, **ref_caches}, user, ctx_lens=ctx, tokens_per_request=q_len
    )
    failed = {k: v["metadata"] for k, v in res.items() if not v["passed"]}
    assert not failed, failed

    # Bytes outside the new rows -- other layers' slots, pages of no request, padding -- are untouched.
    masks = caches.written_bytes(ctx, q_len)
    for kind, pool in caches.pools.items():
        changed = (pool.view(torch.uint8) != before[kind].view(torch.uint8)).any(-1)
        assert not bool((changed & ~masks[kind]).any()), f"{kind} pool written outside the new rows"


def test_decode_stack_graph_replay_with_new_lengths():
    """Captured once, replayed with other cache lengths: equal to an eager launch at those lengths, bit for bit."""
    ds = _ops()
    num_requests, q_len, max_ctx = 4, 5, 4608
    ctx = [100, 1200, 2500, 4000]
    inputs = kf.generate_inputs(
        {"num_requests": num_requests, "tokens_per_request": q_len, "max_ctx": max_ctx},
        torch.device("cuda"),
        3,
        ctx_lens=ctx,
    )
    caches = _PagedCaches(inputs, num_requests, max_ctx)
    snap = caches.snapshot()
    lens = inputs["cache_lens"].clone()
    out = torch.empty(num_requests * q_len, 1536, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_op(ds, inputs, caches, lens, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_op(ds, inputs, caches, lens, out)
    lens.add_(7)
    caches.restore(snap)
    graph.replay()
    torch.cuda.synchronize()
    replayed, replayed_pools = out.clone(), caches.snapshot()
    caches.restore(snap)
    eager = _run_op(ds, inputs, caches, lens, torch.empty_like(out))
    torch.cuda.synchronize()
    assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16))
    for kind in caches.pools:
        assert torch.equal(
            replayed_pools[kind].view(torch.uint8), caches.pools[kind].view(torch.uint8)
        )


def test_op_reference_matches_kf_reference():
    """The op's own PyTorch ``reference`` (paged) agrees with the KF reference (contiguous) under the checker."""
    ds = _ops()
    num_requests, q_len, max_ctx, ctx = 2, 3, 1024, [700, 1000]
    inputs = kf.generate_inputs(
        {"num_requests": num_requests, "tokens_per_request": q_len, "max_ctx": max_ctx},
        torch.device("cuda"),
        11,
        ctx_lens=ctx,
    )
    caches = _PagedCaches(inputs, num_requests, max_ctx)
    out = torch.empty(num_requests * q_len, 1536, dtype=torch.bfloat16, device="cuda")
    ds.decode_stack.reference(
        inputs["hidden_states"],
        inputs["ple_token_embeds"],
        inputs["cache_lens"],
        caches.layer_views,
        caches.tables["s"],
        caches.tables["f"],
        caches.indptr,
        _weights(inputs, ds),
        out,
    )
    ref_caches = {
        k: inputs[k].clone() for k in ("k_cache_swa", "v_cache_swa", "k_cache_full", "v_cache_full")
    }
    ref_out = kf.run(*[ref_caches.get(n, inputs.get(n)) for n in kf.INPUT_NAMES])
    res = kf.check_outputs(
        {"out_hidden": ref_out, **ref_caches},
        {"out_hidden": out, **caches.gather(num_requests)},
        ctx_lens=ctx,
        tokens_per_request=q_len,
    )
    failed = {k: v["metadata"] for k, v in res.items() if not v["passed"]}
    assert not failed, failed
