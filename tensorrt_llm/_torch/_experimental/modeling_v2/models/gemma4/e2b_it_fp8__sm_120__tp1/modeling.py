# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Gemma-4 E2B-it (ModelOpt FP8) / sm_120 / tp1, one-model MTP fused into the forward.

The text-only Gemma-4 E2B-it checkpoint quantized with ModelOpt FP8 (per-tensor static scales, FP8 lm_head), on
one RTX PRO 6000 Blackwell Server Edition, with an FP8 paged KV cache. 35 decoder layers, hidden 1536, per-layer
embeddings (PLE); every fifth layer is full attention (head_dim 512, proportional RoPE), the others sliding-window
(512 keys, head_dim 256); layers 15-34 compute only queries and read the KV of layers 13 / 14, and have a
double-wide MLP.

**Two execution paths, chosen per step from host integers.**

* Decode (no context requests, at most 8 requests of at most 8 tokens -- every CUDA-graph-captured step): one
  launch of the decoder-stack megakernel (``ops/decode_stack.py``, native code in ``csrc/``) computes the PLE
  projection, the 35 layers and the final norm, reading and writing the paged FP8 KV in place.
* Everything else (prefill, mixed batches): the same math layer by layer with PyTorch ops, FP8 ``_scaled_mm``
  and this target's Triton paged-KV append and attention (``ops/paged_kv.py``); the context requests' rows attend
  with FlashInfer's batch prefill (``ops/flashinfer_prefill.py``, called directly and planned per step with the
  host-known prompt lengths). Layers 15-34 write no KV, so they run only on the rows the step reads: each context
  request's last prompt row and the generation rows.

**Speculative decoding runs in the model.** With one-model MTP and the Gemma-4 assistant checkpoint
(``Gemma4AssistantForCausalLM``: 4 Q-only layers of width 256 that read the target's committed KV of layers 13 and
14), the forward also computes the verify logits, strict acceptance and the ``K`` draft steps
(``Gemma4MtpStep``), and returns the dict the speculative sampler reads. No speculative worker is built. Without
speculation the forward returns logits. Routing (``../routing.py``) admits nothing else.

**Attention metadata.** The engine prepares FlashInfer metadata (Gemma-4's default on this GPU); the target reads
only its per-pool page tables and page offsets and never plans or calls the backend's FlashInfer wrappers, so the
FlashInfer restrictions on SM120 (draft views and reservation-width page tables need trtllm-gen) do not apply. Its
own FlashInfer prefill serves context rows only, planned with their exact lengths (no runtime KV length). Token
positions come from ``position_ids``, which the engine corrects on the device under the overlap scheduler. The
draft chain leaves the metadata untouched (it never runs the engine's attention).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    disable_v2_by_default,
    step_contract_enabled,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import KVCacheLayerSpec, ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.sampler.ops import custom as fused_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_tail_kernels import strict_accept
from tensorrt_llm.quantization.mode import QuantAlgo

from ..routing import speculation_refused
from . import weights as _weights
from .ops.decode_stack import (
    EPS,
    HEAD_DIM_FULL,
    HEAD_DIM_SLIDING,
    HIDDEN,
    KV_SOURCE_FULL,
    KV_SOURCE_SLIDING,
    MAX_REQUESTS,
    MAX_TOKENS_PER_REQUEST,
    NUM_HEADS,
    NUM_LAYERS,
    PLE_DIM,
    SLIDING_WINDOW,
    WEIGHT_NAMES,
    decode_stack,
    head_dim_of,
    layer_kind,
    rope_inv_freq,
)
from .ops.elementwise import gelu_tanh_mul_fp8, qk_norm_rope, quantize_fp8
from .ops.flashinfer_prefill import PrefillPlan, flashinfer_paged_prefill
from .ops.norm import rms_norm
from .ops.paged_kv import paged_attention, paged_kv_append, request_blocks

_SM = (12, 0)
_SOFTCAP = 30.0
# Flash-decoding splits of the assistant's attention (one query row per request over the committed KV): the
# full layer reads up to the whole context, the sliding layers at most 512 keys.
_DRAFT_SPLITS_FULL = 16
_DRAFT_SPLITS_SLIDING = 4
_NUM_KV_LAYERS = 15


# ---------------------------------------------------------------------------
# Small math helpers (bf16 activations, fp32 internals)
# ---------------------------------------------------------------------------


def _rms(
    x: torch.Tensor, weight: torch.Tensor | None = None, residual: torch.Tensor | None = None
) -> torch.Tensor:
    """Gemma-4 RMSNorm (``x * rsqrt(mean(x^2) + eps) * weight``), then ``+ residual`` when given; one kernel."""
    return rms_norm(x, weight, residual, EPS)


def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """ModelOpt per-tensor FP8 linear: static activation quantization, then an FP8 GEMM with both scales."""
    return _fp8_gemm(quantize_fp8(x, input_scale), weight, weight_scale, input_scale, out_dtype)


def _fp8_gemm(
    xq: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``(xq @ weight^T) * input_scale * weight_scale`` of already quantized activations."""
    return torch._scaled_mm(
        xq,
        weight.t(),
        scale_a=input_scale.reshape(()).float(),
        scale_b=weight_scale.reshape(()).float(),
        out_dtype=out_dtype,
    )


def _gelu_tanh_mul(gate_up: torch.Tensor) -> torch.Tensor:
    inter = gate_up.shape[-1] // 2
    return (F.gelu(gate_up[:, :inter].float(), approximate="tanh") * gate_up[:, inter:].float()).to(
        torch.bfloat16
    )


class _Rope:
    """cos / sin tables of the sliding (head_dim 256) and full (head_dim 512, proportional) RoPE."""

    def __init__(self, max_positions: int, device: torch.device):
        positions = torch.arange(max_positions, device=device, dtype=torch.float32)
        self.tables = {}
        for head_dim in (HEAD_DIM_SLIDING, HEAD_DIM_FULL):
            ang = positions[:, None] * rope_inv_freq(head_dim, device)[None]
            self.tables[head_dim] = (ang.cos(), ang.sin())

    def at(self, positions: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos / sin rows of ``positions``, shaped to broadcast over heads."""
        cos, sin = self.tables[head_dim]
        return cos[positions].unsqueeze(1), sin[positions].unsqueeze(1)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        xf = x.float()
        x1, x2 = xf[..., :half], xf[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# KV cache view: the engine's paged pools, through the FlashInfer metadata's page tables
# ---------------------------------------------------------------------------


class _KvView:
    """The 15 KV-producing layers' cache buffers and the sliding / full pool page tables of one KV manager."""

    def __init__(self, md: AttentionMetadata):
        self.manager = md.kv_cache_manager
        self.caches = [
            self.manager.get_buffers(layer, kv_layout="HND") for layer in range(_NUM_KV_LAYERS)
        ]
        for layer, cache in enumerate(self.caches):
            assert cache.dtype == torch.float8_e4m3fn and tuple(cache.shape[1:3]) == (2, 1), (
                layer,
                cache.dtype,
                tuple(cache.shape),
            )
            assert cache.shape[-1] == head_dim_of(layer) and cache.stride(-1) == 1, (
                layer,
                cache.stride(),
            )
            assert cache.stride(3) == cache.shape[-1], (layer, cache.stride())
        self.page_size = self.caches[0].shape[3]
        pools = md._vswa_layer_to_pool
        self.pool_sliding = None if pools is None else pools[KV_SOURCE_SLIDING]
        self.pool_full = None if pools is None else pools[KV_SOURCE_FULL]
        if pools is not None:
            assert {pools[layer] for layer in range(_NUM_KV_LAYERS) if layer % 5 != 4} == {
                self.pool_sliding
            }
            assert {pools[layer] for layer in range(_NUM_KV_LAYERS) if layer % 5 == 4} == {
                self.pool_full
            }

    def page_table(self, md: AttentionMetadata, sliding: bool) -> torch.Tensor:
        pool = self.pool_sliding if sliding else self.pool_full
        if pool is None or md._vswa_pool_indices_cache is None:
            return md._paged_kv_indices
        return md._vswa_pool_indices_cache[pool]

    @staticmethod
    def request_starts(md: AttentionMetadata) -> torch.Tensor:
        """``[num_seqs]`` int32: each request's first entry in the flat page tables (contexts first)."""
        return _KvView._indptr(md)[:-1]

    @staticmethod
    def request_pages(md: AttentionMetadata) -> torch.Tensor:
        """``[num_seqs]`` int32: the number of pages each request's page-table row holds."""
        indptr = _KvView._indptr(md)
        return indptr[1:] - indptr[:-1]

    @staticmethod
    def _indptr(md: AttentionMetadata) -> torch.Tensor:
        """``[num_seqs + 1]`` offsets of the requests' rows in the flat page tables (contexts first)."""
        num_contexts, num_gens = md.num_contexts, md.num_generations
        if num_contexts == 0:
            return md.paged_kv_indptr_decode[: num_gens + 1]
        contexts = md.paged_kv_indptr_prefill[: num_contexts + 1]
        if num_gens == 0:
            return contexts
        return torch.cat(
            [contexts, md.paged_kv_indptr_decode[1 : num_gens + 1] + md.num_context_blocks]
        )


def _check_metadata_surface(md: AttentionMetadata) -> None:
    """``TRTLLM_MODELING_V2_VALIDATE=1``: the (private) FlashInfer metadata fields this target reads exist."""
    for name in (
        "kv_cache_manager",
        "_vswa_layer_to_pool",
        "_vswa_pool_indices_cache",
        "_paged_kv_indices",
        "paged_kv_indptr_decode",
        "paged_kv_indptr_prefill",
        "num_context_blocks",
        "num_blocks",
        "kv_cache_params",
        "num_contexts",
        "num_generations",
        "seq_lens",
        "seq_lens_cuda",
    ):
        if not hasattr(md, name):
            raise AssertionError(
                f"attention metadata {type(md).__name__} has no {name}; this target reads FlashInfer metadata"
            )


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------


class Gemma4E2bModel(DecoderModel):
    """Embeddings + PLE + 35 decoder layers + final norm; returns the final hidden states ``[T, 1536]``."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.w = _weights.declare_target()
        # The scaled word embedding multiplies by sqrt(hidden) rounded to bf16 (39.25, not 39.19).
        self.embed_scale = torch.tensor(math.sqrt(HIDDEN), dtype=torch.bfloat16).item()
        spec_config = model_config.spec_config
        # Overlap can hold one pending verify block while the current one and the drafter use the next.
        headroom = 0 if spec_config is None else 2 * (spec_config.max_draft_len + 1)
        self.max_positions = model_config.pretrained_config.max_position_embeddings + headroom
        self._rope: _Rope | None = None
        self._kv: _KvView | None = None
        self._stack_weights: list[torch.Tensor] | None = None
        self._validate = step_contract_enabled()
        # Prefill runs the KV-shared layers 15-34 on the rows the step reads only (``shared_kv_rows``).
        self.skip_shared_kv_layers = True
        # Context rows attend with FlashInfer's batch prefill (planned per step from host lengths).
        self.flashinfer_prefill = True
        self._prefill_plans: dict[tuple[int, bool], PrefillPlan] | None = None

    def post_load_weights(self) -> None:
        device = self.w["embed_tokens"].device
        self._rope = _Rope(self.max_positions, device)
        self._stack_weights = [self.w[name] for name in WEIGHT_NAMES]

    # -- step inputs ---------------------------------------------------------

    def kv_view(self, md: AttentionMetadata) -> _KvView:
        if self._kv is None or self._kv.manager is not md.kv_cache_manager:
            if self._validate:
                _check_metadata_surface(md)
            self._kv = _KvView(md)
        return self._kv

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``embed_tokens(ids) * bf16(sqrt(1536))``, as the checkpoint's scaled word embedding computes it."""
        return F.embedding(input_ids, self.w["embed_tokens"]) * self.embed_scale

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        all_rows: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Final hidden states ``[T, 1536]``. Unless ``all_rows``, a prefill / mixed step computes only the rows the
        step reads -- each context request's last row and every generation row -- and leaves the others zero."""
        assert input_ids is not None and inputs_embeds is None, "the target embeds token ids itself"
        md = attn_metadata
        num_tokens = input_ids.shape[0]
        positions = position_ids.reshape(-1)[:num_tokens]
        hidden = self.embed(input_ids)
        ple_tokens = (
            F.embedding(input_ids, self.w["embed_tokens_per_layer"]) * float(PLE_DIM) ** 0.5
        ).view(num_tokens, NUM_LAYERS, PLE_DIM)
        kv = self.kv_view(md)
        num_requests = md.num_seqs
        if (
            md.num_contexts == 0
            and 1 <= num_requests <= MAX_REQUESTS
            and num_tokens % num_requests == 0
            and num_tokens // num_requests <= MAX_TOKENS_PER_REQUEST
            and kv.page_size == 32
        ):
            return self._decode(md, kv, hidden, ple_tokens, positions)
        assert not torch.cuda.is_current_stream_capturing(), (
            "the layer-by-layer path builds its block tables on the host and is not CUDA-graph capturable; "
            f"got a captured step with {md.num_contexts} contexts and {num_tokens} tokens for {num_requests} requests"
        )
        return self._layers(md, kv, hidden, ple_tokens, positions, all_rows)

    # -- decode: the megakernel ----------------------------------------------

    def _decode(self, md, kv: _KvView, hidden, ple_tokens, positions) -> torch.Tensor:
        num_requests = md.num_seqs
        q_len = hidden.shape[0] // num_requests
        indptr = md.paged_kv_indptr_decode[: num_requests + 1]
        pages = indptr[1:] - indptr[:-1]
        # Cached tokens per request = the position of its first new token. Clamped to the request's addressable
        # pages, so CUDA-graph padding rows (whose positions are arbitrary) can never write outside their pages.
        cache_lens = positions[::q_len][:num_requests]
        cache_lens = torch.minimum(
            cache_lens.clamp(min=0), (pages * kv.page_size - q_len).clamp(min=0)
        ).to(torch.int32)
        out = torch.empty_like(hidden)
        decode_stack(
            hidden,
            ple_tokens.contiguous(),
            cache_lens,
            kv.caches,
            kv.page_table(md, sliding=True),
            kv.page_table(md, sliding=False),
            indptr,
            self._stack_weights,
            out,
        )
        return out

    # -- prefill / mixed: layer by layer -------------------------------------

    def _plan_prefill(self, md, kv: _KvView, tables) -> dict[tuple[int, bool], PrefillPlan] | None:
        """FlashInfer plans of the step's context requests, keyed (head_dim, last row only): every prompt row
        (layers 0-14) and each context's last row (layers 15-34 under the skip). None without contexts."""
        num_contexts = md.num_contexts
        if not self.flashinfer_prefill or num_contexts == 0:
            return None
        if self._prefill_plans is None:
            device = kv.caches[0].device
            self._prefill_plans = {
                (head_dim, last): PrefillPlan(
                    NUM_HEADS,
                    head_dim,
                    SLIDING_WINDOW if head_dim == HEAD_DIM_SLIDING else 0,
                    kv.page_size,
                    device,
                )
                for head_dim in (HEAD_DIM_SLIDING, HEAD_DIM_FULL)
                for last in (False, True)
            }
        # Host lengths: no chunked prefill and no block reuse are routed, but the cached prefix is honored.
        q_lens = md.seq_lens[:num_contexts].tolist()
        cached = md.kv_cache_params.num_cached_tokens_per_seq[:num_contexts]
        kv_lens = [c + n for c, n in zip(cached, q_lens)]
        page_counts = list(md.num_blocks[:num_contexts])
        for (head_dim, last), plan in self._prefill_plans.items():
            if last and not self.skip_shared_kv_layers:
                continue
            plan.plan([1] * num_contexts if last else q_lens, kv_lens, page_counts,
                      tables[head_dim == HEAD_DIM_SLIDING])  # fmt: skip
        return self._prefill_plans

    def _layers(self, md, kv: _KvView, hidden, ple_tokens, positions, all_rows) -> torch.Tensor:
        tables = {True: kv.page_table(md, sliding=True), False: kv.page_table(md, sliding=False)}
        return prefill_layers(
            self.w,
            self._rope,
            kv.caches,
            tables,
            kv.request_starts(md),
            md.seq_lens[: md.num_seqs].tolist(),
            md.num_contexts,
            hidden,
            ple_tokens,
            positions,
            skip_shared_kv_layers=self.skip_shared_kv_layers and not all_rows,
            prefill_plans=self._plan_prefill(md, kv, tables),
        )


class _StepRows:
    """The host-built row layout of one layer-by-layer pass: each row's request and, per head_dim, the attention
    block table, its tokens per block and how many of its blocks belong to the first ``num_contexts`` requests."""

    def __init__(self, seq_lens: list[int], device: torch.device, num_contexts: int = 0):
        self.num_rows = sum(seq_lens)
        self.token_req = torch.repeat_interleave(
            torch.arange(len(seq_lens), dtype=torch.int32), torch.tensor(seq_lens)
        ).to(device, non_blocking=True)
        self.blocks = {}
        self.context_blocks = {}
        for head_dim in (HEAD_DIM_SLIDING, HEAD_DIM_FULL):
            # The prefill block size also for the kept rows of the KV-shared layers (one row per context
            # block): a row then sees the same tiles, masks and accumulation order as in the full pass.
            tpb = paged_attention.tokens_per_block(NUM_HEADS, head_dim)
            self.blocks[head_dim] = (request_blocks(seq_lens, tpb, device), tpb)
            self.context_blocks[head_dim] = sum(-(-n // tpb) for n in seq_lens[:num_contexts])


def shared_kv_rows(seq_lens: list[int], num_contexts: int) -> tuple[list[int], list[int]] | None:
    """The rows the KV-shared layers (15-34) must compute, and the per-request row counts among them.

    Layers 15-34 write no KV (they read layers 13 / 14), so a context request's prompt rows feed nothing after
    layer 14 but its last row: the first-token logits and the assistant's recurrent hidden state. Generation rows
    are all kept (each one is a verify/decode logits row). ``None`` when every row is kept.
    """
    if num_contexts == 0 or all(n == 1 for n in seq_lens[:num_contexts]):
        return None
    rows, start = [], 0
    for r, n in enumerate(seq_lens):
        if r < num_contexts:
            rows.append(start + n - 1)
        else:
            rows.extend(range(start, start + n))
        start += n
    return rows, [1] * num_contexts + list(seq_lens[num_contexts:])


def prefill_layers(
    w,
    rope: _Rope,
    caches: list[torch.Tensor],
    tables: dict[bool, torch.Tensor],
    starts: torch.Tensor,
    seq_lens: list[int],
    num_contexts: int,
    hidden: torch.Tensor,
    ple_tokens: torch.Tensor,
    positions: torch.Tensor,
    skip_shared_kv_layers: bool = True,
    prefill_plans: dict[tuple[int, bool], PrefillPlan] | None = None,
) -> torch.Tensor:
    """The 35 decoder layers and the final norm, layer by layer, over requests laid out back to back (contexts
    first); appends the new K/V of layers 0-14 to the paged caches and returns the final hidden states ``[T, 1536]``.

    With ``skip_shared_kv_layers`` the KV-shared layers 15-34 run only on the rows ``shared_kv_rows`` keeps (the
    last row of each context request and every generation row); the other rows of the result are zero. Those are
    the only rows the step reads (logits and the drafter's input), and every row's KV comes from layers 0-14, so
    the cache bytes are the same either way.

    With ``prefill_plans`` (``Gemma4E2bModel._plan_prefill``), the context rows attend with FlashInfer's batch
    prefill (lengths known on the host, so planned exactly) and only the generation rows of a mixed step with the
    Triton kernel.
    """
    num_tokens = hidden.shape[0]
    device = hidden.device
    assert sum(seq_lens) == num_tokens, (seq_lens, num_tokens)
    positions = positions.to(torch.int32)
    step = _StepRows(seq_lens, device, num_contexts)
    subset = shared_kv_rows(seq_lens, num_contexts) if skip_shared_kv_layers else None
    keep = None

    proj = (hidden @ w["ple_model_proj_w"].t()) * (HIDDEN**-0.5)
    proj = _rms(proj.view(num_tokens, NUM_LAYERS, PLE_DIM), w["ple_proj_norm_w"])
    ple = (proj + ple_tokens) * (2.0**-0.5)

    h = hidden
    for i in range(NUM_LAYERS):
        if i == _NUM_KV_LAYERS and subset is not None:
            keep = torch.tensor(subset[0], dtype=torch.long).to(device, non_blocking=True)
            h, ple, positions = h[keep], ple[keep], positions[keep]
            step = _StepRows(subset[1], device, num_contexts)
        rows = step.num_rows
        kind, j = layer_kind(i)
        k = kind.lower()
        head_dim = head_dim_of(i)
        sliding = head_dim == HEAD_DIM_SLIDING
        scales = w[f"{k}_scales"][j]
        norms = w["layer_norm_w"][i]
        x = _rms(h, norms[0])
        if kind in "AB":
            qkv = _fp8_linear(x, w[f"{k}_qkv_w"][j], scales[0, 0], scales[0, 1])
            q_size = NUM_HEADS * head_dim
            cos, sin = rope.tables[head_dim]
            q = qk_norm_rope(
                qkv[:, :q_size].view(rows, NUM_HEADS, head_dim), w[f"{k}_qk_norm_w"][j, 0], cos, sin,
                positions, EPS,
            )  # fmt: skip
            key = qk_norm_rope(
                qkv[:, q_size : q_size + head_dim].view(rows, 1, head_dim), w[f"{k}_qk_norm_w"][j, 1],
                cos, sin, positions, EPS,
            )[:, 0]  # fmt: skip
            value = _rms(qkv[:, q_size + head_dim :])
            paged_kv_append(
                key, value, caches[i], tables[sliding], starts, step.token_req, positions
            )
            cache = caches[i]
            o_row = 1
        else:
            cos, sin = rope.tables[head_dim]
            q = (x @ w[f"{k}_q_w"][j].t()).view(rows, NUM_HEADS, head_dim)
            q = qk_norm_rope(q, w[f"{k}_q_norm_w"][j], cos, sin, positions, EPS)
            cache = caches[KV_SOURCE_SLIDING if sliding else KV_SOURCE_FULL]
            o_row = 0
        block_table, tpb = step.blocks[head_dim]
        window = SLIDING_WINDOW if sliding else 0
        if prefill_plans is None:
            attn = paged_attention(
                q,
                cache,
                tables[sliding],
                starts,
                step.token_req,
                positions,
                block_table,
                tpb,
                window,
            )
        else:
            plan = prefill_plans[(head_dim, keep is not None)]
            ctx_rows = num_contexts if keep is not None else sum(seq_lens[:num_contexts])
            if ctx_rows == rows:
                attn = flashinfer_paged_prefill(plan, q, cache)
            else:
                # Generation rows of a mixed step: Triton over their blocks only (blocks never span requests,
                # and the contexts' rows come first), then FlashInfer writes the context rows.
                gen_blocks = block_table[step.context_blocks[head_dim] :]
                attn = paged_attention(
                    q,
                    cache,
                    tables[sliding],
                    starts,
                    step.token_req,
                    positions,
                    gen_blocks,
                    tpb,
                    window,
                )
                flashinfer_paged_prefill(plan, q[:ctx_rows], cache, out=attn[:ctx_rows])
        o = _fp8_linear(attn.view(rows, -1), w[f"{k}_o_w"][j], scales[o_row, 0], scales[o_row, 1])
        h = _rms(o, norms[1], h)
        gate_up = _fp8_linear(
            _rms(h, norms[2]), w[f"{k}_gu_w"][j], scales[o_row + 1, 0], scales[o_row + 1, 1]
        )
        down = _fp8_gemm(
            gelu_tanh_mul_fp8(gate_up, scales[o_row + 2, 1]),
            w[f"{k}_down_w"][j],
            scales[o_row + 2, 0],
            scales[o_row + 2, 1],
        )
        h = _rms(down, norms[3], h)
        gate = F.gelu(h @ w["ple_gate_w"][i].t(), approximate="tanh") * ple[:, i]
        h = _rms(gate @ w["ple_proj_w"][i].t(), norms[4], h)
        h = (h.float() * w["layer_scalar"][i]).to(torch.bfloat16)
    h = _rms(h, w["final_norm_w"])
    if keep is None:
        return h
    return h.new_zeros(num_tokens, HIDDEN).index_copy_(0, keep, h)


# ---------------------------------------------------------------------------
# The drafter: the Gemma-4 assistant
# ---------------------------------------------------------------------------


class Gemma4AssistantDraft(nn.Module):
    """The Gemma-4 assistant: 4 Q-only layers (width 256) over the target's committed KV of layers 13 / 14.

    One draft step for ``B`` rows (one per request):

        x  = pre_projection(cat(target_embed(token), h_prev))              [B, 256]
        4 x (RMSNorm -> q_proj -> q_norm -> RoPE at the draft position -> attention over the target's KV, anchored
             at kv_len - 1 -> o_proj -> norm + residual -> gelu-tanh MLP -> norm + residual -> * layer_scalar)
        h  = RMSNorm(x);  h_next = post_projection(h)                       [B, 1536]
        token = argmax over the 32 x 128 candidates of the top-32 centroids of h . embed_tokens[candidate]

    The engine loads its checkpoint into this module through ``load_draft_weights`` (``draft_config`` is set), and
    reads ``model_config`` / ``config`` off it.
    """

    def __init__(self, draft_config: ModelConfig):
        super().__init__()
        self.model_config = draft_config
        cfg = draft_config.pretrained_config
        self.cfg = cfg
        tc = cfg.text_config
        self.num_layers = tc.num_hidden_layers
        self.layer_types = list(tc.layer_types)
        self.num_heads = tc.num_attention_heads
        self.head_dims = [
            tc.head_dim if t == "sliding_attention" else tc.global_head_dim
            for t in self.layer_types
        ]
        self.num_centroids = cfg.num_centroids
        self.top_k = cfg.centroid_intermediate_top_k
        self.vocab_size = tc.vocab_size
        self.w = _weights.declare_assistant(cfg)
        self.register_buffer(
            "token_ordering", torch.empty(self.vocab_size, dtype=torch.long), persistent=False
        )

    @property
    def config(self):
        return self.cfg

    def load_weights(self, weights, weight_mapper=None, **kwargs) -> None:
        _weights.load_assistant(self, weights)

    def step(
        self,
        tokens: torch.Tensor,
        prev_hidden: torch.Tensor,
        rope: dict[int, tuple[torch.Tensor, torch.Tensor]],
        target: Gemma4E2bModel,
        kv: _KvView,
        tables: dict[bool, torch.Tensor],
        starts: torch.Tensor,
        rows: torch.Tensor,
        anchors: torch.Tensor,
        blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft step; returns (draft tokens [B] int32, next recurrent hidden [B, 1536])."""
        w = self.w
        num_rows = tokens.shape[0]
        x = torch.cat([target.embed(tokens), prev_hidden], -1) @ w["pre_projection"].t()
        for i in range(self.num_layers):
            head_dim = self.head_dims[i]
            sliding = self.layer_types[i] == "sliding_attention"
            norms = w[f"l{i}_norms"]
            q = (_rms(x, norms[0]) @ w[f"l{i}_q_proj"].t()).view(num_rows, self.num_heads, head_dim)
            q = _Rope.apply(_rms(q, w[f"l{i}_q_norm"]), *rope[head_dim])
            attn = paged_attention(
                q.contiguous(),
                kv.caches[KV_SOURCE_SLIDING if sliding else KV_SOURCE_FULL],
                tables[sliding],
                starts,
                rows,
                anchors,
                blocks,
                1,
                SLIDING_WINDOW if sliding else 0,
                num_splits=_DRAFT_SPLITS_SLIDING if sliding else _DRAFT_SPLITS_FULL,
            )
            x = _rms(attn.view(num_rows, -1) @ w[f"l{i}_o_proj"].t(), norms[1], x)
            mlp = _gelu_tanh_mul(_rms(x, norms[2]) @ w[f"l{i}_gate_up"].t()) @ w[f"l{i}_down"].t()
            x = _rms(mlp, norms[3], x)
            x = (x.float() * w[f"l{i}_layer_scalar"]).to(torch.bfloat16)
        h = _rms(x, w["norm"])
        next_hidden = h @ w["post_projection"].t()
        return self._masked_argmax(h), next_hidden

    def _masked_argmax(self, h: torch.Tensor) -> torch.Tensor:
        """Greedy draft token of the centroid-masked head: the best of the top-k centroids' tokens (smallest id on
        ties, as an argmax over the full masked vocabulary picks)."""
        _, top = torch.topk(h @ self.w["centroids"].t(), k=self.top_k, dim=-1)
        per_centroid = self.vocab_size // self.num_centroids
        candidates = self.token_ordering.view(self.num_centroids, per_centroid)[top].flatten(1)
        logits = torch.bmm(
            h.unsqueeze(1), self.w["embed_tokens"][candidates].transpose(1, 2)
        ).squeeze(1)
        best = logits.max(-1, keepdim=True).values
        ids = torch.where(logits == best, candidates, torch.iinfo(torch.long).max)
        return ids.min(-1).values.to(torch.int32)


# ---------------------------------------------------------------------------
# The speculative step
# ---------------------------------------------------------------------------


class DraftChainGraphs:
    """CUDA graphs of the assistant's ``K``-step draft chain, for the steps the engine runs eagerly.

    Decode steps are captured whole by the engine, but a prefill or mixed step runs the forward eagerly, and its
    draft chain is ~80 small launches per draft step: host-bound, about 1.3 ms per step on top of a GPU-bound
    prefill, so ~6 ms of TTFT at K=5 and batch 1. One graph per ``key`` (the caller's: request count, ``K`` and the
    addresses of the tensors ``chain`` reads besides its inputs), captured on first use. The inputs are copied into
    the graph's static buffers; the result is cloned out (a later replay reuses the buffer). Same kernels on the
    same shapes as the eager chain, so the drafts are the same.
    """

    def __init__(self) -> None:
        self._graphs: dict[
            tuple, tuple[torch.cuda.CUDAGraph, dict[str, torch.Tensor], torch.Tensor]
        ] = {}
        self._pool = None

    def __call__(self, chain, inputs: dict[str, torch.Tensor], key: tuple) -> torch.Tensor:
        key = key + tuple((name, tuple(t.shape), t.dtype) for name, t in inputs.items())
        entry = self._graphs.get(key)
        if entry is None:
            static = {name: t.clone() for name, t in inputs.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                chain(**static)  # Triton compilation and cuBLAS handles happen outside the capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                out = chain(**static)
            self._pool = graph.pool()
            entry = self._graphs[key] = (graph, static, out)
        graph, static, out = entry
        for name, t in inputs.items():
            static[name].copy_(t)
        graph.replay()
        return out.clone()


class Gemma4MtpStep:
    """One engine step of one-model MTP with the Gemma-4 assistant, after the target decoder ran.

    1. Target tokens of the verify rows (argmax for an all-greedy batch, else each row's sampling parameters),
       strict prefix acceptance of the previous drafts (``strict_accept``).
    2. ``K`` draft steps of the assistant, one row per request: the last accepted token (a context request's next
       prompt token when its chunk is not the last) and the target's final hidden state of that row; the draft
       position is one past it and stays fixed across steps; the queries read the committed KV only.
    3. Return what the speculative sampler reads. Shapes depend on (batch size, contexts, K) only.

    In the steps the engine runs eagerly (prefill and mixed batches) the draft chain replays a CUDA graph
    (``DraftChainGraphs``); captured decode steps run it inline.
    """

    def __init__(self) -> None:
        self.graph_draft_chain = True
        self._draft_graphs = DraftChainGraphs()

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Any,
        target: Gemma4E2bModel,
        draft: Gemma4AssistantDraft,
    ) -> dict[str, torch.Tensor]:
        if spec_metadata.is_ragged_verify:
            raise ValueError(
                "the Gemma-4 E2B modeling_v2 target verifies a uniform draft width only"
            )
        md = attn_metadata
        batch_size = md.num_seqs
        num_contexts = md.num_contexts
        num_gens = batch_size - num_contexts
        draft_len = spec_metadata.runtime_draft_len
        target_tokens = self._sample_target(logits, spec_metadata)
        if draft_len == 0:
            tokens = target_tokens[:batch_size].to(torch.int32).unsqueeze(1)
            return {
                "logits": logits,
                "new_tokens": tokens,
                "new_tokens_lens": torch.ones(batch_size, dtype=torch.int32, device=logits.device),
                "next_draft_tokens": torch.zeros(
                    (batch_size, 0), dtype=torch.int32, device=logits.device
                ),
                "next_new_tokens": tokens.clone(),
            }
        if num_gens > 0:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, draft_len)
        else:
            draft_tokens = torch.empty((0, draft_len), dtype=torch.int32, device=logits.device)
        accepted_tokens, num_accepted = strict_accept(
            target_tokens, draft_tokens, num_contexts, batch_size
        )

        # Draft inputs: the row each request drafts from, its token and its position.
        seq_lens = md.seq_lens_cuda[:batch_size].long()
        seq_starts = torch.cumsum(seq_lens, 0) - seq_lens
        rec_rows = seq_starts + seq_lens - 1
        if num_gens > 0:
            rec_rows[num_contexts:] = (
                seq_starts[num_contexts:] + num_accepted[num_contexts:].long() - 1
            )
        batch_rows = spec_metadata.batch_indices_cuda[:batch_size].long()
        tokens = accepted_tokens[batch_rows, num_accepted.long() - 1].to(torch.int32)
        if num_contexts > 0:
            lookahead = spec_metadata.context_prompt_lookahead_tokens[:num_contexts]
            tokens[:num_contexts] = torch.where(
                lookahead != INVALID_PROMPT_LOOKAHEAD_TOKEN,
                lookahead.to(torch.int32),
                tokens[:num_contexts],
            )
        positions = position_ids.reshape(-1)[rec_rows].to(torch.int32)
        hidden = hidden_states[rec_rows]

        kv = target.kv_view(md)
        tables = {True: kv.page_table(md, sliding=True), False: kv.page_table(md, sliding=False)}
        starts = kv.request_starts(md)
        # The draft queries see the committed KV up to and including the last accepted token. Clamped to each
        # request's pages, so CUDA-graph padding rows (whose positions are arbitrary) never read past their row.
        anchors = torch.minimum(
            positions, (kv.request_pages(md) * kv.page_size - 1).to(torch.int32)
        )
        # The draft position (one past the last accepted token) is the same for every draft step and layer.
        draft_positions = (positions + 1).clamp(0, target.max_positions - 1)

        def chain(tokens, hidden, starts, anchors, draft_positions):
            rows = torch.arange(tokens.shape[0], dtype=torch.int32, device=tokens.device)
            blocks = torch.stack([rows, rows + 1], 1)
            rope = {hd: target._rope.at(draft_positions, hd) for hd in set(draft.head_dims)}
            drafts = []
            for _ in range(draft_len):
                tokens, hidden = draft.step(
                    tokens, hidden, rope, target, kv, tables, starts, rows, anchors, blocks
                )
                drafts.append(tokens)
            return torch.stack(drafts, 1)

        inputs = dict(
            tokens=tokens,
            hidden=hidden,
            starts=starts,
            anchors=anchors,
            draft_positions=draft_positions,
        )
        if self.graph_draft_chain and not torch.cuda.is_current_stream_capturing():
            # The graph also bakes in the page tables' and caches' addresses (stable per KV manager / metadata).
            key = (
                batch_size,
                draft_len,
                *(table.data_ptr() for table in tables.values()),
                *(cache.data_ptr() for cache in kv.caches),
            )
            next_draft_tokens = self._draft_graphs(chain, inputs, key)
        else:
            next_draft_tokens = chain(**inputs)
        last_accepted = accepted_tokens[batch_rows, num_accepted.long() - 1]
        next_new_tokens = torch.cat(
            [last_accepted.unsqueeze(1).to(torch.int32), next_draft_tokens], 1
        )
        return {
            "logits": logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    @staticmethod
    def _sample_target(logits: torch.Tensor, spec_metadata: Any) -> torch.Tensor:
        """The target token of every logits row: the argmax, or a sample from that row's sampling parameters."""
        if spec_metadata.is_all_greedy_sample:
            return torch.argmax(logits, dim=-1)
        rows = logits.shape[0]
        temperatures = spec_metadata.temperatures[:rows]
        top_ks = spec_metadata.top_ks[:rows]
        top_ps = spec_metadata.top_ps[:rows]
        min_ps = spec_metadata.min_ps[:rows]
        seed, offset = spec_metadata.seeds[:rows], spec_metadata.offsets[:rows]
        mode = spec_metadata.advanced_sampling_mode
        if mode.is_fused:
            return fused_sampling.fused_sample_from_logits(
                logits, temperatures, top_ks, top_ps, min_ps, seed=seed, offset=offset
            )
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(mode, top_ks, top_ps)
        return flashinfer_sampling.sample_from_logits_op(
            logits, temperatures, top_ks, top_ps, seed=seed, offset=offset
        )


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


def _kv_cache_layer_specs(config: Any) -> list[KVCacheLayerSpec]:
    """Per-layer KV geometry: one KV head; head_dim 512 on full layers, 256 and a 512-token window on the others."""
    return [
        KVCacheLayerSpec(
            head_dim=head_dim_of(i),
            num_kv_heads=1,
            num_q_heads=NUM_HEADS,
            attention_window=None if i % 5 == 4 else int(config.sliding_window),
        )
        for i in range(NUM_LAYERS)
    ]


def _check_construction(model_config: ModelConfig) -> None:
    """What routing checks and a direct construction would not: the GPU, the checkpoint, the speculation."""
    capability = torch.cuda.get_device_capability()
    if capability != _SM:
        raise ValueError(
            f"this target is written for sm120; the device is sm{capability[0]}{capability[1]}"
        )
    quant_config = model_config.quant_config
    if quant_config is None or quant_config.quant_algo != QuantAlgo.FP8:
        raise ValueError("this target loads the ModelOpt FP8 Gemma-4 E2B-it checkpoint")
    refused = speculation_refused(model_config.spec_config)
    if refused:
        raise ValueError(
            "the Gemma-4 E2B modeling_v2 target runs no speculation or one-model MTP with the Gemma-4 assistant; "
            f"it does not run {', '.join(refused)}"
        )
    if model_config.mapping.world_size != 1:
        raise ValueError("this target runs on one GPU")


@disable_v2_by_default
class ModelingV2Gemma4E2bItFp8Sm120Tp1(DecoderModelForCausalLM):
    """The class ``route`` returns: the Gemma-4 E2B-it FP8 decoder, its FP8 LM head and, with one-model MTP, the
    Gemma-4 assistant run inside the forward."""

    # The engine reads this to treat ``speculative_model`` as an external draft checkpoint (the assistant), not as
    # MTP heads replacing the target's.
    build_mtp_draft_model_from_config = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # The target reads FlashInfer metadata's per-pool page tables (it never calls a FlashInfer wrapper).
        return {"attn_backend": "FLASHINFER"}

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        return "V2"

    def __init__(self, model_config: ModelConfig) -> None:
        _check_construction(model_config)
        model_config.set_kv_cache_layer_specs(_kv_cache_layer_specs(model_config.pretrained_config))
        super().__init__(
            Gemma4E2bModel(model_config),
            config=model_config,
            hidden_size=HIDDEN,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        spec_config = model_config.spec_config
        self.spec_config = spec_config
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        self.draft_config = None
        self.draft_model = None
        self.mtp_step = None
        if spec_config is not None:
            self.draft_config = ModelConfig.from_pretrained(
                spec_config.speculative_model,
                trust_remote_code=True,
                mapping=model_config.mapping,
                spec_config=None,
                max_num_tokens=model_config.max_num_tokens,
            )
            self.draft_config.extra_attrs = model_config.extra_attrs
            self.draft_model = Gemma4AssistantDraft(self.draft_config)
            self.epilogue.append(self.draft_model)
            self.mtp_step = Gemma4MtpStep()

    def set_guided_decoder(self, guided_decoder: Any) -> bool:
        """Guided decoding is not supported: the speculative step runs no guided decoder."""
        del guided_decoder
        return False

    def load_weights(self, weights: Any, weight_mapper: Any = None, **kwargs: Any) -> None:
        _weights.load_target(self, weights)

    def load_draft_weights(self, weights: Any, weight_mapper: Any = None, **kwargs: Any) -> None:
        self.draft_model.load_weights(weights)

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load_weights()

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """FP8 lm_head (fp32 out) and Gemma-4's final logit softcap 30."""
        lm_head = self.lm_head
        logits = _fp8_linear(
            hidden, lm_head.weight, lm_head.weight_scale, lm_head.input_scale, torch.float32
        )
        return torch.tanh(logits * (1.0 / _SOFTCAP)) * _SOFTCAP

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: Any = None,
        resource_manager: Any = None,
        **kwargs: Any,
    ):
        del resource_manager
        if input_ids is None:
            input_ids = kwargs.get("orig_input_ids")
        hidden = self.model(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            # With MTP the engine always asks for context logits (it has gather_ids); the fused step reads only the
            # gather_ids rows and each request's draft row, which the prefill keeps.
            all_rows=return_context_logits and (self.mtp_step is None or spec_metadata is None),
        )
        if attn_metadata.padded_num_tokens is not None:
            hidden = hidden[: attn_metadata.num_tokens]
            input_ids = input_ids[: attn_metadata.num_tokens]
            position_ids = position_ids[..., : attn_metadata.num_tokens]
        if self.mtp_step is None or spec_metadata is None:
            if not return_context_logits:
                last = torch.cumsum(attn_metadata.seq_lens_cuda, 0, dtype=torch.long) - 1
                hidden = hidden[last]
            return self._logits(hidden)
        logits = self._logits(hidden[spec_metadata.gather_ids])
        return self.mtp_step(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            target=self.model,
            draft=self.draft_model,
        )
