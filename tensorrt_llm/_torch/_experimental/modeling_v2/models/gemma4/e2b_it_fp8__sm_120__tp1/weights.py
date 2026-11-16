# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint mapping of the Gemma-4 E2B-it FP8 target and of the Gemma-4 assistant drafter.

Target (``Gemma4ForCausalLM``, text-only ModelOpt FP8 checkpoint, keys ``model.layers.<i>.*``): the weights are
held stacked per layer kind, in exactly the layout the decoder-stack megakernel reads (``ops/decode_stack.py``,
``WEIGHT_NAMES``), and the layer-by-layer path reads per-layer views of the same tensors, so nothing is stored
twice.

* Fused projections. ``q/k/v_proj`` (layers 0-14) and ``gate/up_proj`` (all layers) are concatenated row-wise.
  Each FP8 shard keeps its own ModelOpt scales in the checkpoint; the fused weight gets the largest weight scale
  and each shard is requantized ``e4m3(bf16(w) * s_shard / s_max)``, and the activation scale is the largest
  input scale -- the semantics of the built-in FP8 linear (``FP8QDQLinearMethod.rescale_fused_weights``), so
  both implementations multiply the same FP8 codes.
* Layers 15-34 have a BF16 ``q_proj`` (excluded from quantization) and no k/v: they read layers 13 / 14's KV.
* ``*_scales[j]`` rows are ``(weight_scale, input_scale)`` of qkv, o, gate_up, down (A/B kinds) or o, gate_up,
  down (C/D kinds). ``layer_norm_w[i]`` stacks input, post-attention, pre-feedforward, post-feedforward and
  post-per-layer-input norms. ``layer_scalar`` is fp32.
* The FP8 lm_head (weight, weight_scale, input_scale) is loaded into the engine's ``LMHead``.

Assistant (``Gemma4AssistantForCausalLM``): 4 Q-only layers of hidden size 256 (3 sliding with 4 x 256 heads,
1 full with 4 x 512), BF16, tied output embedding, the centroid-masked head (``masked_embedding.*``) and the
pre/post projections to and from the target's hidden size.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .ops.decode_stack import (
    HEAD_DIM_FULL,
    HEAD_DIM_SLIDING,
    HIDDEN,
    LAYERS_A,
    LAYERS_B,
    LAYERS_C,
    LAYERS_D,
    NUM_HEADS,
    NUM_LAYERS,
    PLE_DIM,
    WEIGHT_NAMES,
)

F8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
VOCAB = 262144
INTER = 6144

_NORM_NAMES = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "post_per_layer_input_norm",
)


def _shapes() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    s, f = HEAD_DIM_SLIDING, HEAD_DIM_FULL
    shapes = {
        "embed_tokens": ((VOCAB, HIDDEN), BF16),
        "embed_tokens_per_layer": ((VOCAB, NUM_LAYERS * PLE_DIM), BF16),
        "ple_model_proj_w": ((NUM_LAYERS * PLE_DIM, HIDDEN), BF16),
        "ple_proj_norm_w": ((PLE_DIM,), BF16),
        "layer_norm_w": ((NUM_LAYERS, 5, HIDDEN), BF16),
        "layer_scalar": ((NUM_LAYERS,), torch.float32),
        "ple_gate_w": ((NUM_LAYERS, PLE_DIM, HIDDEN), BF16),
        "ple_proj_w": ((NUM_LAYERS, HIDDEN, PLE_DIM), BF16),
        "final_norm_w": ((HIDDEN,), BF16),
    }
    for kind, layers, hd, double in (("a", LAYERS_A, s, False), ("b", LAYERS_B, f, False),
                                     ("c", LAYERS_C, s, True), ("d", LAYERS_D, f, True)):  # fmt: skip
        n = len(layers)
        inter = INTER * (2 if double else 1)
        if kind in "ab":
            shapes[f"{kind}_qkv_w"] = ((n, NUM_HEADS * hd + 2 * hd, HIDDEN), F8)
            shapes[f"{kind}_qk_norm_w"] = ((n, 2, hd), BF16)
        else:
            shapes[f"{kind}_q_w"] = ((n, NUM_HEADS * hd, HIDDEN), BF16)
            shapes[f"{kind}_q_norm_w"] = ((n, hd), BF16)
        shapes[f"{kind}_o_w"] = ((n, HIDDEN, NUM_HEADS * hd), F8)
        shapes[f"{kind}_gu_w"] = ((n, 2 * inter, HIDDEN), F8)
        shapes[f"{kind}_down_w"] = ((n, HIDDEN, inter), F8)
        shapes[f"{kind}_scales"] = ((n, 4 if kind in "ab" else 3, 2), torch.float32)
    assert set(WEIGHT_NAMES) <= set(shapes)
    return shapes


def declare_target() -> nn.ParameterDict:
    """Every target weight, empty (meta-init intercepts the allocation; the engine materializes it)."""
    return nn.ParameterDict(
        {
            name: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            for name, (shape, dtype) in _shapes().items()
        }
    )


class _Checkpoint:
    """Name lookup over the engine's weight mapping, tolerant of the multimodal ``model.language_model.`` prefix."""

    def __init__(self, weights: Mapping[str, Any]):
        self.weights = weights
        self.used: set[str] = set()

    def __call__(self, name: str) -> torch.Tensor:
        for key in (name, name.replace("model.", "model.language_model.", 1)):
            if key in self.weights:
                self.used.add(key)
                return self.weights[key]
        raise KeyError(f"checkpoint tensor {name} is missing")


def _scalar(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(()).to(device="cuda", dtype=torch.float32)


def _fuse_fp8(
    ckpt: _Checkpoint, prefixes: list[str], out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate FP8 shards into ``out`` with one (max) weight scale; returns (weight_scale, input_scale)."""
    w_scales = [_scalar(ckpt(p + ".weight_scale")) for p in prefixes]
    in_scales = [_scalar(ckpt(p + ".input_scale")) for p in prefixes]
    w_max = torch.stack(w_scales).max()
    row = 0
    for p, s in zip(prefixes, w_scales):
        shard = ckpt(p + ".weight").to("cuda")
        assert shard.dtype == F8, (p, shard.dtype)
        if len(prefixes) > 1:
            shard = shard.to(BF16).mul_(s).div_(w_max).to(F8)
        out[row : row + shard.shape[0]].copy_(shard)
        row += shard.shape[0]
    assert row == out.shape[0], (prefixes, row, out.shape)
    return w_max, torch.stack(in_scales).max()


@torch.no_grad()
def load_target(model: nn.Module, weights: Mapping[str, Any]) -> None:
    """Fill ``model.model.w`` and ``model.lm_head`` from the HF checkpoint."""
    ckpt = _Checkpoint(weights)
    w = model.model.w

    def copy(name: str, src: torch.Tensor) -> None:
        dst = w[name]
        assert tuple(src.shape) == tuple(dst.shape), (name, tuple(src.shape), tuple(dst.shape))
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    copy("embed_tokens", ckpt("model.embed_tokens.weight"))
    copy("embed_tokens_per_layer", ckpt("model.embed_tokens_per_layer.weight"))
    copy("ple_model_proj_w", ckpt("model.per_layer_model_projection.weight"))
    copy("ple_proj_norm_w", ckpt("model.per_layer_projection_norm.weight"))
    copy("final_norm_w", ckpt("model.norm.weight"))
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for k, norm in enumerate(_NORM_NAMES):
            w["layer_norm_w"][i, k].copy_(ckpt(f"{p}.{norm}.weight").to("cuda", BF16))
        w["layer_scalar"][i].copy_(_scalar(ckpt(f"{p}.layer_scalar")))
        w["ple_gate_w"][i].copy_(ckpt(f"{p}.per_layer_input_gate.weight").to("cuda", BF16))
        w["ple_proj_w"][i].copy_(ckpt(f"{p}.per_layer_projection.weight").to("cuda", BF16))

    for kind, layers in (("a", LAYERS_A), ("b", LAYERS_B), ("c", LAYERS_C), ("d", LAYERS_D)):
        scales = w[f"{kind}_scales"]
        for j, i in enumerate(layers):
            p = f"model.layers.{i}"
            rows = []
            if kind in "ab":
                rows.append(
                    _fuse_fp8(
                        ckpt, [f"{p}.self_attn.{n}_proj" for n in "qkv"], w[f"{kind}_qkv_w"][j]
                    )
                )
                w[f"{kind}_qk_norm_w"][j, 0].copy_(
                    ckpt(f"{p}.self_attn.q_norm.weight").to("cuda", BF16)
                )
                w[f"{kind}_qk_norm_w"][j, 1].copy_(
                    ckpt(f"{p}.self_attn.k_norm.weight").to("cuda", BF16)
                )
            else:
                q = ckpt(f"{p}.self_attn.q_proj.weight")
                assert q.dtype == BF16, (p, q.dtype)
                w[f"{kind}_q_w"][j].copy_(q.to("cuda"))
                w[f"{kind}_q_norm_w"][j].copy_(
                    ckpt(f"{p}.self_attn.q_norm.weight").to("cuda", BF16)
                )
            rows.append(_fuse_fp8(ckpt, [f"{p}.self_attn.o_proj"], w[f"{kind}_o_w"][j]))
            rows.append(
                _fuse_fp8(ckpt, [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"], w[f"{kind}_gu_w"][j])
            )
            rows.append(_fuse_fp8(ckpt, [f"{p}.mlp.down_proj"], w[f"{kind}_down_w"][j]))
            for r, (w_scale, in_scale) in enumerate(rows):
                scales[j, r, 0] = w_scale
                scales[j, r, 1] = in_scale

    lm_head = model.lm_head
    assert lm_head.weight.dtype == F8, (
        f"the engine built a {lm_head.weight.dtype} LM head; this target runs the checkpoint's FP8 lm_head"
    )
    lm_head.weight.copy_(ckpt("lm_head.weight").to("cuda"))
    lm_head.weight_scale.copy_(
        _scalar(ckpt("lm_head.weight_scale")).reshape(lm_head.weight_scale.shape)
    )
    lm_head.input_scale.copy_(
        _scalar(ckpt("lm_head.input_scale")).reshape(lm_head.input_scale.shape)
    )

    unused = sorted(k for k in weights.keys() if k not in ckpt.used)
    assert not unused, f"checkpoint tensors this target does not load: {unused[:8]}"


def declare_assistant(config: Any) -> nn.ParameterDict:
    """The assistant's weights, from its ``text_config`` (hidden 256, 4 layers) and backbone hidden size."""
    tc = config.text_config
    hidden, inter = tc.hidden_size, tc.intermediate_size
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "embed_tokens": ((tc.vocab_size, hidden), BF16),
        "centroids": ((config.num_centroids, hidden), BF16),
        "pre_projection": ((hidden, 2 * config.backbone_hidden_size), BF16),
        "post_projection": ((config.backbone_hidden_size, hidden), BF16),
        "norm": ((hidden,), BF16),
    }
    for i, layer_type in enumerate(tc.layer_types):
        hd = tc.head_dim if layer_type == "sliding_attention" else tc.global_head_dim
        heads = tc.num_attention_heads
        shapes.update(
            {
                f"l{i}_q_proj": ((heads * hd, hidden), BF16),
                f"l{i}_q_norm": ((hd,), BF16),
                f"l{i}_o_proj": ((hidden, heads * hd), BF16),
                f"l{i}_gate_up": ((2 * inter, hidden), BF16),
                f"l{i}_down": ((hidden, inter), BF16),
                f"l{i}_norms": ((4, hidden), BF16),
                f"l{i}_layer_scalar": ((1,), torch.float32),
            }
        )
    return nn.ParameterDict(
        {
            name: nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            for name, (shape, dtype) in shapes.items()
        }
    )


@torch.no_grad()
def load_assistant(draft: nn.Module, weights: Mapping[str, Any]) -> None:
    """Fill the drafter's weights and its token-ordering buffer from the assistant checkpoint."""
    ckpt = _Checkpoint(weights)
    w = draft.w

    def copy(name: str, src: torch.Tensor) -> None:
        dst = w[name]
        assert tuple(src.shape) == tuple(dst.shape), (name, tuple(src.shape), tuple(dst.shape))
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    copy("embed_tokens", ckpt("model.embed_tokens.weight"))
    copy("centroids", ckpt("masked_embedding.centroids.weight"))
    copy("pre_projection", ckpt("pre_projection.weight"))
    copy("post_projection", ckpt("post_projection.weight"))
    copy("norm", ckpt("model.norm.weight"))
    draft.token_ordering.copy_(ckpt("masked_embedding.token_ordering").to("cuda", torch.long))
    for i in range(draft.num_layers):
        p = f"model.layers.{i}"
        copy(f"l{i}_q_proj", ckpt(f"{p}.self_attn.q_proj.weight"))
        copy(f"l{i}_q_norm", ckpt(f"{p}.self_attn.q_norm.weight"))
        copy(f"l{i}_o_proj", ckpt(f"{p}.self_attn.o_proj.weight"))
        copy(
            f"l{i}_gate_up",
            torch.cat([ckpt(f"{p}.mlp.gate_proj.weight"), ckpt(f"{p}.mlp.up_proj.weight")], 0),
        )
        copy(f"l{i}_down", ckpt(f"{p}.mlp.down_proj.weight"))
        copy(f"l{i}_norms", torch.stack([ckpt(f"{p}.{n}.weight") for n in _NORM_NAMES[:4]]))
        copy(f"l{i}_layer_scalar", ckpt(f"{p}.layer_scalar").reshape(1).float())
    unused = sorted(k for k in weights.keys() if k not in ckpt.used)
    assert not unused, f"assistant tensors the drafter does not load: {unused[:8]}"
