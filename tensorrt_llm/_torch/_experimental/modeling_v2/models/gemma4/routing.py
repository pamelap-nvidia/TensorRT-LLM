# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Gemma4ForCausalLM config lands. Read this file and you know.

One target, ``e2b_it_fp8__sm_120__tp1``: the text-only Gemma-4 E2B-it checkpoint quantized with ModelOpt FP8
(per-tensor static scales, FP8 lm_head) on one SM120 GPU (RTX PRO 6000 Blackwell Server Edition), FP8 KV cache.
It serves no speculative decoding and one-model MTP with the Gemma-4 assistant checkpoint, which it runs inside
its own forward. Every other deployment raises ``NoTarget``: under ``auto`` the engine then builds the built-in
Gemma-4 model, under ``require`` that is the error.

The multimodal ``Gemma4ForConditionalGeneration`` is not routed here: the target loads no vision or audio tower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture the target is written for: the megakernel uses SM120 FP8 mma.sync and bulk copies.
_SM = (12, 0)

# Gemma-4 E2B's text-config shape. Shape-sniffing pins the geometry the decoder-stack megakernel is compiled for;
# it does not identify the checkpoint's training run.
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "hidden_size_per_layer_input",
    "num_kv_shared_layers",
    "intermediate_size",
    "head_dim",
    "global_head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "vocab_size",
    "vocab_size_per_layer_input",
)
_SHAPE = (35, 1536, 256, 20, 6144, 256, 512, 8, 1, 512, 262144, 262144)
# Every fifth layer is full attention, the rest sliding-window.
_LAYER_TYPES = tuple(
    "full_attention" if i % 5 == 4 else "sliding_attention" for i in range(_SHAPE[0])
)

# The megakernel's envelope: at most 8 requests of at most 8 tokens per decode step.
_MAX_BATCH = 8
_MAX_TOKENS_PER_REQUEST = 8

_ASSISTANT_ARCHITECTURE = "Gemma4AssistantForCausalLM"


def _field(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _checkpoint_refused(model_config: ModelConfig) -> list[str]:
    """What about the checkpoint the target does not load; empty when it is the E2B-it FP8 text checkpoint."""
    config = model_config.pretrained_config
    refused = []
    shape = tuple(_field(config, name) for name in _SHAPE_FIELDS)
    if shape != _SHAPE:
        refused.append(f"text config shape {dict(zip(_SHAPE_FIELDS, shape))} (not Gemma-4 E2B)")
    if tuple(_field(config, "layer_types") or ()) != _LAYER_TYPES:
        refused.append("a layer-type pattern other than E2B's (every fifth layer full attention)")
    flags = {
        "MoE blocks": bool(_field(config, "enable_moe_block")),
        "attention_k_eq_v": bool(_field(config, "attention_k_eq_v")),
        "no double-wide MLP on the KV-shared layers": not _field(config, "use_double_wide_mlp"),
        "tied word embeddings": bool(_field(config, "tie_word_embeddings")),
        "bidirectional attention": _field(config, "use_bidirectional_attention") is not None,
        "a final logit softcap other than 30": _field(config, "final_logit_softcapping") != 30.0,
    }
    refused.extend(name for name, present in flags.items() if present)
    quant_config = getattr(model_config, "quant_config", None)
    quant_algo = None if quant_config is None else quant_config.quant_algo
    if quant_algo != QuantAlgo.FP8:
        refused.append(
            f"weight quantization {quant_algo.name if quant_algo else 'none'} (only ModelOpt FP8)"
        )
    if getattr(model_config, "quant_config_dict", None) is not None:
        refused.append("a mixed-precision quantization config")
    elif quant_algo == QuantAlgo.FP8 and not _checkpoint_has_fp8_lm_head(model_config):
        refused.append(
            "a BF16 lm_head (the target runs the FP8 lm_head of the text-only checkpoint)"
        )
    return refused


def _checkpoint_has_fp8_lm_head(model_config: ModelConfig) -> bool:
    """Whether the checkpoint stores ``lm_head.weight_scale``, the rule the engine builds an FP8 LM head by."""
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

    return DecoderModelForCausalLM._checkpoint_has_lm_head_scale(model_config)


def speculation_refused(spec_config: Any) -> list[str]:
    """The speculative features the target does not compute; empty for none or the fused Gemma-4 MTP chain.

    The target runs the Gemma-4 assistant as a linear one-model MTP-Eagle chain with strict acceptance inside its
    forward. Its constructor checks this again against direct construction.
    """
    if spec_config is None:
        return []
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP with the Gemma-4 assistant)"]
    refused = {
        "vanilla MTP (only MTP-Eagle one-model)": not spec_config.spec_dec_mode.is_mtp_eagle_one_model(),
        "no assistant checkpoint (speculative_model)": spec_config.speculative_model is None,
        "a dynamic draft tree": spec_config.use_dynamic_tree,
        "draft trees": spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len),
        "rejection sampling": spec_config.use_rejection_sampling,
        "relaxed acceptance": spec_config.use_relaxed_acceptance_for_thinking,
        "occurrence penalties": spec_config.enable_penalty,
        "sa_config": spec_config.sa_config is not None,
        "a draft-length schedule": spec_config.draft_len_schedule is not None,
        f"max_draft_len {spec_config.max_draft_len} (at most {_MAX_TOKENS_PER_REQUEST - 1})": (
            spec_config.max_draft_len is not None
            and spec_config.max_draft_len + 1 > _MAX_TOKENS_PER_REQUEST
        ),
    }
    refused = [name for name, present in refused.items() if present]
    if spec_config.speculative_model is not None and not refused:
        architecture = _assistant_architecture(spec_config.speculative_model)
        if architecture != _ASSISTANT_ARCHITECTURE:
            refused.append(f"a draft checkpoint of architecture {architecture}")
    return refused


def _assistant_architecture(path: Any) -> str | None:
    """``architectures[0]`` of the draft checkpoint's config.json, or None when it cannot be read."""
    import json
    import os

    config_path = os.path.join(str(path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            architectures = json.load(f).get("architectures") or [None]
    except (OSError, ValueError):
        return None
    return architectures[0]


def _engine_refused(model_config: ModelConfig, llm_args: TorchLlmArgs) -> list[str]:
    """The engine features the target does not run; empty when it runs this deployment."""
    mapping = model_config.mapping
    refused = {
        f"tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} (only one GPU)": (
            mapping.world_size != 1 or mapping.tp_size != 1
        ),
        "attention DP": mapping.enable_attention_dp,
        f"max_batch_size {llm_args.max_batch_size} (at most {_MAX_BATCH})": (
            llm_args.max_batch_size is None or llm_args.max_batch_size > _MAX_BATCH
        ),
        "guided decoding": llm_args.guided_decoding_backend is not None,
        f"attention backend {llm_args.attn_backend} (the target reads FlashInfer metadata)": (
            llm_args.attn_backend != "FLASHINFER"
        ),
        "KV cache block reuse": bool(llm_args.kv_cache_config.enable_block_reuse),
        "chunked prefill": bool(llm_args.enable_chunked_prefill),
        "LoRA": getattr(llm_args, "lora_config", None) is not None,
    }
    refused = [name for name, present in refused.items() if present]
    kv_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_algo != QuantAlgo.FP8:
        refused.append(f"KV cache {getattr(kv_algo, 'name', kv_algo) or 'bf16'} (only FP8)")
    return refused


def refused(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> list[str]:
    """Every reason the target does not take this deployment; empty when it does."""
    reasons = []
    if tuple(compute_capability) != _SM:
        reasons.append(f"compute capability {tuple(compute_capability)} (only sm120)")
    reasons += _checkpoint_refused(model_config)
    reasons += _engine_refused(model_config, llm_args)
    reasons += speculation_refused(model_config.spec_config)
    return reasons


@routes("Gemma4ForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import of modeling code.
    """
    reasons = refused(model_config, llm_args, compute_capability)
    if reasons:
        raise NoTarget(
            "the gemma4 target runs the ModelOpt FP8 Gemma-4 E2B-it text checkpoint on one sm120 GPU with an FP8 KV "
            "cache, without speculation or with one-model MTP from the Gemma-4 assistant; this deployment has "
            + "; ".join(reasons)
        )
    from .e2b_it_fp8__sm_120__tp1.modeling import ModelingV2Gemma4E2bItFp8Sm120Tp1

    return ModelingV2Gemma4E2bItFp8Sm120Tp1
