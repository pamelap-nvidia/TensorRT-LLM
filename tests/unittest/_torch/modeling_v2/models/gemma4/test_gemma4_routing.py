# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 E2B target routing, driven by synthetic configs on CPU.

``refused`` is the whole decision; the accepted cases check that it is empty, and the declined cases that both
``refused`` names the reason and ``modeling_v2_resolve`` falls back under ``auto`` / raises under ``require``.
"""

import json

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.gemma4 import routing
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM120 = (12, 0)


@pytest.fixture(autouse=True)
def _on_sm120(monkeypatch):
    """Route as if this were an RTX PRO 6000 with the FP8 lm_head checkpoint, wherever the test runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM120)
    monkeypatch.setattr(routing, "_checkpoint_has_fp8_lm_head", lambda model_config: True)


@pytest.fixture
def assistant(tmp_path):
    """A draft checkpoint directory declaring the Gemma-4 assistant architecture."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4AssistantForCausalLM"]})
    )
    return str(tmp_path)


def _e2b_config(**overrides) -> PretrainedConfig:
    fields = dict(
        architectures=["Gemma4ForCausalLM"],
        model_type="gemma4_text",
        num_hidden_layers=35,
        hidden_size=1536,
        hidden_size_per_layer_input=256,
        num_kv_shared_layers=20,
        intermediate_size=6144,
        head_dim=256,
        global_head_dim=512,
        num_attention_heads=8,
        num_key_value_heads=1,
        sliding_window=512,
        vocab_size=262144,
        vocab_size_per_layer_input=262144,
        layer_types=["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(35)],
        enable_moe_block=False,
        attention_k_eq_v=False,
        use_double_wide_mlp=True,
        tie_word_embeddings=False,
        use_bidirectional_attention=None,
        final_logit_softcapping=30.0,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _model_config(
    pretrained_config=None, quant_algo=QuantAlgo.FP8, spec_config=None, **mapping_kwargs
):
    return ModelConfig(
        pretrained_config=pretrained_config or _e2b_config(),
        mapping=Mapping(**(mapping_kwargs or dict(world_size=1, tp_size=1))),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo),
    )


def _llm_args(mode="require", **kwargs) -> TorchLlmArgs:
    args = dict(
        model="/tmp/dummy_model",
        modeling_v2=mode,
        max_batch_size=8,
        attn_backend="FLASHINFER",
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=False),
    )
    args.update(kwargs)
    return TorchLlmArgs(**args)


@pytest.mark.parametrize(
    "draft_len", [None, 1, 4, 5, 7], ids=lambda k: "no-spec" if k is None else f"mtp{k}"
)
def test_e2b_fp8_on_sm120_is_admitted(draft_len, assistant):
    spec = (
        None
        if draft_len is None
        else MTPDecodingConfig(max_draft_len=draft_len, speculative_model=assistant)
    )
    config = _model_config(spec_config=spec)
    assert routing.refused(config, _llm_args(), _SM120) == []


def _declined(config, llm_args, capability=_SM120) -> str:
    reasons = routing.refused(config, llm_args, capability)
    assert reasons, "expected the deployment to be declined"
    return "; ".join(reasons)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(num_hidden_layers=42, layer_types=["sliding_attention"] * 42), "not Gemma-4 E2B"),
        (dict(hidden_size=2560), "not Gemma-4 E2B"),
        (dict(enable_moe_block=True), "MoE"),
        (dict(attention_k_eq_v=True), "attention_k_eq_v"),
        (dict(use_double_wide_mlp=False), "double-wide"),
        (dict(final_logit_softcapping=None), "softcap"),
    ],
)
def test_other_checkpoints_are_declined(overrides, reason):
    assert reason in _declined(_model_config(_e2b_config(**overrides)), _llm_args())


@pytest.mark.parametrize("algo", [None, QuantAlgo.NVFP4, QuantAlgo.FP8_BLOCK_SCALES])
def test_other_quantizations_are_declined(algo):
    assert "only ModelOpt FP8" in _declined(_model_config(quant_algo=algo), _llm_args())


def test_bf16_lm_head_is_declined(monkeypatch):
    monkeypatch.setattr(routing, "_checkpoint_has_fp8_lm_head", lambda model_config: False)
    assert "BF16 lm_head" in _declined(_model_config(), _llm_args())


@pytest.mark.parametrize("capability", [(10, 0), (9, 0), (12, 1)])
def test_other_gpus_are_declined(capability):
    assert "only sm120" in _declined(_model_config(), _llm_args(), capability)


@pytest.mark.parametrize(
    "llm_kwargs, reason",
    [
        (dict(max_batch_size=16), "max_batch_size"),
        (dict(attn_backend="TRTLLM"), "FlashInfer metadata"),
        (dict(enable_chunked_prefill=True), "chunked prefill"),
        (dict(kv_cache_config=KvCacheConfig(dtype="fp8", enable_block_reuse=True)), "block reuse"),
        (dict(kv_cache_config=KvCacheConfig(dtype="auto", enable_block_reuse=False)), "only FP8"),
        (dict(guided_decoding_backend="xgrammar"), "guided decoding"),
    ],
)
def test_unsupported_engine_features_are_declined(llm_kwargs, reason):
    assert reason in _declined(_model_config(), _llm_args(**llm_kwargs))


def test_tensor_parallelism_is_declined():
    assert "only one GPU" in _declined(_model_config(world_size=2, tp_size=2), _llm_args())


@pytest.mark.parametrize(
    "spec_kwargs, reason",
    [
        (dict(max_draft_len=8), "at most 7"),
        (dict(max_draft_len=3, use_rejection_sampling=True), "rejection sampling"),
        (dict(max_draft_len=3, use_relaxed_acceptance_for_thinking=True), "relaxed acceptance"),
        (dict(max_draft_len=1, use_mtp_vanilla=True), "vanilla MTP"),
    ],
)
def test_unsupported_mtp_is_declined(spec_kwargs, reason, assistant):
    spec = MTPDecodingConfig(speculative_model=assistant, **spec_kwargs)
    assert reason in _declined(_model_config(spec_config=spec), _llm_args())


def test_mtp_needs_the_assistant_checkpoint(tmp_path):
    assert "no assistant checkpoint" in _declined(
        _model_config(spec_config=MTPDecodingConfig(max_draft_len=3)), _llm_args()
    )
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
    spec = MTPDecodingConfig(max_draft_len=3, speculative_model=str(tmp_path))
    assert "a draft checkpoint of architecture Gemma4ForCausalLM" in _declined(
        _model_config(spec_config=spec), _llm_args()
    )


def test_other_speculation_is_declined(assistant):
    spec = Eagle3DecodingConfig(max_draft_len=3, speculative_model=assistant)
    assert "only MTP" in _declined(_model_config(spec_config=spec), _llm_args())


def test_declined_deployment_falls_back_under_auto_and_raises_under_require():
    config = _model_config(quant_algo=QuantAlgo.NVFP4)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="only ModelOpt FP8"):
        modeling_v2_resolve(config, _llm_args("require"))
