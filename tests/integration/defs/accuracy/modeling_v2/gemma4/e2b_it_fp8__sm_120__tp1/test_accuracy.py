# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole-model gates for the gemma4 modeling_v2 target e2b_it_fp8__sm_120__tp1.

The ModelOpt FP8 Gemma-4 E2B-it text checkpoint (FP8 lm_head) on one RTX PRO 6000, FP8 KV cache: GSM8K without
speculation and with the fused one-model MTP (the Gemma-4 assistant, five drafts, overlap scheduler, CUDA graphs),
plus the acceptance length. Every test passes ``modeling_v2="require"``, so a configuration that missed the
target's routing fails instead of testing the built-in model; the routing unit test
(``test_gemma4_routing.py``) pins that these configurations route to the target.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig, SamplingParams

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness, assert_acceptance_length_for_llm
from ..._env import scoped_env_overrides

# The text-only Gemma4ForCausalLM checkpoint of gemma-4-E2B-it quantized with ModelOpt 0.46 FP8 (per-tensor
# static weights and activations, lm_head included; PLE gates/projections and the KV-shared layers' q_proj BF16).
_MODEL_PATH = f"{llm_models_root()}/gemma/gemma-4-E2B-it-FP8-textonly"
_ASSISTANT_PATH = f"{llm_models_root()}/gemma/gemma-4-E2B-it-assistant"


@contextmanager
def _llm(speculative_config=None, **llm_kwargs) -> Iterator[LLM]:
    """The deployment the target is written for, with its step contract checked on the first forward."""
    with (
        scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
        LLM(
            _MODEL_PATH,
            modeling_v2="require",
            attn_backend="FLASHINFER",
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.5, enable_block_reuse=False, dtype="fp8"
            ),
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=5248,
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, batch_sizes=[1, 2, 4, 8]),
            speculative_config=speculative_config,
            env_overrides=rank_env,
            **llm_kwargs,
        ) as llm,
    ):
        yield llm


@pytest.mark.skipif(
    get_sm_version() != 120, reason="the target is written for SM120 (RTX PRO 6000)"
)
class TestModelingV2Gemma4E2bItFp8Sm120Tp1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-4-E2B-it"
    MODEL_PATH = _MODEL_PATH
    # The instruction-tuned checkpoint is evaluated in its chat format, zero-shot, until its end-of-turn token
    # (the harness default -- five raw few-shot exemplars -- scores ~8% on both Modeling V1 and this target).
    EXTRA_EVALUATOR_KWARGS = {
        "apply_chat_template": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "num_fewshot": 0,
    }
    SAMPLING_PARAMS = SamplingParams(max_tokens=1024, truncate_prompt_tokens=4096, stop=["<turn|>"])

    def _gsm8k(self, llm) -> None:
        GSM8K(self.MODEL_NAME).evaluate(
            llm, sampling_params=self.SAMPLING_PARAMS, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS
        )

    def test_gsm8k(self) -> None:
        with _llm() as llm:
            self._gsm8k(llm)

    def test_gsm8k_mtp(self) -> None:
        # Strict acceptance keeps the greedy tokens of the verify step, so GSM8K alone does not show that the
        # draft chain works; the acceptance length does.
        with _llm(
            MTPDecodingConfig(max_draft_len=5, speculative_model=_ASSISTANT_PATH),
            max_stats_len=-1,
            enable_iter_perf_stats=True,
        ) as llm:
            self._gsm8k(llm)
            assert_acceptance_length_for_llm(
                "TestModelingV2Gemma4E2bItFp8Sm120Tp1::test_gsm8k_mtp", llm
            )
