# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 E2B-it (ModelOpt FP8) / sm_120 / tp1, one-model MTP with the Gemma-4 assistant fused into the forward."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)  # torch.ops.mv2_gemma4_e2b_it_fp8_sm_120_tp1
