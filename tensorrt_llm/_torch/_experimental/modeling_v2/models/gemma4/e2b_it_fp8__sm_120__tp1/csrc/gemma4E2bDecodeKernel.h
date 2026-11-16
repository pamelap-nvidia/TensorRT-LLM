/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace gemma4_e2b_decode
{
typedef __nv_bfloat16 bf16;

//! Kernel arguments. Pointers to device memory except where noted; R requests x Q tokens (T = R * Q <= 64).
//! kvk/kvv/kvps: for each KV-producing layer 0..14, the address of page 0's K / V rows and the page stride in bytes.
//! pt_s / pt_f: flat page tables of the sliding / full KV pool; pt_indptr[r]: offset of request r's pages in them.
//! cache_lens[r]: tokens already cached for request r (device; the new tokens go to cache_lens[r] .. + Q - 1).
struct Params
{
    bf16 const* hidden;
    bf16 const* ple_tok;
    int const* cache_lens;
    uint8_t* kvk[15];
    uint8_t* kvv[15];
    long long kvps[15]; // paged KV: page-0 K/V address, page stride (bytes)
    int const* pt_s;
    int const* pt_f;
    int const* pt_indptr; // page tables (sliding / full pool), per-request offsets
    bf16 const* ple_mp_w;
    bf16 const* ple_pn_w;
    bf16 const* ln_w;
    float const* lscalar;
    bf16 const* ple_gate_w;
    bf16 const* ple_proj_w;
    bf16 const* final_w;
    uint8_t const* a_qkv;
    uint8_t const* a_o;
    uint8_t const* a_gu;
    uint8_t const* a_down;
    bf16 const* a_qkn;
    float const* a_sc;
    uint8_t const* b_qkv;
    uint8_t const* b_o;
    uint8_t const* b_gu;
    uint8_t const* b_down;
    bf16 const* b_qkn;
    float const* b_sc;
    bf16 const* c_q;
    uint8_t const* c_o;
    uint8_t const* c_gu;
    uint8_t const* c_down;
    bf16 const* c_qn;
    float const* c_sc;
    bf16 const* d_q;
    uint8_t const* d_o;
    uint8_t const* d_gu;
    uint8_t const* d_down;
    bf16 const* d_qn;
    float const* d_sc;
    bf16* out;
    int R, Q, T, L;
    int ng, nic; // attention: head groups per request, key chunks per (request, group)
};
} // namespace gemma4_e2b_decode

//! Launch the decoder stack on `stream`. Fills prm.Q / L / ng / nic from prm.T and prm.R. Graph-capturable.
cudaError_t launchGemma4E2bDecodeStack(gemma4_e2b_decode::Params& prm, int numSms, cudaStream_t stream);

TRTLLM_MV2_NAMESPACE_END
