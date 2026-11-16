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

#include "gemma4/e2b_it_fp8__sm_120__tp1/csrc/gemma4E2bDecodeKernel.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

using gemma4_e2b_decode::bf16;

// Weight list order, element dtype and shape (see ops/decode_stack.py, which builds it).
struct WeightSpec
{
    char const* name;
    at::ScalarType dtype;
    std::array<int64_t, 3> shape; // trailing zeros = fewer dims
};

constexpr int kNumWeights = 31;
constexpr std::array<WeightSpec, kNumWeights> kWeights{{
    {"ple_model_proj_w", at::kBFloat16, {8960, 1536, 0}},
    {"ple_proj_norm_w", at::kBFloat16, {256, 0, 0}},
    {"layer_norm_w", at::kBFloat16, {35, 5, 1536}},
    {"layer_scalar", at::kFloat, {35, 0, 0}},
    {"ple_gate_w", at::kBFloat16, {35, 256, 1536}},
    {"ple_proj_w", at::kBFloat16, {35, 1536, 256}},
    {"final_norm_w", at::kBFloat16, {1536, 0, 0}},
    {"a_qkv_w", at::kFloat8_e4m3fn, {12, 2560, 1536}},
    {"a_o_w", at::kFloat8_e4m3fn, {12, 1536, 2048}},
    {"a_gu_w", at::kFloat8_e4m3fn, {12, 12288, 1536}},
    {"a_down_w", at::kFloat8_e4m3fn, {12, 1536, 6144}},
    {"a_qk_norm_w", at::kBFloat16, {12, 2, 256}},
    {"a_scales", at::kFloat, {12, 4, 2}},
    {"b_qkv_w", at::kFloat8_e4m3fn, {3, 5120, 1536}},
    {"b_o_w", at::kFloat8_e4m3fn, {3, 1536, 4096}},
    {"b_gu_w", at::kFloat8_e4m3fn, {3, 12288, 1536}},
    {"b_down_w", at::kFloat8_e4m3fn, {3, 1536, 6144}},
    {"b_qk_norm_w", at::kBFloat16, {3, 2, 512}},
    {"b_scales", at::kFloat, {3, 4, 2}},
    {"c_q_w", at::kBFloat16, {16, 2048, 1536}},
    {"c_o_w", at::kFloat8_e4m3fn, {16, 1536, 2048}},
    {"c_gu_w", at::kFloat8_e4m3fn, {16, 24576, 1536}},
    {"c_down_w", at::kFloat8_e4m3fn, {16, 1536, 12288}},
    {"c_q_norm_w", at::kBFloat16, {16, 256, 0}},
    {"c_scales", at::kFloat, {16, 3, 2}},
    {"d_q_w", at::kBFloat16, {4, 4096, 1536}},
    {"d_o_w", at::kFloat8_e4m3fn, {4, 1536, 4096}},
    {"d_gu_w", at::kFloat8_e4m3fn, {4, 24576, 1536}},
    {"d_down_w", at::kFloat8_e4m3fn, {4, 1536, 12288}},
    {"d_q_norm_w", at::kBFloat16, {4, 512, 0}},
    {"d_scales", at::kFloat, {4, 3, 2}},
}};

void checkWeight(at::Tensor const& w, WeightSpec const& spec, at::Device device)
{
    TORCH_CHECK(w.device() == device && w.is_contiguous(), "decode_stack: weight ", spec.name,
        " must be a contiguous tensor on ", device);
    TORCH_CHECK(w.scalar_type() == spec.dtype, "decode_stack: weight ", spec.name, " has dtype ", w.scalar_type(),
        ", expected ", spec.dtype);
    int64_t const dims = spec.shape[2] != 0 ? 3 : spec.shape[1] != 0 ? 2 : 1;
    TORCH_CHECK(w.dim() == dims, "decode_stack: weight ", spec.name, " has ", w.dim(), " dims, expected ", dims);
    for (int64_t d = 0; d < dims; ++d)
    {
        TORCH_CHECK(w.size(d) == spec.shape[d], "decode_stack: weight ", spec.name, " dim ", d, " is ", w.size(d),
            ", expected ", spec.shape[d]);
    }
}

void decodeStack(at::Tensor const& hiddenStates, at::Tensor const& pleTokenEmbeds, at::Tensor const& cacheLens,
    at::Tensor const& kvMeta, at::Tensor const& pageTableSliding, at::Tensor const& pageTableFull,
    at::Tensor const& pageTableIndptr, at::TensorList weights, at::Tensor& out)
{
    auto const device = hiddenStates.device();
    TORCH_CHECK(device.is_cuda(), "decode_stack: inputs must be CUDA tensors");
    int64_t const numTokens = hiddenStates.size(0);
    int64_t const numRequests = cacheLens.size(0);
    TORCH_CHECK(hiddenStates.dim() == 2 && hiddenStates.size(1) == 1536 && hiddenStates.scalar_type() == at::kBFloat16
            && hiddenStates.is_contiguous(),
        "decode_stack: hidden_states must be contiguous bf16 [T, 1536]");
    TORCH_CHECK(pleTokenEmbeds.dim() == 3 && pleTokenEmbeds.size(0) == numTokens && pleTokenEmbeds.size(1) == 35
            && pleTokenEmbeds.size(2) == 256 && pleTokenEmbeds.scalar_type() == at::kBFloat16
            && pleTokenEmbeds.is_contiguous(),
        "decode_stack: ple_token_embeds must be contiguous bf16 [T, 35, 256]");
    TORCH_CHECK(numRequests >= 1 && numTokens % numRequests == 0 && numTokens / numRequests <= 8 && numTokens <= 64,
        "decode_stack: needs T = R * Q with Q <= 8 and T <= 64, got T=", numTokens, " R=", numRequests);
    TORCH_CHECK(cacheLens.scalar_type() == at::kInt && cacheLens.device() == device && cacheLens.is_contiguous(),
        "decode_stack: cache_lens must be contiguous int32 on the device");
    TORCH_CHECK(kvMeta.device().is_cpu() && kvMeta.scalar_type() == at::kLong && kvMeta.dim() == 2
            && kvMeta.size(0) == 3 && kvMeta.size(1) == 15 && kvMeta.is_contiguous(),
        "decode_stack: kv_meta must be a contiguous CPU int64 [3, 15] tensor");
    for (auto const* t : {&pageTableSliding, &pageTableFull, &pageTableIndptr})
    {
        TORCH_CHECK(t->scalar_type() == at::kInt && t->device() == device && t->is_contiguous(),
            "decode_stack: page tables must be contiguous int32 on the device");
    }
    TORCH_CHECK(pageTableIndptr.numel() >= numRequests, "decode_stack: pt_indptr has fewer than R entries");
    TORCH_CHECK(out.sizes() == hiddenStates.sizes() && out.scalar_type() == at::kBFloat16 && out.is_contiguous()
            && out.device() == device,
        "decode_stack: out must be contiguous bf16 [T, 1536]");
    TORCH_CHECK(static_cast<int>(weights.size()) == kNumWeights, "decode_stack: expected ", kNumWeights,
        " weights, got ", weights.size());
    for (int i = 0; i < kNumWeights; ++i)
    {
        checkWeight(weights[i], kWeights[i], device);
    }

    gemma4_e2b_decode::Params prm{};
    prm.hidden = reinterpret_cast<bf16 const*>(hiddenStates.data_ptr());
    prm.ple_tok = reinterpret_cast<bf16 const*>(pleTokenEmbeds.data_ptr());
    prm.cache_lens = cacheLens.data_ptr<int>();
    int64_t const* meta = kvMeta.data_ptr<int64_t>();
    for (int l = 0; l < 15; ++l)
    {
        prm.kvk[l] = reinterpret_cast<uint8_t*>(meta[l]);
        prm.kvv[l] = reinterpret_cast<uint8_t*>(meta[15 + l]);
        prm.kvps[l] = meta[30 + l];
    }
    prm.pt_s = pageTableSliding.data_ptr<int>();
    prm.pt_f = pageTableFull.data_ptr<int>();
    prm.pt_indptr = pageTableIndptr.data_ptr<int>();

    auto bfp = [&](int i) { return reinterpret_cast<bf16 const*>(weights[i].data_ptr()); };
    auto u8p = [&](int i) { return reinterpret_cast<uint8_t const*>(weights[i].data_ptr()); };
    auto f32p = [&](int i) { return weights[i].data_ptr<float>(); };
    prm.ple_mp_w = bfp(0);
    prm.ple_pn_w = bfp(1);
    prm.ln_w = bfp(2);
    prm.lscalar = f32p(3);
    prm.ple_gate_w = bfp(4);
    prm.ple_proj_w = bfp(5);
    prm.final_w = bfp(6);
    prm.a_qkv = u8p(7);
    prm.a_o = u8p(8);
    prm.a_gu = u8p(9);
    prm.a_down = u8p(10);
    prm.a_qkn = bfp(11);
    prm.a_sc = f32p(12);
    prm.b_qkv = u8p(13);
    prm.b_o = u8p(14);
    prm.b_gu = u8p(15);
    prm.b_down = u8p(16);
    prm.b_qkn = bfp(17);
    prm.b_sc = f32p(18);
    prm.c_q = bfp(19);
    prm.c_o = u8p(20);
    prm.c_gu = u8p(21);
    prm.c_down = u8p(22);
    prm.c_qn = bfp(23);
    prm.c_sc = f32p(24);
    prm.d_q = bfp(25);
    prm.d_o = u8p(26);
    prm.d_gu = u8p(27);
    prm.d_down = u8p(28);
    prm.d_qn = bfp(29);
    prm.d_sc = f32p(30);
    prm.out = reinterpret_cast<bf16*>(out.data_ptr());
    prm.T = static_cast<int>(numTokens);
    prm.R = static_cast<int>(numRequests);

    c10::cuda::CUDAGuard const deviceGuard{device};
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device.index());
    cudaError_t const status
        = launchGemma4E2bDecodeStack(prm, numSms, at::cuda::getCurrentCUDAStream(device.index()).stream());
    TORCH_CHECK(status == cudaSuccess, "decode_stack launch failed: ", cudaGetErrorString(status));
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "decode_stack(Tensor hidden_states, Tensor ple_token_embeds, Tensor cache_lens, Tensor kv_meta, "
        "Tensor pt_s, Tensor pt_f, Tensor pt_indptr, Tensor[] weights, Tensor(a!) out) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("decode_stack", &TRTLLM_MV2_NS::decodeStack);
}
