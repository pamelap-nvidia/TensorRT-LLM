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

// Gemma-4 E2B-it decoder stack (PLE projection, 35 decoder layers, final RMSNorm) for R <= 8 requests x Q <= 8
// tokens, FP8 ModelOpt weights, FP8 paged KV cache: one persistent cooperative kernel, phases separated by grid
// barriers, weights streamed into L2 ahead of use with bulk prefetches.
//
// Provenance: Kernel Factory solution 382d6e41d928 ("gemma4_e2b_fp8_decoder_stack_spec_verify", SM120), with the
// contiguous per-request KV caches replaced by TensorRT-LLM's paged caches (page tables per KV pool, page size
// 1 << KV_PAGE_SHIFT, one KV head). The math is the solution's; see the target's ops/decode_stack.py for the
// contract and the unit test for the PyTorch reference it is checked against.

// Tuning switches of the Kernel Factory solution (its compile options).
#define XC3_RECURRENCE
#define POLL_NS 16
#define VEC_SMEM_V
#define ATTN_CPASYNC_V
#define VEC_RED
#define RELAXED_POLL2
#define NPF_SMALL 28
#define NPF_NT1 31
#define NPF_LARGE 12
#define PF_CH 65536
#define PF_QO
#define PF_BULK
#define PF_LEAN
#define NO_ATTN_PF
#define ATTN_K_FIRST_MT2
#define ATTN_EXTERNAL_PF_K_ONLY
#define PF_WIN_NT3 96
#define WEIGHT_L2_256
#define PF_ALL_MAXNT 3
#define PRUNE_GEMM_TAIL_BARRIER
#define PF_WIN_SMALL 80
#define PF_WIN_LARGE 80
#define ATT_QK2_MT1
#define V_LDSM
#define NO_FP8_CLAMP
#define PF_RATE_ALL_NT1 1700ull
#define PF_KVPROD_ALL 1
#define PF_RATE_ALL_NT2 1700ull
#define ATTN_PREBAR_V
#define PF_RATE_ALL_NT3 1500ull
#define PF_LARGE_ALL
#define PF_VT_LARGE 4
#define PF_RATE_LARGE 1500ull
#define PF_WIN_LARGE 52
#define PF_RATE_KVPROD 1100ull
#define WEIGHT_EF2
#define KV_PERSIST 3
#define KV_POL_EF
#define PF_KVL
#define ATT_PV_F8
#define PF_RATE_KVPROD_MAX_T 40
#define SINGLE_PASS_COMBINE
#define PLE_LAZY_MINNT 3
#define PLE_LAZY_MAXNT 3
#define PLE_P3_MINNT 3
#define PF_HOLD_MINNT 2
#define PF_KVCAP_MID 1300ull
#define PLE_LAZY_EXTRA_NT 6
#define COMB_B 12
#define COMB_B_MAXNT 3
#define CTA_SP
#define PF_KVL_MINT 1
#define COMB_B_GE2 10
#define VEC_SMEM_S
#ifndef KV_PAGE_SHIFT
#define KV_PAGE_SHIFT 5
#endif

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gemma4/e2b_it_fp8__sm_120__tp1/csrc/gemma4E2bDecodeKernel.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace gemma4_e2b_decode
{

namespace cg = cooperative_groups;

#define HID 1536
#define NLAYER 35
#define PLED 256
#define EPSV 1e-6f
#define TMAX 64
#define NTHR 256
#define MAXS 8
#define MAXITEMS 256
#define NREG (1 + 6 * NLAYER)
#ifndef NSM_SMALL
#define NSM_SMALL 160
#endif
#ifndef NSM_LARGE
#define NSM_LARGE 176
#endif
#ifndef NSM_T_SPLIT
#define NSM_T_SPLIT 24
#endif
#ifndef LOADFIRST_MIN_NT
#define LOADFIRST_MIN_NT 1
#endif
#ifndef PG_SPLIT
#define PG_SPLIT 6
#endif
#ifndef PF_LA_MB
#define PF_LA_MB 48
#endif
#define PF_LOOKAHEAD ((unsigned long long) PF_LA_MB * 1024ull * 1024ull)
// Dedicated L2 weight-prefetch CTAs appended after the compute CTAs (they run on SMs the compute grid leaves idle).
#ifndef NPF_SMALL
#define NPF_SMALL 0
#endif
#ifndef NPF_LARGE
#define NPF_LARGE 0
#endif
#ifndef NPF_NT1
#define NPF_NT1 NPF_SMALL
#endif
#define NPF_OF(NT) ((NT) == 1 ? NPF_NT1 : (NT) <= 3 ? NPF_SMALL : NPF_LARGE)

// ---------------------------------------------------------------- scratch
__device__ bf16 g_h[TMAX * HID];
__device__ __align__(16) uint8_t g_xa[TMAX * HID];
__device__ __align__(16) bf16 g_xbf[TMAX * HID];
__device__ __align__(16) uint8_t g_xb[TMAX * 12288];
__device__ __align__(16) bf16 g_qkv[TMAX * 5120];
__device__ __align__(16) float g_qpart[2 * TMAX * 4096]; // split-K partials of the BF16 q projections
__device__ __align__(16) float g_ypart[MAXS * TMAX * HID];
__device__ __align__(16) __half g_ypart_h[TMAX * HID];   // scaled PLE-projection output
__device__ __align__(16) bf16 g_pproj[TMAX * NLAYER * PLED];
__device__ __align__(16) bf16 g_pli[TMAX * NLAYER * PLED];
__device__ __align__(16) bf16 g_gbuf[TMAX * PLED];
__device__ __align__(16) float g_gpart[MAXS * TMAX * PLED];
__device__ unsigned int g_gcnt[16];
#define AROWS 32
__device__ __align__(16) __half g_opart[MAXITEMS * 32 * 512];
__device__ __align__(16) float2 g_ml[MAXITEMS * 32];
__device__ __align__(16) float2 g_rope[TMAX * 192]; // per token: 128 sliding + 64 full (cos, sin)
#define NBAR 512
__device__ unsigned int g_bars[NBAR];
__device__ unsigned int g_attn_ready[NLAYER * 32];
__device__ unsigned int g_done;
#ifdef ATTN_TIMING
__device__ unsigned long long g_atime[NLAYER * 192 * 4];
__device__ unsigned long long g_acyc[NLAYER * 192 * 4];
__shared__ int s_curlayer;

__device__ __forceinline__ unsigned long long gtimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#define ATT_STAMP(L, k)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (threadIdx.x == 0)                                                                                          \
            g_atime[((L) *192 + blockIdx.x) * 4 + (k)] = gtimer();                                                     \
    } while (0)
#else
#define ATT_STAMP(L, k)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

#ifndef KV_PAGE_SHIFT
#define KV_PAGE_SHIFT 5
#endif
#define KV_PS (1 << KV_PAGE_SHIFT)

__device__ __forceinline__ int const* kv_rpt(Params const& p, bool sliding, int r)
{
    return (sliding ? p.pt_s : p.pt_f) + __ldg(p.pt_indptr + r);
}

__device__ __forceinline__ size_t kv_off(int const* rpt, long long ps, int kk, int hd)
{
    return (size_t) __ldg(rpt + (kk >> KV_PAGE_SHIFT)) * (size_t) ps + (size_t) (kk & (KV_PS - 1)) * (size_t) hd;
}

// kind/j (layer_kind numbering) -> KV-owning layer: A j -> j + j/4, B j -> 5j+4, C -> 13, D -> 14
__device__ __forceinline__ void kv_layer(Params const& p, int kind, int j, uint8_t*& kc, uint8_t*& vc, long long& ps)
{
    int const l = kind == 0 ? j + j / 4 : kind == 1 ? 5 * j + 4 : kind == 2 ? 13 : 14;
    kc = p.kvk[l];
    vc = p.kvv[l];
    ps = p.kvps[l];
}

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ float bf2f(bf16 x)
{
    return __bfloat162float(x);
}

union Bf16Pair
{
    uint32_t u;
    bf16 b[2];
};

union Half2Pair
{
    uint32_t u;
    __half2 h;
};

__device__ __forceinline__ float2 ldg_bf2(bf16 const* p)
{
    Bf16Pair v;
    v.u = __ldcg((uint32_t const*) p);
    return make_float2(bf2f(v.b[0]), bf2f(v.b[1]));
}

__device__ __forceinline__ void stg_bf2(bf16* p, float x, float y)
{
    Bf16Pair v;
    v.b[0] = __float2bfloat16_rn(x);
    v.b[1] = __float2bfloat16_rn(y);
    *(uint32_t*) p = v.u;
}

__device__ __forceinline__ float4 ldg_h4(__half const* p)
{
    const uint2 u = __ldcg((uint2 const*) p);
    Half2Pair a, b;
    a.u = u.x;
    b.u = u.y;
    const float2 x = __half22float2(a.h), y = __half22float2(b.h);
    return make_float4(x.x, x.y, y.x, y.y);
}

__device__ __forceinline__ float bfr(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint8_t fp8q(float v)
{
#ifndef NO_FP8_CLAMP
    v = fminf(fmaxf(v, -448.f), 448.f);
#endif
    return (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float gelu_tanh(float x)
{
    float const kBeta = 0.7978845608028654f, kKappa = 0.044715f;
    float inner = kBeta * (x + kKappa * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

__device__ __forceinline__ uint4 ldg_w(void const* p)
{
    uint4 r;
#ifdef WEIGHT_LD_CS
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#elif defined(WEIGHT_L2_256) && defined(WEIGHT_EF2)
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p), "l"(pol));
#elif defined(WEIGHT_L2_256)
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
#elif defined(WEIGHT_EF)
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
#endif
    return r;
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(uint32_t v16)
{
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (v16 & 0xffff), __NV_E4M3);
    return (uint32_t) h.x | ((uint32_t) h.y << 16);
}

__device__ __forceinline__ void mma_f8(
    float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(
    float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(
    float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* sred)
{
    v = warp_sum(v);
    int const w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
        sred[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NTHR / 32; ++i)
        t += sred[i];
    return t;
}

// Each phase has a private arrival counter, so blocks without work may skip its wait and run ahead to
// preload the next phase without aliasing a later grid barrier.
#ifdef PHASE_TIMING
__device__ unsigned long long g_tdone[NBAR];
#endif
__device__ __forceinline__ void grid_arrive(unsigned int& target, unsigned int const need)
{
    __syncthreads();
    target += 1;
    if (threadIdx.x == 0)
    {
#ifdef PHASE_TIMING
        unsigned int old;
        asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(&g_bars[target - 1]) : "memory");
        if (old == need - 1)
        {
            unsigned long long tt;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt));
            g_tdone[target - 1] = tt;
        }
#else
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_bars[target - 1]) : "memory");
#endif
    }
}

__device__ __forceinline__ void grid_wait(unsigned int const target, unsigned int const need)
{
    if (threadIdx.x == 0)
    {
        unsigned int v;
        unsigned int const* bp = &g_bars[target - 1];
#if defined(RELAXED_POLL2)
        // Poll relaxed so an unrelated in-flight weight load cannot stall each retry.  Once the counter appears
        // complete, pair the producers' release with an acquire pattern before consuming cross-CTA results.
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        while (v < need)
        {
            __nanosleep(POLL_NS);
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        }
        // PTX acquire pattern: a strong relaxed read of this location followed in program order by an
        // acquire fence.  This pairs with the producers' release reduction without re-reading the counter.
        asm volatile("fence.acquire.gpu;" ::: "memory");
#elif defined(RELAXED_POLL)
        // relaxed polling (does not wait on this thread's outstanding weight loads), then one acquire fence
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        while (v < need)
        {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        while (v < need)
        {
#ifdef POLL_NS
            __nanosleep(POLL_NS);
#endif
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        }
#endif
    }
    __syncthreads();
}

__device__ __forceinline__ void grid_sync_(unsigned int& target, unsigned int const need)
{
    grid_arrive(target, need);
    grid_wait(target, need);
}

__device__ __forceinline__ void grid_sync(unsigned int& target, unsigned int const need)
{
#ifdef USE_CG_SYNC
    (void) target;
    cg::this_grid().sync();
#else
    grid_sync_(target, need);
#ifdef BAR_TWICE
    grid_sync_(target, need);
#endif
#endif
}

__device__ __forceinline__ void attn_wait_warp(unsigned int const* bp, unsigned int need)
{
    if ((threadIdx.x & 31) == 0)
    {
        unsigned int v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        while (v < need)
        {
            __nanosleep(16);
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        }
    }
    __syncwarp();
}

__device__ __forceinline__ void attn_wait_cta(unsigned int const* bp, unsigned int need)
{
    if (threadIdx.x == 0)
    {
        unsigned int v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        while (v < need)
        {
            __nanosleep(16);
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bp) : "memory");
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------- layer bookkeeping
__device__ __forceinline__ void layer_kind(int i, int& kind, int& j)
{
    if (i < 15)
    {
        if (i % 5 == 4)
        {
            kind = 1;
            j = i / 5;
        }
        else
        {
            kind = 0;
            j = i - i / 5;
        }
    }
    else
    {
        int ii = i - 15;
        if (i % 5 == 4)
        {
            kind = 3;
            j = (i - 19) / 5;
        }
        else
        {
            kind = 2;
            j = ii - ii / 5;
        }
    }
}

__device__ void region(Params const& p, int k, uint8_t const*& ptr, unsigned long long& sz)
{
    if (k == 0)
    {
        ptr = (uint8_t const*) p.ple_mp_w;
        sz = 8960ull * 1536 * 2;
        return;
    }
    int i = (k - 1) / 6, r = (k - 1) % 6, kind, j;
    layer_kind(i, kind, j);
    const size_t J = (size_t) j;
    switch (r)
    {
    case 0:
        if (kind == 0)
        {
            ptr = p.a_qkv + J * 2560 * 1536;
            sz = 2560ull * 1536;
        }
        else if (kind == 1)
        {
            ptr = p.b_qkv + J * 5120 * 1536;
            sz = 5120ull * 1536;
        }
        else if (kind == 2)
        {
            ptr = (uint8_t const*) (p.c_q + J * 2048 * 1536);
            sz = 2048ull * 1536 * 2;
        }
        else
        {
            ptr = (uint8_t const*) (p.d_q + J * 4096 * 1536);
            sz = 4096ull * 1536 * 2;
        }
        break;
    case 1:
        if (kind == 0)
        {
            ptr = p.a_o + J * 1536 * 2048;
            sz = 1536ull * 2048;
        }
        else if (kind == 1)
        {
            ptr = p.b_o + J * 1536 * 4096;
            sz = 1536ull * 4096;
        }
        else if (kind == 2)
        {
            ptr = p.c_o + J * 1536 * 2048;
            sz = 1536ull * 2048;
        }
        else
        {
            ptr = p.d_o + J * 1536 * 4096;
            sz = 1536ull * 4096;
        }
        break;
    case 2:
        if (kind == 0)
        {
            ptr = p.a_gu + J * 12288 * 1536;
            sz = 12288ull * 1536;
        }
        else if (kind == 1)
        {
            ptr = p.b_gu + J * 12288 * 1536;
            sz = 12288ull * 1536;
        }
        else if (kind == 2)
        {
            ptr = p.c_gu + J * 24576 * 1536;
            sz = 24576ull * 1536;
        }
        else
        {
            ptr = p.d_gu + J * 24576 * 1536;
            sz = 24576ull * 1536;
        }
        break;
    case 3:
        if (kind == 0)
        {
            ptr = p.a_down + J * 1536 * 6144;
            sz = 1536ull * 6144;
        }
        else if (kind == 1)
        {
            ptr = p.b_down + J * 1536 * 6144;
            sz = 1536ull * 6144;
        }
        else if (kind == 2)
        {
            ptr = p.c_down + J * 1536 * 12288;
            sz = 1536ull * 12288;
        }
        else
        {
            ptr = p.d_down + J * 1536 * 12288;
            sz = 1536ull * 12288;
        }
        break;
    case 4:
        ptr = (uint8_t const*) (p.ple_gate_w + (size_t) i * 256 * 1536);
        sz = 256ull * 1536 * 2;
        break;
    default:
        ptr = (uint8_t const*) (p.ple_proj_w + (size_t) i * 1536 * 256);
        sz = 1536ull * 256 * 2;
        break;
    }
}

// ---------------------------------------------------------------- GEMM (weights streamed, T <= 64 tokens)
enum
{
    EPI_PLEMP = 0,
    EPI_QKV = 1,
    EPI_PART = 2,
    EPI_GU = 3,
    EPI_PLEGATE = 4,
    EPI_PART16 = 5
};

struct GemmArgs
{
    uint8_t const* W;
    long ldw; // bytes per weight row
    uint8_t const* X;
    long ldx; // bytes per activation row
    int Kb;   // reduction length in bytes
    int mtiles, S, gu_inter;
    int epi;
    float sc, sc2;
    void* out;
    long ldo;
    bf16 const* aux;
};

struct LazyGemmArgs : GemmArgs
{
    int vb, vnb;
    float* part;
};

template <int EPI, bool ALT_PART = false>
__device__ __forceinline__ void store_gemm_value(
    GemmArgs const& a, float x, int col, int n, int ss, float* part = nullptr)
{
    static_assert(EPI >= EPI_PLEMP && EPI <= EPI_PART16);
    if constexpr (EPI == EPI_PLEMP)
    {
        ((bf16*) a.out)[(long) col * a.ldo + n] = __float2bfloat16_rn(bfr(x) * 0.025515518153991442f);
    }
    else if constexpr (EPI == EPI_QKV)
    {
        if (a.S > 1)
        {
            if constexpr (ALT_PART)
                part[((size_t) ss * TMAX + col) * PLED + n] = x;
            else
                g_qpart[((size_t) ss * TMAX + col) * 4096 + n] = x;
            return;
        }
        ((bf16*) a.out)[(long) col * a.ldo + n] = __float2bfloat16_rn(x * a.sc);
    }
    else if constexpr (EPI == EPI_PART)
    {
        ((float*) a.out)[(long) ss * TMAX * HID + (long) col * HID + n] = x;
    }
    else if constexpr (EPI == EPI_PART16)
    {
        ((__half*) a.out)[(long) col * HID + n] = __float2half_rn(x * 0.0625f);
    }
    else
    {
        static_assert(EPI == EPI_PLEGATE);
        float const gt = bfr(x);
        float const gl = bfr(gelu_tanh(gt));
        float const pl = bf2f(__ldcg(a.aux + (long) col * (NLAYER * PLED) + n));
        ((bf16*) a.out)[(long) col * PLED + n] = __float2bfloat16_rn(gl * pl);
    }
}

__device__ __forceinline__ uint4 ld_x(void const* p)
{
    uint4 r;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// Block tasks: blockIdx.x + k*gridDim.x (task = 16-row m-tile x k-split s).  Tasks are processed in waves of
// MW in {8,4,2,1}; in a wave warp w owns task (w % MW) over K-slice (w / MW) of KS = 8/MW interleaved 64-byte
// chunks.  Weight loads run through a D-deep register ring; activation fragments are L1-cached and fetched one
// unit ahead.  Per-warp partials are reduced through smem in a fixed order (deterministic).
template <int NT>
struct GemmD
{
#ifdef GEMM_D4
    static constexpr int D = NT >= 6 ? 2 : 4;
#else
    static constexpr int D = NT >= 6 ? 2 : 3;
#endif
};

template <int NT>
struct GemmRed
{
    static constexpr int NT8 = NT * 8;
#ifdef VEC_RED
    static constexpr bool V = NT >= 2;
    static constexpr bool V4 = NT >= 5;
    // RS % 32 == 8 makes each 16-lane float2 subrequest cover all 32 banks.
    static constexpr int RS = V ? (((NT8 + 23) & ~31) + 8) : (NT8 + 1);
    static constexpr int TS = 16 * RS;

    __device__ __forceinline__ static int off(int row, int col)
    {
        return row * RS + col;
    }
#elif defined(BANKFREE_RED)
    static constexpr bool V = false;
    static constexpr int CS = 4 * NT8 + 8;
    static constexpr int TS = 4 * CS;

    __device__ __forceinline__ static int off(int row, int col)
    {
        return ((row >> 3) * 2 + (col & 1)) * CS + (col >> 1) * 8 + (row & 7);
    }
#else
    static constexpr bool V = false;
    static constexpr int RS = NT8 + 1;
    static constexpr int TS = 16 * RS;

    __device__ __forceinline__ static int off(int row, int col)
    {
        return row * RS + col;
    }
#endif
};

__device__ __forceinline__ void cp_async16(void* smem_dst, void const* gsrc)
{
    unsigned const sa = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gsrc) : "memory");
}

__device__ __forceinline__ uint64_t make_kvpol(bool last)
{
    uint64_t pol;
#ifdef KV_POL_EF
    // KV read again by later layers keeps normal priority; KV read once streams evict-first
    if (last)
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
    else
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#else
    if (last)
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    else
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
#endif
    return pol;
}

__device__ __forceinline__ uint4 ldcg_hint(void const* ptr, uint64_t pol)
{
    uint4 r;
    asm("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(ptr), "l"(pol));
    return r;
}

__device__ __forceinline__ void cp_async16_hint(void* smem_dst, void const* gsrc, uint64_t pol)
{
    unsigned const sa = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(sa), "l"(gsrc), "l"(pol)
                 : "memory");
}
#if defined(ATTN_NOLOAD)
#define KLDG(ptr) make_uint4(0x30303030u, 0x30303030u, 0x30303030u, 0x30303030u)
#define VCPA(d, src)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#elif defined(KV_PERSIST)
#define KLDG(ptr) ldcg_hint((ptr), kvpol)
#define VCPA(d, src) cp_async16_hint((d), (src), kvpol)
#else
#define KLDG(ptr) __ldcg(ptr)
#define VCPA(d, src) cp_async16((d), (src))
#endif

template <int NT, int EPI, bool ALT_PART = false>
__device__ __forceinline__ void gemm_epilogue(
    GemmArgs const& a, int T, float const* slots, int mt, int ss, float* part = nullptr)
{
    using GR = GemmRed<NT>;
    constexpr int NT8 = GR::NT8;
    constexpr int TS = GR::TS;
    int const tid = threadIdx.x, lane = tid & 31;
    int const g = lane >> 2, c = lane & 3;
    if constexpr (EPI == EPI_PLEGATE)
    {
        if (a.S > 1)
        {
            // split-K PLE gate: write fp32 partial, the last of the S splits of this m-tile reduces in fixed order
            __shared__ int s_last;
#ifdef VEC_RED
            if constexpr (GR::V4)
            {
                for (int idx = tid; idx < 4 * NT8; idx += NTHR)
                {
                    int const row = idx & 15, col = (idx >> 4) * 4;
                    if (col < T)
                    {
                        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                        {
                            const float4 z = *(float4 const*) (slots + q * TS + GR::off(row, col));
                            v.x += z.x;
                            v.y += z.y;
                            v.z += z.z;
                            v.w += z.w;
                        }
                        int const n = mt * 16 + row;
                        g_gpart[((size_t) ss * TMAX + col) * PLED + n] = v.x;
                        if (col + 1 < T)
                            g_gpart[((size_t) ss * TMAX + col + 1) * PLED + n] = v.y;
                        if (col + 2 < T)
                            g_gpart[((size_t) ss * TMAX + col + 2) * PLED + n] = v.z;
                        if (col + 3 < T)
                            g_gpart[((size_t) ss * TMAX + col + 3) * PLED + n] = v.w;
                    }
                }
            }
            else if constexpr (GR::V)
            {
                for (int idx = tid; idx < 8 * NT8; idx += NTHR)
                {
                    int const tile = idx >> 5;
                    int const row = ((tile & 1) << 3) + g;
                    int const col = ((tile >> 1) << 3) + 2 * c;
                    if (col < T)
                    {
                        float2 v = make_float2(0.f, 0.f);
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                        {
                            const float2 z = *(float2 const*) (slots + q * TS + GR::off(row, col));
                            v.x += z.x;
                            v.y += z.y;
                        }
                        int const n = mt * 16 + row;
                        g_gpart[((size_t) ss * TMAX + col) * PLED + n] = v.x;
                        if (col + 1 < T)
                            g_gpart[((size_t) ss * TMAX + col + 1) * PLED + n] = v.y;
                    }
                }
            }
            else
#endif
            {
                for (int idx = tid; idx < 16 * NT8; idx += NTHR)
                {
                    int const row = idx & 15, col = idx >> 4;
                    if (col < T)
                    {
                        float v = 0.f;
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                            v += slots[q * TS + GR::off(row, col)];
                        g_gpart[((size_t) ss * TMAX + col) * PLED + mt * 16 + row] = v;
                    }
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                __threadfence();
                unsigned int const old = atomicAdd(&g_gcnt[mt], 1u);
                s_last = (old == (unsigned) (a.S - 1));
            }
            __syncthreads();
            if (s_last)
            {
                __threadfence();
                if constexpr (GR::V4)
                {
                    for (int idx = tid; idx < 4 * T; idx += NTHR)
                    {
                        int const row = (idx & 3) * 4, col = idx >> 2;
                        int const n = mt * 16 + row;
                        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                        for (int q = 0; q < 6; ++q)
                        {
                            const float4 z = __ldcg((float4 const*) &g_gpart[((size_t) q * TMAX + col) * PLED + n]);
                            v.x += z.x;
                            v.y += z.y;
                            v.z += z.z;
                            v.w += z.w;
                        }
                        const float2 p01 = ldg_bf2(a.aux + (long) col * (NLAYER * PLED) + n);
                        const float2 p23 = ldg_bf2(a.aux + (long) col * (NLAYER * PLED) + n + 2);
                        float const g0 = bfr(gelu_tanh(bfr(v.x))), g1 = bfr(gelu_tanh(bfr(v.y)));
                        float const g2 = bfr(gelu_tanh(bfr(v.z))), g3 = bfr(gelu_tanh(bfr(v.w)));
                        bf16* out = (bf16*) a.out + (long) col * PLED + n;
                        stg_bf2(out, g0 * p01.x, g1 * p01.y);
                        stg_bf2(out + 2, g2 * p23.x, g3 * p23.y);
                    }
                }
                else
                {
                    for (int idx = tid; idx < 8 * T; idx += NTHR)
                    {
                        int const row = (idx & 7) * 2, col = idx >> 3;
                        int const n = mt * 16 + row;
                        float2 v = make_float2(0.f, 0.f);
#pragma unroll
                        for (int q = 0; q < 6; ++q)
                        {
                            const float2 z = __ldcg((float2 const*) &g_gpart[((size_t) q * TMAX + col) * PLED + n]);
                            v.x += z.x;
                            v.y += z.y;
                        }
                        const float2 pl = ldg_bf2(a.aux + (long) col * (NLAYER * PLED) + n);
                        float const g0 = bfr(gelu_tanh(bfr(v.x)));
                        float const g1 = bfr(gelu_tanh(bfr(v.y)));
                        stg_bf2((bf16*) a.out + (long) col * PLED + n, g0 * pl.x, g1 * pl.y);
                    }
                }
                if (tid == 0)
                    g_gcnt[mt] = 0u;
            }
            return;
        }
    }
    if constexpr (EPI == EPI_GU)
    {
#ifdef VEC_RED
        if constexpr (GR::V)
        {
            for (int idx = tid; idx < 4 * NT8; idx += NTHR)
            {
                int const tile = idx >> 5;
                int const row = g, col = (tile << 3) + 2 * c;
                if (col < T)
                {
                    float2 gs = make_float2(0.f, 0.f), us = make_float2(0.f, 0.f);
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                    {
                        const float2 g2 = *(float2 const*) (slots + q * TS + GR::off(row, col));
                        const float2 u2 = *(float2 const*) (slots + q * TS + GR::off(row + 8, col));
                        gs.x += g2.x;
                        gs.y += g2.y;
                        us.x += u2.x;
                        us.y += u2.y;
                    }
                    float const gate0 = bfr(gs.x * a.sc), up0 = bfr(us.x * a.sc);
                    ((uint8_t*) a.out)[(long) col * a.ldo + mt * 8 + row] = fp8q(bfr(gelu_tanh(gate0) * up0) / a.sc2);
                    if (col + 1 < T)
                    {
                        float const gate1 = bfr(gs.y * a.sc), up1 = bfr(us.y * a.sc);
                        ((uint8_t*) a.out)[(long) (col + 1) * a.ldo + mt * 8 + row]
                            = fp8q(bfr(gelu_tanh(gate1) * up1) / a.sc2);
                    }
                }
            }
        }
        else
#endif
        {
            for (int idx = tid; idx < 8 * NT8; idx += NTHR)
            {
                int const row = idx & 7, col = idx >> 3;
                if (col < T)
                {
                    float gs = 0.f, us = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                    {
                        gs += slots[q * TS + GR::off(row, col)];
                        us += slots[q * TS + GR::off(row + 8, col)];
                    }
                    float const gate = bfr(gs * a.sc), up = bfr(us * a.sc);
                    float const act = bfr(gelu_tanh(gate) * up);
                    ((uint8_t*) a.out)[(long) col * a.ldo + mt * 8 + row] = fp8q(act / a.sc2);
                }
            }
        }
    }
    else
    {
#ifdef VEC_RED
        if constexpr (GR::V)
        {
            for (int idx = tid; idx < 8 * NT8; idx += NTHR)
            {
                int const tile = idx >> 5;
                int const row = ((tile & 1) << 3) + g;
                int const col = ((tile >> 1) << 3) + 2 * c;
                if (col < T)
                {
                    float2 v = make_float2(0.f, 0.f);
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                    {
                        const float2 z = *(float2 const*) (slots + q * TS + GR::off(row, col));
                        v.x += z.x;
                        v.y += z.y;
                    }
                    int const n = mt * 16 + row;
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                    {
                        int const cj = col + j;
                        if (cj < T)
                        {
                            float const x = j ? v.y : v.x;
                            store_gemm_value<EPI, ALT_PART>(a, x, cj, n, ss, part);
                        }
                    }
                }
            }
        }
        else
#endif
        {
            for (int idx = tid; idx < 16 * NT8; idx += NTHR)
            {
                int const row = idx & 15, col = idx >> 4;
                if (col < T)
                {
                    float v = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; ++q)
                        v += slots[q * TS + GR::off(row, col)];
                    int const n = mt * 16 + row;
                    store_gemm_value<EPI, ALT_PART>(a, v, col, n, ss, part);
                }
            }
        }
    }
}

// Block tasks: blockIdx.x + k*gridDim.x (task = 16-row m-tile x k-split s), processed one at a time by all 8
// warps (warp w takes 64-byte chunks w, w+8, ... so a block streams 512 contiguous bytes per weight row).
// The weight-load ring is D units deep and runs across task boundaries; each task's per-warp partials are
// reduced in fixed order through double-buffered smem slots (one block sync per task, deterministic).
// Task mapping: block b always works on K-slice s = b % S (m-tiles q, q + nbs, ... with q = b / S), so all of a
// block's tasks read the same activation columns.  XC > 0: the block's activation fragments for its K-slice
// (cpw == XC units) are loaded once into registers after the barrier wait and reused by every task.
template <int NT, bool F8, int XC, int EPI, bool VIRTUAL = false, bool ALT_PART = false, typename GA = GemmArgs>
__device__ __noinline__ void gemm(const GA& ga, int T, float* red, unsigned int const* wt)
{
    constexpr int D = XC == 0 ? GemmD<NT>::D : 3;
    using GR = GemmRed<NT>;
    constexpr int NT8 = GR::NT8;
    constexpr int TS = GR::TS;
    const GA a = ga;
    float* part = nullptr;
    if constexpr (ALT_PART)
        part = a.part;
    int const tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
    int const S = a.S;
    int const Ks = a.Kb / S;
    int const cpw = XC > 0 ? XC : (Ks >> 9); // reused paths make the hot task boundary compile-time
    int nb;
    if constexpr (VIRTUAL)
        nb = a.vnb;
    else
        nb = (int) gridDim.x - NPF_OF(NT);
#ifdef GEMM_NB
    if (nb > GEMM_NB)
        nb = GEMM_NB;
#endif
    int bid;
    if constexpr (VIRTUAL)
        bid = a.vb;
    else
        bid = (int) blockIdx.x;
    if (bid >= nb)
        return;
    int const ss = bid % S, q0 = bid / S;
    int const nbs = (nb - ss + S - 1) / S; // blocks sharing K-slice ss
    int const ntl = q0 < a.mtiles ? (a.mtiles - q0 + nbs - 1) / nbs : 0;
    if (ntl <= 0)
        return;
    int const nu = ntl * cpw;
    int const rpm = a.gu_inter ? 8 : 16;
    long const hid = (long) (a.gu_inter ? a.gu_inter : 8) * a.ldw;
    int const kslice = ss * Ks;
    uint8_t const* wbase = a.W + (long) g * a.ldw + w * 64 + 16 * c + kslice;
    uint8_t const* xbase = a.X + w * 64 + kslice;
    int xoff[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        xoff[nt] = min(nt * 8 + g, T - 1) * (int) a.ldx + 16 * c;
#define CUR_NEXT(K, I, WO, KO)                                                                                         \
    {                                                                                                                  \
        if (++(I) == cpw)                                                                                              \
        {                                                                                                              \
            (I) = 0;                                                                                                   \
            ++(K);                                                                                                     \
            KO = 0;                                                                                                    \
            WO = (long) ((q0 + (K) *nbs) * rpm) * a.ldw;                                                               \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            (WO) += 512;                                                                                               \
            (KO) += 512;                                                                                               \
        }                                                                                                              \
    }
    int lk = 0, li = 0, lko = 0;
    long lwo = (long) (q0 * rpm) * a.ldw;
    uint4 wl[D], wh[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
    {
        if (d < nu)
        {
            wl[d] = ldg_w(wbase + lwo);
            wh[d] = ldg_w(wbase + lwo + hid);
            CUR_NEXT(lk, li, lwo, lko);
        }
    }
#ifdef XC3_DEEP
    // second weight set (task 1) for the two-task-deep ring of the XC == 3 path
    uint4 wl2[3], wh2[3];
    if constexpr (XC == 3)
    {
        if (ntl > 1)
        {
            long const o1 = (long) (q0 + nbs) * rpm * a.ldw;
#pragma unroll
            for (int d = 0; d < 3; ++d)
            {
                wl2[d] = ldg_w(wbase + o1 + d * 512);
                wh2[d] = ldg_w(wbase + o1 + d * 512 + hid);
            }
        }
    }
#endif
#ifdef XC3_TASK_PF_SMALL
    if constexpr (XC == 3 && NT <= 6)
    {
        if (ntl > 1 && w < 2)
        {
            long const po = (long) (q0 + nbs) * rpm * a.ldw;
#pragma unroll
            for (int pu = 0; pu < 3; ++pu)
            {
                uint8_t const* pp = a.W + po + (long) g * a.ldw + kslice + pu * 512 + c * 128 + (w ? hid : 0);
                asm volatile("prefetch.global.L2 [%0];" ::"l"(pp));
            }
        }
    }
#endif
    if (wt)
        grid_wait(*wt, gridDim.x - NPF_OF(NT)); // activations are produced by the previous phase
    constexpr int XR = XC > 0 ? XC : 1;
    uint4 xr[XR][NT];                           // XC > 0: block-constant activation fragments
    uint4 xc[NT];                               // XC == 0: streamed fragments of the current unit
    int xk = 0, xi = 0, xko = 0;
    long xwo = 0;
    if (XC > 0)
    {
#pragma unroll
        for (int i = 0; i < XR; ++i)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                xr[i][nt] = ld_x(xbase + xoff[nt] + i * 512);
    }
    else
    {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            xc[nt] = ld_x(xbase + xoff[nt]);
        CUR_NEXT(xk, xi, xwo, xko);
    }
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
    {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
#ifdef XC3_DEEP
    if constexpr (XC == 3)
    {
        // Two weight sets alternate between tasks: task kt+2's loads are issued right after task kt's MMAs,
        // so each load has two task periods to arrive.
        long const task_stride = (long) nbs * rpm * a.ldw;
        long next_wo = (long) (q0 + 2 * nbs) * rpm * a.ldw;
#define XC3_TASK(WL, WH, KT)                                                                                           \
    {                                                                                                                  \
        _Pragma("unroll") for (int d = 0; d < 3; ++d)                                                                  \
        {                                                                                                              \
            _Pragma("unroll") for (int nt = 0; nt < NT; ++nt)                                                          \
            {                                                                                                          \
                const uint4 xv = xr[d][nt];                                                                            \
                if (F8)                                                                                                \
                {                                                                                                      \
                    mma_f8(acc[nt], WL[d].x, WH[d].x, WL[d].y, WH[d].y, xv.x, xv.y);                                   \
                    mma_f8(acc[nt], WL[d].z, WH[d].z, WL[d].w, WH[d].w, xv.z, xv.w);                                   \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    mma_bf16(acc[nt], WL[d].x, WH[d].x, WL[d].y, WH[d].y, xv.x, xv.y);                                 \
                    mma_bf16(acc[nt], WL[d].z, WH[d].z, WL[d].w, WH[d].w, xv.z, xv.w);                                 \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if ((KT) + 2 < ntl)                                                                                            \
        {                                                                                                              \
            _Pragma("unroll") for (int d = 0; d < 3; ++d)                                                              \
            {                                                                                                          \
                WL[d] = ldg_w(wbase + next_wo + d * 512);                                                              \
                WH[d] = ldg_w(wbase + next_wo + d * 512 + hid);                                                        \
            }                                                                                                          \
        }                                                                                                              \
        next_wo += task_stride;                                                                                        \
        float* slots = red + ((KT) &1) * 8 * TS;                                                                       \
        _Pragma("unroll") for (int nt = 0; nt < NT; ++nt)                                                              \
        {                                                                                                              \
            const int col = nt * 8 + 2 * c;                                                                            \
            float* rp = slots + w * TS;                                                                                \
            if constexpr (GR::V)                                                                                       \
            {                                                                                                          \
                *(float2*) (rp + GR::off(g, col)) = make_float2(acc[nt][0], acc[nt][1]);                               \
                *(float2*) (rp + GR::off(g + 8, col)) = make_float2(acc[nt][2], acc[nt][3]);                           \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                rp[GR::off(g, col)] = acc[nt][0];                                                                      \
                rp[GR::off(g, col + 1)] = acc[nt][1];                                                                  \
                rp[GR::off(g + 8, col)] = acc[nt][2];                                                                  \
                rp[GR::off(g + 8, col + 1)] = acc[nt][3];                                                              \
            }                                                                                                          \
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;                                                   \
        }                                                                                                              \
        __syncthreads();                                                                                               \
        gemm_epilogue<NT, EPI, ALT_PART>(a, T, slots, q0 + (KT) *nbs, ss, part);                                       \
    }
        for (int kt = 0; kt < ntl; kt += 2)
        {
            XC3_TASK(wl, wh, kt)
            if (kt + 1 < ntl)
                XC3_TASK(wl2, wh2, kt + 1)
        }
#undef XC3_TASK
    }
    else
#elif defined(XC3_RECURRENCE)
    if constexpr (XC == 3)
    {
        long const task_stride = (long) nbs * rpm * a.ldw;
        long next_wo = (long) (q0 + nbs) * rpm * a.ldw;
        for (int kt = 0; kt < ntl; ++kt)
        {
#pragma unroll
            for (int d = 0; d < 3; ++d)
            {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                {
                    const uint4 xv = xr[d][nt];
                    if (F8)
                    {
                        mma_f8(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                        mma_f8(acc[nt], wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                    }
                    else
                    {
                        mma_bf16(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                        mma_bf16(acc[nt], wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                    }
                }
                if (kt + 1 < ntl)
                {
                    wl[d] = ldg_w(wbase + next_wo + d * 512);
                    wh[d] = ldg_w(wbase + next_wo + d * 512 + hid);
                }
            }
#ifdef XC3_TASK_PF_SMALL
            if constexpr (NT <= 6)
            {
                int const pk = kt + 2;
                if (pk < ntl && w < 2)
                {
                    long const po = (long) (q0 + pk * nbs) * rpm * a.ldw;
#pragma unroll
                    for (int pu = 0; pu < 3; ++pu)
                    {
                        uint8_t const* pp = a.W + po + (long) g * a.ldw + kslice + pu * 512 + c * 128 + (w ? hid : 0);
                        asm volatile("prefetch.global.L2 [%0];" ::"l"(pp));
                    }
                }
            }
#endif
            float* slots = red + (kt & 1) * 8 * TS;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                int const col = nt * 8 + 2 * c;
                float* rp = slots + w * TS;
#ifdef VEC_RED
                if constexpr (GR::V)
                {
                    *(float2*) (rp + GR::off(g, col)) = make_float2(acc[nt][0], acc[nt][1]);
                    *(float2*) (rp + GR::off(g + 8, col)) = make_float2(acc[nt][2], acc[nt][3]);
                }
                else
#endif
                {
                    rp[GR::off(g, col)] = acc[nt][0];
                    rp[GR::off(g, col + 1)] = acc[nt][1];
                    rp[GR::off(g + 8, col)] = acc[nt][2];
                    rp[GR::off(g + 8, col + 1)] = acc[nt][3];
                }
                acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
            }
            __syncthreads();
            gemm_epilogue<NT, EPI, ALT_PART>(a, T, slots, q0 + kt * nbs, ss, part);
            next_wo += task_stride;
        }
    }
    else
#endif
    {
#ifdef MMA_ILP
        constexpr int NA2 = NT < 3 ? NT : 1;
        float acc2[NA2][4];
#pragma unroll
        for (int nt = 0; nt < NA2; ++nt)
        {
            acc2[nt][0] = acc2[nt][1] = acc2[nt][2] = acc2[nt][3] = 0.f;
        }
#endif
        int ic = 0, kc = 0; // chunk index within the current task, current local task
        for (int u0 = 0; u0 < nu; u0 += D)
        {
#pragma unroll
            for (int d = 0; d < D; ++d)
            {
                int const u = u0 + d;
                if (u < nu)
                {
                    uint4 xn[NT];
                    if (XC == 0 && u + 1 < nu)
                    {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                            xn[nt] = ld_x(xbase + xoff[nt] + xko);
                        CUR_NEXT(xk, xi, xwo, xko);
                    }
#ifdef MMA_ILP
                    // Issue every n-tile's first K-half before the second half. For NT<3 use a second
                    // accumulator so no adjacent MMA has a read-after-write dependency.
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        const uint4 xv = XC == 0 ? xc[nt] : xr[XC == 1 ? 0 : d][nt];
                        if (F8)
                            mma_f8(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                        else
                            mma_bf16(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        const uint4 xv = XC == 0 ? xc[nt] : xr[XC == 1 ? 0 : d][nt];
                        float* ac = NT < 3 ? acc2[nt] : acc[nt];
                        if (F8)
                            mma_f8(ac, wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                        else
                            mma_bf16(ac, wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                    }
#else
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
                    {
                        const uint4 xv = XC == 0 ? xc[nt] : xr[XC == 1 ? 0 : d][nt]; // XC == 3 == D: unit u % 3 == d
                        if (F8)
                        {
                            mma_f8(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                            mma_f8(acc[nt], wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                        }
                        else
                        {
                            mma_bf16(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv.x, xv.y);
                            mma_bf16(acc[nt], wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv.z, xv.w);
                        }
                    }
#endif
                    if (u + D < nu)
                    {
                        wl[d] = ldg_w(wbase + lwo);
                        wh[d] = ldg_w(wbase + lwo + hid);
                        CUR_NEXT(lk, li, lwo, lko);
                    }
                    if (XC == 0)
                    {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                            xc[nt] = xn[nt];
                    }
                    if (++ic == cpw)
                    { // task boundary: deterministic cross-warp reduction + epilogue
                        float* slots = red + (kc & 1) * 8 * TS;
#ifdef MMA_ILP
                        if (NT < 3)
                        {
#pragma unroll
                            for (int nt = 0; nt < NA2; ++nt)
                            {
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    acc[nt][e] += acc2[nt][e];
                                    acc2[nt][e] = 0.f;
                                }
                            }
                        }
#endif
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                        {
                            int const col = nt * 8 + 2 * c;
                            float* rp = slots + w * TS;
#ifdef VEC_RED
                            if constexpr (GR::V)
                            {
                                *(float2*) (rp + GR::off(g, col)) = make_float2(acc[nt][0], acc[nt][1]);
                                *(float2*) (rp + GR::off(g + 8, col)) = make_float2(acc[nt][2], acc[nt][3]);
                            }
                            else
#endif
                            {
                                rp[GR::off(g, col)] = acc[nt][0];
                                rp[GR::off(g, col + 1)] = acc[nt][1];
                                rp[GR::off(g + 8, col)] = acc[nt][2];
                                rp[GR::off(g + 8, col + 1)] = acc[nt][3];
                            }
                            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
                        }
                        __syncthreads();
                        gemm_epilogue<NT, EPI, ALT_PART>(a, T, slots, q0 + kc * nbs, ss, part);
                        ic = 0;
                        ++kc;
                    }
                }
            }
        }
    }
#undef CUR_NEXT
#ifndef PRUNE_GEMM_TAIL_BARRIER
    __syncthreads();
#endif
}

// ---------------------------------------------------------------- attention (split-KV partial)
template <int HD>
__device__ __forceinline__ void rope_load(float2* cssn, int lane, int t)
{
    constexpr int NE = HD / 32;
    constexpr bool SLIDING = HD == 256;
    constexpr int NC = SLIDING ? NE / 2 : 2;
    float2 const* tab = g_rope + t * 192 + (SLIDING ? 0 : 128);
#pragma unroll
    for (int e = 0; e < NC; ++e)
    {
        int const d = lane + 32 * e;
        cssn[e] = __ldcg(tab + d);
    }
}

template <int HD>
__device__ __forceinline__ void rope_apply2(float* x, float2 const* cssn, int lane)
{
    constexpr int NE = HD / 32;
    constexpr int NC = HD == 256 ? NE / 2 : 2;
#pragma unroll
    for (int e = 0; e < NC; ++e)
    {
        float const cs = cssn[e].x, sn = cssn[e].y;
        float const lo = x[e], hi = x[e + NE / 2];
        x[e] = bfr(__fadd_rn(__fmul_rn(lo, cs), __fmul_rn(-hi, sn)));
        x[e + NE / 2] = bfr(__fadd_rn(__fmul_rn(hi, cs), __fmul_rn(lo, sn)));
    }
}

template <int HD>
__device__ __forceinline__ void rope_apply(float* x, int lane, int t, bool sliding)
{
    constexpr int NE = HD / 32;
    int const nrot = sliding ? HD / 2 : 64;
    float2 const* tab = g_rope + t * 192 + (sliding ? 0 : 128);
#pragma unroll
    for (int e = 0; e < NE / 2; ++e)
    {
        int const d = lane + 32 * e;
        if (d < nrot)
        {
            const float2 cssn = __ldcg(tab + d);
            float const cs = cssn.x, sn = cssn.y;
            float const lo = x[e], hi = x[e + NE / 2];
            x[e] = bfr(__fadd_rn(__fmul_rn(lo, cs), __fmul_rn(-hi, sn)));
            x[e + NE / 2] = bfr(__fadd_rn(__fmul_rn(hi, cs), __fmul_rn(lo, sn)));
        }
    }
}

__device__ __forceinline__ void ldsm_v8t(uint32_t& r0, uint32_t& r1, uint8_t const* p)
{
    // 16 key rows x 16 fp8 dimensions, transposed into two 8-column MMA fragments.
    unsigned const a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a)
                 : "memory");
}

template <int HD, int MTMAX, bool Q8>
struct AttnSmem
{
    static constexpr int KSTEP = MTMAX == 1 ? 32 : 64;
    static constexpr int QS = HD + 4; // f16 per q row
    static constexpr int QSB = Q8 ? HD + 8 : 2 * QS;
#ifdef VEC_SMEM_V
    static constexpr int VS = HD + 16; // bytes per V row, 16B aligned
#else
    static constexpr int VS = HD + 4; // bytes per V row
#endif
#ifdef VEC_SMEM_S
    static constexpr int SS = KSTEP + 2; // floats per score row, float2 aligned
#else
    static constexpr int SS = KSTEP + 1;
#endif
#ifdef ATTN_VDB
    // V tiles are double buffered (the next step's tile streams in during this step).  The 64-key layout
    // keeps P inside the bytes of its own score row to fit the second tile.
    static constexpr int NVB = 2;
    static constexpr bool PALIAS = MTMAX == 2;
#else
    static constexpr int NVB = 1;
    static constexpr bool PALIAS = false;
#endif
    static constexpr int PS = PALIAS ? 2 * SS : KSTEP + 8;
    static constexpr int OFF_V = 32 * QSB;
    static constexpr int OFF_S = OFF_V + NVB * KSTEP * VS;
    static constexpr int OFF_P = PALIAS ? OFF_S : OFF_S + 32 * SS * 4;
    static constexpr int OFF_A = PALIAS ? OFF_S + 32 * SS * 4 : OFF_P + 32 * PS * 2;
    static constexpr int OFF_NK = OFF_A + 32 * 4;
    static constexpr int OFF_NV = OFF_NK + 8 * HD;
#ifdef ATT_QK2_MT1
    static constexpr int OFF_S2 = OFF_NV + 8 * HD;
    static constexpr int BYTES = OFF_S2 + (MTMAX == 1 ? 32 * SS * 4 : 0);
#else
    static constexpr int BYTES = OFF_NV + 8 * HD;
#endif
};

// One item = (request, head group, key chunk).  Rows = Q x heads-in-group (<= 32).
template <int HD, int MTMAX, bool WRITE_KV, bool Q8>
__device__ __noinline__ bool attn_item(Params const& p, int item, int N, bf16 const* qn, bf16 const* kn, uint8_t* kc,
    uint8_t* vc, long long kvps, char* smem, unsigned int wtarget, unsigned int wneed, bool kvlast)
{
#ifdef KV_PERSIST
    const uint64_t kvpol = make_kvpol(kvlast);
#endif
    typedef AttnSmem<HD, MTMAX, Q8> SM;
    constexpr int KSTEP = SM::KSTEP, QS = SM::QS, VS = SM::VS, SS = SM::SS, PS = SM::PS;
    constexpr int NTW = HD / 64; // PV n-tiles per warp
    constexpr int NE = HD / 32;
    constexpr bool SLIDING = HD == 256;
    constexpr int NC = SLIDING ? NE / 2 : 2;
    constexpr int MT = MTMAX;
    constexpr int VCH = HD / 16;
    __half* qs = (__half*) smem;
    uint8_t* q8s = (uint8_t*) smem;
    uint8_t* Vs = (uint8_t*) (smem + SM::OFF_V);
    float* Ss = (float*) (smem + SM::OFF_S);
    __half* Ps = (__half*) (smem + SM::OFF_P);
    float* As = (float*) (smem + SM::OFF_A);
    uint8_t* NK = (uint8_t*) (smem + SM::OFF_NK);
    uint8_t* NV = (uint8_t*) (smem + SM::OFF_NV);
#ifdef ATT_QK2_MT1
    float* Ss2 = MTMAX == 1 ? (float*) (smem + SM::OFF_S2) : nullptr;
#endif
#ifdef ATT_PV_F8
    uint8_t* P8 = (uint8_t*) (smem + SM::OFF_P);
#ifndef ATT_PV_F8_SCALE
#define ATT_PV_F8_SCALE 16.0f
#endif
#endif

    int const tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
    int const Q = p.Q, L = p.L, NG = p.ng, NIc = p.nic;
    int const HG = 8 / NG;
    int const jj = item % NIc, rg = item / NIc;
    int const hg = rg % NG, r = rg / NG;
    int const cl = p.cache_lens[r];
    int const kend = cl + Q;
    int const kstart = SLIDING ? max(0, cl - 511) : 0;
    int const nk = kend - kstart;
    int const csz = (((nk + NIc - 1) / NIc) + KSTEP - 1) & ~(KSTEP - 1);
    int const ke = kend - jj * csz;
    if (ke <= kstart)
        return false;
    int const kb = max(kstart, ke - csz);
    int const rows = Q * HG;
    int const* rpt = kv_rpt(p, SLIDING, r);
    (void) L;
#define KVO(kk_) kv_off(rpt, kvps, (kk_), HD)
    bool const use_new = WRITE_KV && jj == 0;
#ifdef ATTN_PREBAR_V
    // Old step-0 V rows are independent of the preceding QKV phase.  Start their
    // async transfer before waiting for that phase, then patch new/padded rows
    // after the producer has completed.
    for (int i = tid; i < KSTEP * VCH; i += NTHR)
    {
        int const key = i / VCH, col = (i - key * VCH) * 16;
        int const kk = kb + key;
        if (kk < ke && !(use_new && kk >= cl))
            VCPA(Vs + key * VS + col, vc + KVO(kk) + col);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    if (wneed)
        grid_wait(wtarget, wneed);
    ATT_STAMP(s_curlayer, 0);
#ifndef NO_ATTN_PF
    { // L2 prefetch of the first chunk's K/V rows; overlaps q/k/v preparation
        constexpr int LPR = HD / 128;
#ifdef ATTN_PF2
        constexpr int PFK = 2 * KSTEP;
#else
        constexpr int PFK = KSTEP;
#endif
        for (int li = tid; li < 2 * PFK * LPR; li += NTHR)
        {
            int const isv = li / (PFK * LPR), rem = li - isv * PFK * LPR;
            int const key = rem / LPR, ln = rem - key * LPR;
            int const kk = kb + key;
            if (kk < ke && !(use_new && kk >= cl))
            {
                uint8_t const* pa = (isv ? vc : kc) + KVO(kk) + ln * 128;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(pa));
            }
        }
    }
#endif

#ifdef BATCH_PREP
    // ---- q rows: rmsnorm(weight) + rope -> smem; jj == 0 items also build the new K/V rows (head group 0
    //      writes them to the cache).  Each warp walks its row slots with the next slot's loads in flight.
    {
        constexpr int QPW = MT * 2;             // q-row slots per warp
        int const ns = QPW + (use_new ? 2 : 0); // plus new K/V row slots (2Q <= 16)
#pragma unroll
        for (int sl = 0; sl < QPW; ++sl)
        { // zero padded q rows
            int const row = w + 8 * sl;
            if (row >= rows)
            {
#pragma unroll
                for (int e = 0; e < NE; ++e)
                {
                    if constexpr (Q8)
                        q8s[row * SM::QSB + lane + 32 * e] = 0;
                    else
                        ((__half*) (smem + row * SM::QSB))[lane + 32 * e] = __float2half_rn(0.f);
                }
            }
        }
        float xa[NE], xb[NE];
        float2 ca[NC], cb[NC];
#define PREP_LOAD(SL, X, CS)                                                                                           \
    {                                                                                                                  \
        const bool isq_ = (SL) < QPW;                                                                                  \
        const int row_ = w + 8 * (isq_ ? (SL) : (SL) -QPW);                                                            \
        if (isq_ ? row_ < rows : row_ < 2 * Q)                                                                         \
        {                                                                                                              \
            if (isq_ && !WRITE_KV)                                                                                     \
            {                                                                                                          \
                const int tl_ = row_ / HG, h_ = hg * HG + (row_ - tl_ * HG);                                           \
                const int t_ = r * Q + tl_;                                                                            \
                const float* q0_ = g_qpart + (size_t) t_ * 4096 + h_ * HD;                                             \
                const float* q1_ = q0_ + (size_t) TMAX * 4096;                                                         \
                _Pragma("unroll") for (int e = 0; e < NE; ++e) X[e]                                                    \
                    = bfr(__ldcg(q0_ + lane + 32 * e) + __ldcg(q1_ + lane + 32 * e));                                  \
                rope_load<HD>(CS, lane, t_);                                                                           \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                int t_, col_;                                                                                          \
                bool rope_;                                                                                            \
                if (isq_)                                                                                              \
                {                                                                                                      \
                    const int tl_ = row_ / HG;                                                                         \
                    t_ = r * Q + tl_;                                                                                  \
                    col_ = (hg * HG + (row_ - tl_ * HG)) * HD;                                                         \
                    rope_ = true;                                                                                      \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    t_ = r * Q + (row_ >> 1);                                                                          \
                    col_ = 8 * HD + ((row_ & 1) ? HD : 0);                                                             \
                    rope_ = !(row_ & 1);                                                                               \
                }                                                                                                      \
                const bf16* src_ = g_qkv + (size_t) t_ * N + col_;                                                     \
                _Pragma("unroll") for (int e = 0; e < NE; ++e) X[e] = bf2f(__ldcg(src_ + lane + 32 * e));              \
                if (rope_)                                                                                             \
                    rope_load<HD>(CS, lane, t_);                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }
        PREP_LOAD(0, xa, ca)
#pragma unroll 1
        for (int sl = 0; sl < ns; ++sl)
        {
            if (sl + 1 < ns)
                PREP_LOAD(sl + 1, xb, cb)
            bool const isq = sl < QPW;
            int const row = w + 8 * (isq ? sl : sl - QPW);
            if (isq ? row < rows : row < 2 * Q)
            {
                float ss = 0.f;
#pragma unroll
                for (int e = 0; e < NE; ++e)
                    ss += xa[e] * xa[e];
                ss = warp_sum(ss);
                float const rs = rsqrtf(ss / (float) HD + EPSV);
                bool const isv = !isq && (row & 1);
                bf16 const* nwp = isq ? qn : kn;
                if (!isv)
                {
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                        xa[e] = bfr((xa[e] * rs) * bf2f(nwp[lane + 32 * e]));
                    rope_apply2<HD>(xa, ca, lane);
                }
                else
                {
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                        xa[e] = bfr(xa[e] * rs);
                }
                if (isq)
                {
                    __half* qd = (__half*) (smem + row * SM::QSB);
                    uint8_t* q8d = q8s + row * SM::QSB;
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                    {
                        if constexpr (Q8)
                            q8d[lane + 32 * e] = fp8q(xa[e]);
                        else
                            qd[lane + 32 * e] = __float2half_rn(xa[e]);
                    }
                }
                else if constexpr (WRITE_KV)
                {
                    int const tl = row >> 1;
                    uint8_t* sdst = (isv ? NV : NK) + tl * HD;
                    uint8_t* gdst = (isv ? vc : kc) + KVO(cl + tl);
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                    {
                        const uint8_t b = (uint8_t) __nv_cvt_float_to_fp8(xa[e], __NV_SATFINITE, __NV_E4M3);
                        sdst[lane + 32 * e] = b;
                        if (hg == 0)
                            gdst[lane + 32 * e] = b;
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < NE; ++e)
                xa[e] = xb[e];
#pragma unroll
            for (int e = 0; e < NC; ++e)
                ca[e] = cb[e];
        }
#undef PREP_LOAD
    }
#else
    // ---- q: rmsnorm(weight) + rope -> smem f16
    for (int row = w; row < MT * 16; row += 8)
    {
        __half* qd = (__half*) (smem + row * SM::QSB);
        uint8_t* q8d = q8s + row * SM::QSB;
        if (row >= rows)
        {
#pragma unroll
            for (int e = 0; e < NE; ++e)
            {
                if constexpr (Q8)
                    q8d[lane + 32 * e] = 0;
                else
                    qd[lane + 32 * e] = __float2half_rn(0.f);
            }
            continue;
        }
        int const tl = row / HG, h = hg * HG + (row - tl * HG);
        int const t = r * Q + tl;
        float x[NE], wv[NE];
        float2 cssn[NC];
        float ss = 0.f;
        if constexpr (WRITE_KV)
        {
            bf16 const* src = g_qkv + (size_t) t * N + h * HD;
#pragma unroll
            for (int e = 0; e < NE; ++e)
            {
                x[e] = bf2f(__ldcg(src + lane + 32 * e));
                wv[e] = bf2f(qn[lane + 32 * e]);
            }
        }
        else
        { // KV-shared layers: BF16 q projection = sum of two split-K fp32 partials, rounded to bf16
            float const* q0 = g_qpart + (size_t) t * 4096 + h * HD;
            float const* q1 = q0 + (size_t) TMAX * 4096;
#pragma unroll
            for (int e = 0; e < NE; ++e)
            {
                x[e] = bfr(__ldcg(q0 + lane + 32 * e) + __ldcg(q1 + lane + 32 * e));
                wv[e] = bf2f(qn[lane + 32 * e]);
            }
        }
        rope_load<HD>(cssn, lane, t);
#pragma unroll
        for (int e = 0; e < NE; ++e)
            ss += x[e] * x[e];
        ss = warp_sum(ss);
        float const rs = rsqrtf(ss / (float) HD + EPSV);
#pragma unroll
        for (int e = 0; e < NE; ++e)
            x[e] = bfr((x[e] * rs) * wv[e]);
        rope_apply2<HD>(x, cssn, lane);
#pragma unroll
        for (int e = 0; e < NE; ++e)
        {
            if constexpr (Q8)
                q8d[lane + 32 * e] = fp8q(x[e]);
            else
                qd[lane + 32 * e] = __float2half_rn(x[e]);
        }
    }
    // ---- new K/V rows: computed by every jj==0 item into smem; head group 0 also writes the cache
    if constexpr (WRITE_KV)
        if (use_new)
        {
            // Continue the warp round-robin after the q rows so uneven row counts do
            // not leave one warp with both an extra q row and an extra K/V row.
            for (int rr = (w - rows) & 7; rr < 2 * Q; rr += 8)
            {
                int const tl = rr >> 1;
                bool const isv = rr & 1;
                int const t = r * Q + tl;
                bf16 const* src = g_qkv + (size_t) t * N + 8 * HD + (isv ? HD : 0);
                float x[NE], wv[NE];
                float2 cssn[NC];
                float ss = 0.f;
#pragma unroll
                for (int e = 0; e < NE; ++e)
                {
                    x[e] = bf2f(__ldcg(src + lane + 32 * e));
                    wv[e] = isv ? 1.f : bf2f(kn[lane + 32 * e]);
                }
                if (!isv)
                    rope_load<HD>(cssn, lane, t);
#pragma unroll
                for (int e = 0; e < NE; ++e)
                    ss += x[e] * x[e];
                ss = warp_sum(ss);
                float const rs = rsqrtf(ss / (float) HD + EPSV);
                if (!isv)
                {
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                        x[e] = bfr((x[e] * rs) * wv[e]);
                    rope_apply2<HD>(x, cssn, lane);
                }
                else
                {
#pragma unroll
                    for (int e = 0; e < NE; ++e)
                        x[e] = bfr(x[e] * rs);
                }
                uint8_t* sdst = (isv ? NV : NK) + tl * HD;
                uint8_t* gdst = (isv ? vc : kc) + KVO(cl + tl);
#pragma unroll
                for (int e = 0; e < NE; ++e)
                {
                    const uint8_t b = (uint8_t) __nv_cvt_float_to_fp8(x[e], __NV_SATFINITE, __NV_E4M3);
                    sdst[lane + 32 * e] = b;
                    if (hg == 0)
                        gdst[lane + 32 * e] = b;
                }
            }
        }
#endif
    __syncthreads();
    ATT_STAMP(s_curlayer, 1);
#ifdef ATTN_PREBAR_V
    // cp.async owns only old rows; fill the disjoint new and out-of-range rows.
    for (int i = tid; i < KSTEP * VCH; i += NTHR)
    {
        int const key = i / VCH, col = (i - key * VCH) * 16;
        int const kk = kb + key;
        if (!(kk < ke && !(use_new && kk >= cl)))
        {
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (kk < ke)
                v = *(uint4 const*) (NV + (kk - cl) * HD + col);
            *(uint4*) (Vs + key * VS + col) = v;
        }
    }
#endif

#ifdef ATT_QK2_MT1
    // The one-M schedule uses all eight warps: warp pairs own the two HD halves
    // for the same eight keys.  Keep the next K fragments live while softmax/PV
    // consume the current score tile.
    constexpr int QK2_NDB = HD / 128;
    int const qk2_nt = w & 3, qk2_dh = w >> 2;
    uint4 qk2_kv[QK2_NDB];
    auto qk2_load = [&](int k0x)
    {
        const int kk = min(k0x + qk2_nt * 8 + g, ke - 1);
        const bool fromsm = use_new && kk >= cl;
        const uint8_t* kp = (fromsm ? NK + (kk - cl) * HD : kc + KVO(kk)) + qk2_dh * (HD / 2) + 16 * c;
#pragma unroll
        for (int db = 0; db < QK2_NDB; ++db)
            qk2_kv[db] = fromsm ? *(const uint4*) (kp + db * 64) : KLDG((const uint4*) (kp + db * 64));
    };
    if constexpr (MTMAX == 1)
        qk2_load(kb);
#endif

    float m_run = -1e30f, l_run = 0.f;
    float oacc[MTMAX][NTW][4];
#pragma unroll
    for (int mt = 0; mt < MTMAX; ++mt)
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt)
        {
            oacc[mt][nt][0] = oacc[mt][nt][1] = oacc[mt][nt][2] = oacc[mt][nt][3] = 0.f;
        }
    int const sr = tid >> 3, sub = tid & 7;
    int const qpos_sr = cl + sr / HG;
    bool const srvalid = sr < rows;

#ifdef ATTN_VDB
    uint4 kvr[MTMAX == 2 ? HD / 64 : 1];
    auto mt2_kload = [&](int k0x)
    {
        const int kk = min(k0x + w * 8 + g, ke - 1);
        const bool fromsm = use_new && kk >= cl;
        const uint8_t* kp = fromsm ? NK + (kk - cl) * HD + 16 * c : kc + KVO(kk) + 16 * c;
#pragma unroll
        for (int db = 0; db < HD / 64; ++db)
            kvr[MTMAX == 2 ? db : 0] = fromsm ? *(const uint4*) (kp + db * 64) : KLDG((const uint4*) (kp + db * 64));
    };
    if constexpr (MTMAX == 2)
        mt2_kload(kb);
    int st = 0;
#ifdef ATTN_TIMING
    long long cy0 = 0, cy1 = 0, cy2 = 0, cy3 = 0, ct = clock64();
#define CYACC(v)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        const long long t2_ = clock64();                                                                               \
        v += t2_ - ct;                                                                                                 \
        ct = t2_;                                                                                                      \
    } while (0)
#else
#define CYACC(v)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif
    for (int k0 = kb; k0 < ke; k0 += KSTEP, ++st)
    {
        uint8_t* Vc = Vs + (st & 1) * KSTEP * VS;
        { // stage the next step's V tile into the other buffer (last read by the previous step's PV)
            int const k1 = k0 + KSTEP;
            uint8_t* Vn = Vs + ((st + 1) & 1) * KSTEP * VS;
            if (k1 < ke)
            {
                for (int i = tid; i < KSTEP * VCH; i += NTHR)
                {
                    int const key = i / VCH, col = (i - key * VCH) * 16;
                    int const kk = k1 + key;
                    uint8_t* vd = Vn + key * VS + col;
                    if (kk < ke && !(use_new && kk >= cl))
                    {
                        VCPA(vd, vc + KVO(kk) + col);
                    }
                    else
                    {
                        uint4 v = make_uint4(0u, 0u, 0u, 0u);
                        if (kk < ke)
                            v = *(uint4 const*) (NV + (kk - cl) * HD + col);
                        *(uint4*) vd = v;
                    }
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        if constexpr (MTMAX == 1)
        {
            if constexpr (Q8)
            {
                uint8_t const* qa = q8s + g * SM::QSB + 16 * c + qk2_dh * (HD / 2);
                uint8_t const* qb = qa + 8 * SM::QSB;
                float acc[2][4];
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2)
                    acc[s2][0] = acc[s2][1] = acc[s2][2] = acc[s2][3] = 0.f;
#pragma unroll
                for (int db = 0; db < QK2_NDB; ++db)
                {
                    const uint32_t kw[4] = {qk2_kv[db].x, qk2_kv[db].y, qk2_kv[db].z, qk2_kv[db].w};
#pragma unroll
                    for (int s2 = 0; s2 < 2; ++s2)
                    {
                        const uint2 A0 = *(uint2 const*) (qa + db * 64 + 8 * s2);
                        const uint2 A1 = *(uint2 const*) (qb + db * 64 + 8 * s2);
                        mma_f8(acc[s2], A0.x, A1.x, A0.y, A1.y, kw[2 * s2], kw[2 * s2 + 1]);
                    }
                }
                if (k0 + KSTEP < ke)
                    qk2_load(k0 + KSTEP);
                float* sd = (qk2_dh ? Ss2 : Ss) + g * SS + qk2_nt * 8 + 2 * c;
                sd[0] = acc[0][0] + acc[1][0];
                sd[1] = acc[0][1] + acc[1][1];
                sd[8 * SS] = acc[0][2] + acc[1][2];
                sd[8 * SS + 1] = acc[0][3] + acc[1][3];
            }
            else
            {
                __half const* qa = qs + g * QS + 16 * c + qk2_dh * (HD / 2);
                __half const* qb = qa + 8 * QS;
                float acc[4][4];
#pragma unroll
                for (int s2 = 0; s2 < 4; ++s2)
                    acc[s2][0] = acc[s2][1] = acc[s2][2] = acc[s2][3] = 0.f;
#pragma unroll
                for (int db = 0; db < QK2_NDB; ++db)
                {
                    const uint32_t kw[4] = {qk2_kv[db].x, qk2_kv[db].y, qk2_kv[db].z, qk2_kv[db].w};
#pragma unroll
                    for (int s2 = 0; s2 < 4; ++s2)
                    {
                        const uint2 A0 = *(uint2 const*) (qa + db * 64 + 4 * s2);
                        const uint2 A1 = *(uint2 const*) (qb + db * 64 + 4 * s2);
                        mma_f16(acc[s2], A0.x, A1.x, A0.y, A1.y, cvt_e4m3x2(kw[s2]), cvt_e4m3x2(kw[s2] >> 16));
                    }
                }
                if (k0 + KSTEP < ke)
                    qk2_load(k0 + KSTEP);
                float* sd = (qk2_dh ? Ss2 : Ss) + g * SS + qk2_nt * 8 + 2 * c;
                sd[0] = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
                sd[1] = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
                sd[8 * SS] = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
                sd[8 * SS + 1] = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
            }
        }
        else
        {
            static_assert(MTMAX == 1 || Q8, "two-M attention uses e4m3 q");
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int db = 0; db < HD / 64; ++db)
            {
                const uint32_t kw[4] = {kvr[db].x, kvr[db].y, kvr[db].z, kvr[db].w};
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2)
                {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        uint8_t const* qa = q8s + (mt * 16 + g) * SM::QSB + 16 * c + db * 64 + 8 * s2;
                        const uint2 A0 = *(uint2 const*) qa;
                        const uint2 A1 = *(uint2 const*) (qa + 8 * SM::QSB);
                        mma_f8(acc[mt], A0.x, A1.x, A0.y, A1.y, kw[2 * s2], kw[2 * s2 + 1]);
                    }
                }
            }
            if (k0 + KSTEP < ke)
                mt2_kload(k0 + KSTEP); // next K overlaps softmax and PV
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                float* sd = Ss + (mt * 16 + g) * SS + w * 8 + 2 * c;
                sd[0] = acc[mt][0];
                sd[1] = acc[mt][1];
                sd[8 * SS] = acc[mt][2];
                sd[8 * SS + 1] = acc[mt][3];
            }
        }
        CYACC(cy0);
        asm volatile("cp.async.wait_group 1;" ::: "memory"); // this step's V tile has landed
        __syncthreads();
        CYACC(cy1);
        // online softmax
        if (sr < MT * 16)
        {
            constexpr int KPT = KSTEP / 8;
            float sv[KPT];
            bool vl[KPT];
            float mx = -INFINITY;
#pragma unroll
            for (int q = 0; q < KPT; ++q)
            {
                int const key = k0 + sub * KPT + q;
                bool const v = srvalid && key < ke && key <= qpos_sr && (!SLIDING || qpos_sr - key < 512);
                sv[q] = MTMAX == 1 ? Ss[sr * SS + sub * KPT + q] + Ss2[sr * SS + sub * KPT + q]
                                   : Ss[sr * SS + sub * KPT + q];
                vl[q] = v;
                if (v)
                    mx = fmaxf(mx, sv[q]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            float const m_new = fmaxf(m_run, mx);
            float const alpha = expf(m_run - m_new);
            if constexpr (SM::PALIAS)
                __syncwarp(); // P overwrites this row's scores
            float sum = 0.f;
#pragma unroll
            for (int q = 0; q < KPT; ++q)
            {
                float const pr = vl[q] ? expf(sv[q] - m_new) : 0.f;
                sum += pr;
                int const col = sub * KPT + q, kp = col & 15;
                int const lg = (col & ~15) + 2 * (kp >> 2) + (kp & 1) + ((kp & 2) ? 8 : 0);
                Ps[sr * PS + lg] = __float2half_rn(pr);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if (sub == 0)
                As[sr] = alpha;
        }
        __syncthreads();
        CYACC(cy2);
        // O += P V
#pragma unroll
        for (int mt = 0; mt < MTMAX; ++mt)
        {
            if (mt < MT)
            {
                float const a0 = As[mt * 16 + g], a1 = As[mt * 16 + 8 + g];
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
                {
                    oacc[mt][nt][0] *= a0;
                    oacc[mt][nt][1] *= a0;
                    oacc[mt][nt][2] *= a1;
                    oacc[mt][nt][3] *= a1;
                }
            }
        }
#pragma unroll
        for (int ks = 0; ks < KSTEP / 16; ++ks)
        {
            if constexpr (MTMAX == 1)
            {
                __half const* pa = Ps + g * PS + ks * 16 + 2 * c;
                const uint32_t a0 = *(uint32_t const*) pa;
                const uint32_t a1 = *(uint32_t const*) (pa + 8 * PS);
                const uint32_t a2 = *(uint32_t const*) (pa + 8);
                const uint32_t a3 = *(uint32_t const*) (pa + 8 * PS + 8);
#pragma unroll
                for (int np = 0; np < NTW / 2; ++np)
                {
                    uint32_t r0, r1;
                    ldsm_v8t(r0, r1, Vc + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                    mma_f16(oacc[0][2 * np], a0, a1, a2, a3, cvt_e4m3x2(r0), cvt_e4m3x2(r0 >> 16));
                    mma_f16(oacc[0][2 * np + 1], a0, a1, a2, a3, cvt_e4m3x2(r1), cvt_e4m3x2(r1 >> 16));
                }
            }
            else
            {
                uint32_t pf[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                {
                    __half const* pa = Ps + (mt * 16 + g) * PS + ks * 16 + 2 * c;
                    pf[mt][0] = *(uint32_t const*) pa;
                    pf[mt][1] = *(uint32_t const*) (pa + 8 * PS);
                    pf[mt][2] = *(uint32_t const*) (pa + 8);
                    pf[mt][3] = *(uint32_t const*) (pa + 8 * PS + 8);
                }
#pragma unroll
                for (int np = 0; np < NTW / 2; ++np)
                {
                    uint32_t r0, r1;
                    ldsm_v8t(r0, r1, Vc + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                    const uint32_t b00 = cvt_e4m3x2(r0), b01 = cvt_e4m3x2(r0 >> 16);
                    const uint32_t b10 = cvt_e4m3x2(r1), b11 = cvt_e4m3x2(r1 >> 16);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        mma_f16(oacc[mt][2 * np], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b00, b01);
                        mma_f16(oacc[mt][2 * np + 1], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b10, b11);
                    }
                }
            }
        }
        __syncthreads();
        CYACC(cy3);
    }
#ifdef ATTN_TIMING
    if (tid == 0)
    {
        unsigned long long* a = g_acyc + ((size_t) s_curlayer * 192 + blockIdx.x) * 4;
        a[0] = cy0;
        a[1] = cy1;
        a[2] = cy2;
        a[3] = cy3;
    }
#endif
#undef CYACC
#else
    for (int k0 = kb; k0 < ke; k0 += KSTEP)
    {
#ifndef NO_ATTN_PF
        // L2 prefetch of the next step's K/V rows (one 128-byte line per thread)
        {
            constexpr int LPR = HD / 128; // lines per key row
            for (int li = tid; li < 2 * KSTEP * LPR; li += NTHR)
            {
                int const isv = li / (KSTEP * LPR), rem = li - isv * KSTEP * LPR;
                int const key = rem / LPR, ln = rem - key * LPR;
#ifdef ATTN_PF2
                int const kk = k0 + 2 * KSTEP + key; // two steps ahead (the prologue covered the first two)
#else
                int const kk = k0 + KSTEP + key;
#endif
                if (kk < ke && !(use_new && kk >= cl))
                {
                    uint8_t const* pa = (isv ? vc : kc) + KVO(kk) + ln * 128;
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(pa));
                }
            }
        }
#endif
#ifdef ATTN_K_FIRST_MT2
        // For the 64-key/two-M path, issue the QK K-fragment loads before the
        // independent V staging so their long-scoreboard latency overlaps cp.async.
        uint4 kv_early[HD / 64];
        if constexpr (MTMAX == 2)
        {
            int const kk = min(k0 + w * 8 + g, ke - 1);
            bool const fromsm = use_new && kk >= cl;
            uint8_t const* kp = fromsm ? NK + (kk - cl) * HD + 16 * c : kc + KVO(kk) + 16 * c;
#pragma unroll
            for (int db = 0; db < HD / 64; ++db)
                kv_early[db] = fromsm ? *(uint4 const*) (kp + db * 64) : KLDG((uint4 const*) (kp + db * 64));
        }
#endif
        // V chunk -> smem (fp8, padded rows).  On the async path, old-cache rows
        // transfer while the independent QK MMA below is in flight; new rows are
        // already in shared memory and retain the direct shared-to-shared copy.
#ifdef ATTN_PREBAR_V
        if (k0 != kb)
        {
#endif
            for (int i = tid; i < KSTEP * VCH; i += NTHR)
            {
                int const key = i / VCH, col = (i - key * VCH) * 16;
                int const kk = k0 + key;
#ifdef ATTN_CPASYNC_V
                uint8_t* vd = Vs + key * VS + col;
                if (kk < ke && !(use_new && kk >= cl))
                {
                    VCPA(vd, vc + KVO(kk) + col);
                }
                else
                {
                    uint4 v = make_uint4(0u, 0u, 0u, 0u);
                    if (kk < ke)
                        v = *(uint4 const*) (NV + (kk - cl) * HD + col);
                    *(uint4*) vd = v;
                }
#else
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (kk < ke)
            {
                if (use_new && kk >= cl)
                    v = *(uint4 const*) (NV + (kk - cl) * HD + col);
                else
                    v = __ldcg((uint4 const*) (vc + KVO(kk) + col));
            }
#ifdef VEC_SMEM_V
            *(uint4*) (Vs + key * VS + col) = v;
#else
            uint32_t* d = (uint32_t*) (Vs + key * VS + col);
            d[0] = v.x;
            d[1] = v.y;
            d[2] = v.z;
            d[3] = v.w;
#endif
#endif
            }
#ifdef ATTN_CPASYNC_V
            asm volatile("cp.async.commit_group;" ::: "memory");
#endif
#ifdef ATTN_PREBAR_V
        }
#endif
        // S = q K^T. The one-M path keeps its compact 32-key schedule; the two-M path
        // lets each warp own eight keys and reuses the K fragments across both M tiles.
        if constexpr (MTMAX == 1)
        {
#ifdef ATT_QK2_MT1
            if constexpr (Q8)
            {
                uint8_t const* qa = q8s + g * SM::QSB + 16 * c + qk2_dh * (HD / 2);
                uint8_t const* qb = qa + 8 * SM::QSB;
                float acc[2][4];
#pragma unroll
                for (int s = 0; s < 2; ++s)
                    acc[s][0] = acc[s][1] = acc[s][2] = acc[s][3] = 0.f;
#pragma unroll
                for (int db = 0; db < QK2_NDB; ++db)
                {
                    const uint32_t kw[4] = {qk2_kv[db].x, qk2_kv[db].y, qk2_kv[db].z, qk2_kv[db].w};
#pragma unroll
                    for (int s = 0; s < 2; ++s)
                    {
                        const uint2 A0 = *(uint2 const*) (qa + db * 64 + 8 * s);
                        const uint2 A1 = *(uint2 const*) (qb + db * 64 + 8 * s);
                        mma_f8(acc[s], A0.x, A1.x, A0.y, A1.y, kw[2 * s], kw[2 * s + 1]);
                    }
                }
                if (k0 + KSTEP < ke)
                    qk2_load(k0 + KSTEP);
                float* sd = (qk2_dh ? Ss2 : Ss) + g * SS + qk2_nt * 8 + 2 * c;
                sd[0] = acc[0][0] + acc[1][0];
                sd[1] = acc[0][1] + acc[1][1];
                sd[8 * SS] = acc[0][2] + acc[1][2];
                sd[8 * SS + 1] = acc[0][3] + acc[1][3];
            }
            else
            {
                __half const* qa = qs + g * QS + 16 * c + qk2_dh * (HD / 2);
                __half const* qb = qa + 8 * QS;
                float acc[4][4];
#pragma unroll
                for (int s = 0; s < 4; ++s)
                    acc[s][0] = acc[s][1] = acc[s][2] = acc[s][3] = 0.f;
#pragma unroll
                for (int db = 0; db < QK2_NDB; ++db)
                {
                    const uint32_t kw[4] = {qk2_kv[db].x, qk2_kv[db].y, qk2_kv[db].z, qk2_kv[db].w};
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                    {
                        const uint2 A0 = *(uint2 const*) (qa + db * 64 + 4 * s);
                        const uint2 A1 = *(uint2 const*) (qb + db * 64 + 4 * s);
                        mma_f16(acc[s], A0.x, A1.x, A0.y, A1.y, cvt_e4m3x2(kw[s]), cvt_e4m3x2(kw[s] >> 16));
                    }
                }
                if (k0 + KSTEP < ke)
                    qk2_load(k0 + KSTEP);
                float* sd = (qk2_dh ? Ss2 : Ss) + g * SS + qk2_nt * 8 + 2 * c;
                sd[0] = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
                sd[1] = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
                sd[8 * SS] = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
                sd[8 * SS + 1] = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
            }
#else
            if (w < 4)
            {
                int const nt = w;
                int const kk = min(k0 + nt * 8 + g, ke - 1);
                bool const fromsm = use_new && kk >= cl;
                uint8_t const* kp = fromsm ? NK + (kk - cl) * HD + 16 * c : kc + KVO(kk) + 16 * c;
                uint4 kv[HD / 64];
#pragma unroll
                for (int db = 0; db < HD / 64; ++db)
                    kv[db] = fromsm ? *(uint4 const*) (kp + db * 64) : KLDG((uint4 const*) (kp + db * 64));
                __half const* qa = qs + g * QS + 16 * c;
                __half const* qb = qa + 8 * QS;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int db = 0; db < HD / 64; ++db)
                {
                    const uint32_t kw[4] = {kv[db].x, kv[db].y, kv[db].z, kv[db].w};
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                    {
                        const uint2 A0 = *(uint2 const*) (qa + db * 64 + 4 * s);
                        const uint2 A1 = *(uint2 const*) (qb + db * 64 + 4 * s);
                        mma_f16(acc, A0.x, A1.x, A0.y, A1.y, cvt_e4m3x2(kw[s]), cvt_e4m3x2(kw[s] >> 16));
                    }
                }
                float* sd = Ss + g * SS + nt * 8 + 2 * c;
                sd[0] = acc[0];
                sd[1] = acc[1];
                sd[8 * SS] = acc[2];
                sd[8 * SS + 1] = acc[3];
            }
#endif
        }
        else
        {
#ifdef ATTN_K_FIRST_MT2
            const uint4(&kv)[HD / 64] = kv_early;
#else
            int const kk = min(k0 + w * 8 + g, ke - 1);
            bool const fromsm = use_new && kk >= cl;
            uint8_t const* kp = fromsm ? NK + (kk - cl) * HD + 16 * c : kc + KVO(kk) + 16 * c;
            uint4 kv[HD / 64];
#pragma unroll
            for (int db = 0; db < HD / 64; ++db)
                kv[db] = fromsm ? *(uint4 const*) (kp + db * 64) : KLDG((uint4 const*) (kp + db * 64));
#endif
            float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            if constexpr (Q8)
            {
#pragma unroll
                for (int db = 0; db < HD / 64; ++db)
                {
                    const uint32_t kw[4] = {kv[db].x, kv[db].y, kv[db].z, kv[db].w};
#pragma unroll
                    for (int s = 0; s < 2; ++s)
                    {
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt)
                        {
                            uint8_t const* qa = q8s + (mt * 16 + g) * SM::QSB + 16 * c + db * 64 + 8 * s;
                            const uint2 A0 = *(uint2 const*) qa;
                            const uint2 A1 = *(uint2 const*) (qa + 8 * SM::QSB);
                            mma_f8(acc[mt], A0.x, A1.x, A0.y, A1.y, kw[2 * s], kw[2 * s + 1]);
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (int db = 0; db < HD / 64; ++db)
                {
                    const uint32_t kw[4] = {kv[db].x, kv[db].y, kv[db].z, kv[db].w};
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                    {
                        const uint32_t b0 = cvt_e4m3x2(kw[s]), b1 = cvt_e4m3x2(kw[s] >> 16);
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt)
                        {
                            __half const* qa = qs + (mt * 16 + g) * QS + 16 * c + db * 64 + 4 * s;
                            const uint2 A0 = *(uint2 const*) qa;
                            const uint2 A1 = *(uint2 const*) (qa + 8 * QS);
                            mma_f16(acc[mt], A0.x, A1.x, A0.y, A1.y, b0, b1);
                        }
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                float* sd = Ss + (mt * 16 + g) * SS + w * 8 + 2 * c;
                sd[0] = acc[mt][0];
                sd[1] = acc[mt][1];
                sd[8 * SS] = acc[mt][2];
                sd[8 * SS + 1] = acc[mt][3];
            }
        }
#ifdef ATTN_CPASYNC_V
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();
        // online softmax
        if (sr < MT * 16)
        {
            constexpr int KPT = KSTEP / 8;
            float sv[KPT];
            bool vl[KPT];
            float mx = -INFINITY;
#pragma unroll
            for (int q = 0; q < KPT; ++q)
            {
                int const key = k0 + sub * KPT + q;
                bool const v = srvalid && key < ke && key <= qpos_sr && (!SLIDING || qpos_sr - key < 512);
#ifdef ATT_QK2_MT1
                sv[q] = MTMAX == 1 ? Ss[sr * SS + sub * KPT + q] + Ss2[sr * SS + sub * KPT + q]
                                   : Ss[sr * SS + sub * KPT + q];
#else
                sv[q] = Ss[sr * SS + sub * KPT + q];
#endif
                vl[q] = v;
                if (v)
                    mx = fmaxf(mx, sv[q]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            float const m_new = fmaxf(m_run, mx);
            float const alpha = expf(m_run - m_new);
            float sum = 0.f;
#ifdef ATT_PV_F8
            uint32_t p8lo = 0, p8hi = 0;
#endif
#pragma unroll
            for (int q = 0; q < KPT; ++q)
            {
                float const pr = vl[q] ? expf(sv[q] - m_new) : 0.f;
                sum += pr;
                int const col = sub * KPT + q, kp = col & 15;
#ifdef ATT_PV_F8
                if constexpr (HD == 512)
                {
                    const uint32_t pb = fp8q(pr * ATT_PV_F8_SCALE);
                    if (q < 4)
                        p8lo |= pb << (8 * q);
                    else
                        p8hi |= pb << (8 * (q - 4));
                }
                else
                {
                    int const lg = (col & ~15) + 2 * (kp >> 2) + (kp & 1) + ((kp & 2) ? 8 : 0);
                    Ps[sr * PS + lg] = __float2half_rn(pr);
                }
#else
                int const lg = (col & ~15) + 2 * (kp >> 2) + (kp & 1) + ((kp & 2) ? 8 : 0);
                Ps[sr * PS + lg] = __float2half_rn(pr);
#endif
            }
#ifdef ATT_PV_F8
            if constexpr (HD == 512)
            {
                uint8_t* pd = P8 + sr * PS + sub * KPT;
                if constexpr (KPT == 8)
                    *(uint2*) pd = make_uint2(p8lo, p8hi);
                else
                    *(uint32_t*) pd = p8lo;
            }
#endif
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if (sub == 0)
                As[sr] = alpha;
        }
        __syncthreads();
        // O += P V
#pragma unroll
        for (int mt = 0; mt < MTMAX; ++mt)
        {
            if (mt < MT)
            {
                float const a0 = As[mt * 16 + g], a1 = As[mt * 16 + 8 + g];
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
                {
                    oacc[mt][nt][0] *= a0;
                    oacc[mt][nt][1] *= a0;
                    oacc[mt][nt][2] *= a1;
                    oacc[mt][nt][3] *= a1;
                }
            }
        }
#ifdef ATT_PV_F8
        // E4M3 probabilities use a fixed power-of-two scale.  Native K=32 FP8 MMA
        // halves the PV instruction count; the scale is removed once at the
        // partial-output store below.
        if constexpr (HD == 512)
        {
#pragma unroll
            for (int ks = 0; ks < KSTEP / 32; ++ks)
            {
                uint32_t pf8[MTMAX][4];
#pragma unroll
                for (int mt = 0; mt < MTMAX; ++mt)
                {
                    uint8_t const* pa = P8 + (mt * 16 + g) * PS + ks * 32 + 4 * c;
                    pf8[mt][0] = *(uint32_t const*) pa;
                    pf8[mt][1] = *(uint32_t const*) (pa + 8 * PS);
                    pf8[mt][2] = *(uint32_t const*) (pa + 16);
                    pf8[mt][3] = *(uint32_t const*) (pa + 8 * PS + 16);
                }
#pragma unroll
                for (int np = 0; np < NTW / 2; ++np)
                {
                    uint32_t b00, b01, b10, b11;
                    uint8_t const* vp = Vs + (ks * 32 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8;
                    ldsm_v8t(b00, b01, vp);
                    ldsm_v8t(b10, b11, vp + 16 * VS);
#pragma unroll
                    for (int mt = 0; mt < MTMAX; ++mt)
                    {
                        mma_f8(oacc[mt][2 * np], pf8[mt][0], pf8[mt][1], pf8[mt][2], pf8[mt][3], b00, b10);
                        mma_f8(oacc[mt][2 * np + 1], pf8[mt][0], pf8[mt][1], pf8[mt][2], pf8[mt][3], b01, b11);
                    }
                }
            }
        }
        else
        {
#pragma unroll
            for (int ks = 0; ks < KSTEP / 16; ++ks)
            {
                if constexpr (MTMAX == 1)
                {
                    __half const* pa = Ps + g * PS + ks * 16 + 2 * c;
                    const uint32_t a0 = *(uint32_t const*) pa;
                    const uint32_t a1 = *(uint32_t const*) (pa + 8 * PS);
                    const uint32_t a2 = *(uint32_t const*) (pa + 8);
                    const uint32_t a3 = *(uint32_t const*) (pa + 8 * PS + 8);
#pragma unroll
                    for (int np = 0; np < NTW / 2; ++np)
                    {
                        uint32_t r0, r1;
                        ldsm_v8t(r0, r1, Vs + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                        mma_f16(oacc[0][2 * np], a0, a1, a2, a3, cvt_e4m3x2(r0), cvt_e4m3x2(r0 >> 16));
                        mma_f16(oacc[0][2 * np + 1], a0, a1, a2, a3, cvt_e4m3x2(r1), cvt_e4m3x2(r1 >> 16));
                    }
                }
                else
                {
                    uint32_t pf[2][4];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        __half const* pa = Ps + (mt * 16 + g) * PS + ks * 16 + 2 * c;
                        pf[mt][0] = *(uint32_t const*) pa;
                        pf[mt][1] = *(uint32_t const*) (pa + 8 * PS);
                        pf[mt][2] = *(uint32_t const*) (pa + 8);
                        pf[mt][3] = *(uint32_t const*) (pa + 8 * PS + 8);
                    }
#pragma unroll
                    for (int np = 0; np < NTW / 2; ++np)
                    {
                        uint32_t r0, r1;
                        ldsm_v8t(r0, r1, Vs + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                        const uint32_t b00 = cvt_e4m3x2(r0), b01 = cvt_e4m3x2(r0 >> 16);
                        const uint32_t b10 = cvt_e4m3x2(r1), b11 = cvt_e4m3x2(r1 >> 16);
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt)
                        {
                            mma_f16(oacc[mt][2 * np], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b00, b01);
                            mma_f16(oacc[mt][2 * np + 1], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b10, b11);
                        }
                    }
                }
            }
        }
#else
#pragma unroll
        for (int ks = 0; ks < KSTEP / 16; ++ks)
        {
            if constexpr (MTMAX == 1)
            {
                __half const* pa = Ps + g * PS + ks * 16 + 2 * c;
                const uint32_t a0 = *(uint32_t const*) pa;
                const uint32_t a1 = *(uint32_t const*) (pa + 8 * PS);
                const uint32_t a2 = *(uint32_t const*) (pa + 8);
                const uint32_t a3 = *(uint32_t const*) (pa + 8 * PS + 8);
#ifdef V_LDSM
#pragma unroll
                for (int np = 0; np < NTW / 2; ++np)
                {
                    uint32_t r0, r1;
                    ldsm_v8t(r0, r1, Vs + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                    mma_f16(oacc[0][2 * np], a0, a1, a2, a3, cvt_e4m3x2(r0), cvt_e4m3x2(r0 >> 16));
                    mma_f16(oacc[0][2 * np + 1], a0, a1, a2, a3, cvt_e4m3x2(r1), cvt_e4m3x2(r1 >> 16));
                }
#else
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
                {
                    int const d = (w * NTW + nt) * 8 + g;
                    int const vk = ks * 16 + 4 * c;
                    uint8_t const* vp = Vs + vk * VS + d;
                    const uint32_t lo = (uint32_t) vp[0] | ((uint32_t) vp[VS] << 8);
                    const uint32_t hi = (uint32_t) vp[2 * VS] | ((uint32_t) vp[3 * VS] << 8);
                    const uint32_t b0 = cvt_e4m3x2(lo), b1 = cvt_e4m3x2(hi);
                    mma_f16(oacc[0][nt], a0, a1, a2, a3, b0, b1);
                }
#endif
            }
            else
            {
                uint32_t pf[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                {
                    __half const* pa = Ps + (mt * 16 + g) * PS + ks * 16 + 2 * c;
                    pf[mt][0] = *(uint32_t const*) pa;
                    pf[mt][1] = *(uint32_t const*) (pa + 8 * PS);
                    pf[mt][2] = *(uint32_t const*) (pa + 8);
                    pf[mt][3] = *(uint32_t const*) (pa + 8 * PS + 8);
                }
#ifdef V_LDSM
#pragma unroll
                for (int np = 0; np < NTW / 2; ++np)
                {
                    uint32_t r0, r1;
                    ldsm_v8t(r0, r1, Vs + (ks * 16 + (lane & 15)) * VS + (w * NTW + 2 * np) * 8);
                    const uint32_t b00 = cvt_e4m3x2(r0), b01 = cvt_e4m3x2(r0 >> 16);
                    const uint32_t b10 = cvt_e4m3x2(r1), b11 = cvt_e4m3x2(r1 >> 16);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        mma_f16(oacc[mt][2 * np], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b00, b01);
                        mma_f16(oacc[mt][2 * np + 1], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b10, b11);
                    }
                }
#else
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
                {
                    int const d = (w * NTW + nt) * 8 + g;
                    int const vk = ks * 16 + 4 * c;
                    uint8_t const* vp = Vs + vk * VS + d;
                    const uint32_t lo = (uint32_t) vp[0] | ((uint32_t) vp[VS] << 8);
                    const uint32_t hi = (uint32_t) vp[2 * VS] | ((uint32_t) vp[3 * VS] << 8);
                    const uint32_t b0 = cvt_e4m3x2(lo), b1 = cvt_e4m3x2(hi);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                        mma_f16(oacc[mt][nt], pf[mt][0], pf[mt][1], pf[mt][2], pf[mt][3], b0, b1);
                }
#endif
            }
        }
#endif
#ifdef PRUNE_TAIL_BARRIERS
        if (k0 + KSTEP < ke)
            __syncthreads();
#else
        __syncthreads();
#endif
    }
#endif
    ATT_STAMP(s_curlayer, 2);
    // partial outputs: [item][32 rows][HD]
    __half* op = g_opart + (size_t) item * 32 * HD;
#pragma unroll
    for (int mt = 0; mt < MTMAX; ++mt)
    {
        if (mt < MT)
        {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt)
            {
                int const d = (w * NTW + nt) * 8 + 2 * c;
                int const r0 = mt * 16 + g;
#ifdef ATT_PV_F8
                constexpr float pscale_inv = 1.0f / ATT_PV_F8_SCALE;
                float const os = HD == 512 ? pscale_inv : 1.0f;
                if (r0 < rows)
                    *(__half2*) (op + (size_t) r0 * HD + d)
                        = __floats2half2_rn(oacc[mt][nt][0] * os, oacc[mt][nt][1] * os);
                if (r0 + 8 < rows)
                    *(__half2*) (op + (size_t) (r0 + 8) * HD + d)
                        = __floats2half2_rn(oacc[mt][nt][2] * os, oacc[mt][nt][3] * os);
#else
                if (r0 < rows)
                    *(__half2*) (op + (size_t) r0 * HD + d) = __floats2half2_rn(oacc[mt][nt][0], oacc[mt][nt][1]);
                if (r0 + 8 < rows)
                    *(__half2*) (op + (size_t) (r0 + 8) * HD + d) = __floats2half2_rn(oacc[mt][nt][2], oacc[mt][nt][3]);
#endif
            }
        }
    }
    if (srvalid && sub == 0)
        g_ml[item * 32 + sr] = make_float2(m_run, l_run);
    return true;
}

// GEMM r of layer i: 0 qkv/q, 1 o, 2 gate_up, 3 down, 4 PLE gate, 5 PLE proj.  Returns true for FP8 weights.
__device__ bool make_args(Params const& p, int i, int r, GemmArgs& ga)
{
    int kind, j;
    layer_kind(i, kind, j);
    bool const kvp = kind < 2;
    bool const sliding = (kind == 0 || kind == 2);
    int const hd = sliding ? 256 : 512;
    float const* sc = kind == 0 ? p.a_sc + j * 8
        : kind == 1             ? p.b_sc + j * 8
        : kind == 2             ? p.c_sc + j * 6
                                : p.d_sc + j * 6;
    int const so = kvp ? 2 : 0;
    int const inter = kvp ? 6144 : 12288;
    const size_t J = (size_t) j;
    ga.gu_inter = 0;
    ga.sc = 1.f;
    ga.sc2 = 1.f;
    ga.aux = nullptr;
    ga.S = 1;
    switch (r)
    {
    case 0:
    {
        int const N1 = kvp ? 10 * hd : 8 * hd;
        ga.mtiles = N1 / 16;
        ga.epi = EPI_QKV;
        ga.out = g_qkv;
        ga.ldo = N1;
        if (kvp)
        {
            ga.W = kind == 0 ? p.a_qkv + J * 2560 * 1536 : p.b_qkv + J * 5120 * 1536;
            ga.ldw = 1536;
            ga.X = g_xa;
            ga.ldx = 1536;
            ga.Kb = 1536;
            ga.sc = sc[0] * sc[1];
            return true;
        }
        ga.W = kind == 2 ? (uint8_t const*) (p.c_q + J * 2048 * 1536) : (uint8_t const*) (p.d_q + J * 4096 * 1536);
        ga.ldw = 3072;
        ga.X = (uint8_t const*) g_xbf;
        ga.ldx = 3072;
        ga.Kb = 3072;
        ga.S = 2;
        return false;
    }
    case 1:
    {
        int const K = 8 * hd;
        ga.W = kind == 0 ? p.a_o + J * 1536 * 2048
            : kind == 1  ? p.b_o + J * 1536 * 4096
            : kind == 2  ? p.c_o + J * 1536 * 2048
                         : p.d_o + J * 1536 * 4096;
        ga.ldw = K;
        ga.X = g_xb;
        ga.ldx = K;
        ga.Kb = K;
        ga.mtiles = HID / 16;
        ga.S = K / 512;
        ga.epi = EPI_PART;
        ga.out = g_ypart;
        ga.ldo = HID;
        return true;
    }
    case 2:
        ga.W = kind == 0 ? p.a_gu + J * 12288 * 1536
            : kind == 1  ? p.b_gu + J * 12288 * 1536
            : kind == 2  ? p.c_gu + J * 24576 * 1536
                         : p.d_gu + J * 24576 * 1536;
        ga.ldw = 1536;
        ga.X = g_xa;
        ga.ldx = 1536;
        ga.Kb = 1536;
        ga.mtiles = inter / 8;
        ga.gu_inter = inter;
        ga.epi = EPI_GU;
        ga.sc = sc[so + 2] * sc[so + 3];
        ga.sc2 = sc[so + 5];
        ga.out = g_xb;
        ga.ldo = inter;
        return true;
    case 3:
        ga.W = kind == 0 ? p.a_down + J * 1536 * 6144
            : kind == 1  ? p.b_down + J * 1536 * 6144
            : kind == 2  ? p.c_down + J * 1536 * 12288
                         : p.d_down + J * 1536 * 12288;
        ga.ldw = inter;
        ga.X = g_xb;
        ga.ldx = inter;
        ga.Kb = inter;
        ga.mtiles = HID / 16;
        ga.S = kvp ? 4 : 8;
        ga.epi = EPI_PART;
        ga.out = g_ypart;
        ga.ldo = HID;
        return true;
    case 4:
        ga.W = (uint8_t const*) (p.ple_gate_w + (size_t) i * 256 * 1536);
        ga.ldw = 3072;
        ga.X = (uint8_t const*) g_h;
        ga.ldx = 3072;
        ga.Kb = 3072;
        ga.mtiles = 16;
        ga.S = PG_SPLIT;
        ga.epi = EPI_PLEGATE;
        ga.out = g_gbuf;
        ga.ldo = PLED;
        ga.aux = g_pli + i * PLED;
        return false;
    default:
        ga.W = (uint8_t const*) (p.ple_proj_w + (size_t) i * 1536 * 256);
        ga.ldw = 512;
        ga.X = (uint8_t const*) g_gbuf;
        ga.ldx = 512;
        ga.Kb = 512;
        ga.mtiles = HID / 16;
        ga.epi = EPI_PART16;
        ga.out = g_ypart_h;
        ga.ldo = HID;
        return false;
    }
}

#ifndef WT_MIN_NT
#define WT_MIN_NT 99
#endif
#ifndef WT_D
#define WT_D 6
#endif
#ifndef WT_INLINE
#define WT_INLINE __noinline__
#endif
#define WT_XS 1600 // smem bytes per staged activation row: 1536-byte K-slice + 64-byte bank skew

// Warp-task GEMM for the FP8 gate_up / down projections at large token counts.  Every warp owns whole
// 16-row tasks over the block's 1536-byte K-slice (activations staged once in smem), streams its weights
// through a WT_D-deep register ring that runs across task boundaries, and stores the epilogue straight
// from its accumulators: no cross-warp reduction and no block barrier per task.
template <int NT, int EPI>
__device__ WT_INLINE void gemm_wt(GemmArgs const& ga, int T, uint8_t* xs, unsigned int const* wt)
{
    constexpr int KCH = 24; // 64-byte chunks per K-slice
    constexpr int D = WT_D;
    static_assert(KCH % D == 0, "ring depth must divide the chunk count");
    static_assert(EPI == EPI_GU || EPI == EPI_PART, "warp-task epilogues");
    const GemmArgs a = ga;
    int const tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
    int const S = a.S;
    int const Ks = a.Kb / S; // 1536 for every caller
    int const nb = gridDim.x - NPF_OF(NT);
    if ((int) blockIdx.x >= nb)
        return;
    int const ss = (int) blockIdx.x % S, q0 = (int) blockIdx.x / S;
    int const nbs = (nb - ss + S - 1) / S;
    int const ntl = q0 < a.mtiles ? (a.mtiles - q0 + nbs - 1) / nbs : 0;
    int const nwt = ntl > w ? (ntl - w + 7) / 8 : 0; // this warp's tasks: block tasks w, w + 8, ...
    int const rpm = a.gu_inter ? 8 : 16;
    long const hid = (long) (a.gu_inter ? a.gu_inter : 8) * a.ldw;
    int const kslice = ss * Ks;
    uint8_t const* wbase = a.W + (long) g * a.ldw + 16 * c + kslice;
    long const tstride = (long) 8 * nbs * rpm * a.ldw;
    long toff = (long) ((q0 + w * nbs) * rpm) * a.ldw;
    uint4 wl[D], wh[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
    {
        if (nwt > 0)
        {
            wl[d] = ldg_w(wbase + toff + d * 64);
            wh[d] = ldg_w(wbase + toff + d * 64 + hid);
        }
    }
    if (wt)
        grid_wait(*wt, gridDim.x - NPF_OF(NT)); // activations are produced by the previous phase
    if (ntl > 0)
    {
        for (int i = tid; i < T * (KCH * 4); i += NTHR)
        {
            int const t = i / (KCH * 4), q = i - t * (KCH * 4);
            cp_async16(xs + t * WT_XS + q * 16, a.X + (long) t * a.ldx + kslice + q * 16);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if (nwt == 0)
        return;
    int xrow[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        xrow[nt] = min(nt * 8 + g, T - 1) * WT_XS + 16 * c;
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    for (int kt = 0; kt < nwt; ++kt)
    {
        bool const more = kt + 1 < nwt;
        long const noff = toff + tstride;
#pragma unroll 1
        for (int u0 = 0; u0 < KCH; u0 += D)
#pragma unroll
            for (int d = 0; d < D; ++d)
            {
                int const u = u0 + d;
                uint4 xv[NT];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    xv[nt] = *(uint4 const*) (xs + xrow[nt] + u * 64);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    mma_f8(acc[nt], wl[d].x, wh[d].x, wl[d].y, wh[d].y, xv[nt].x, xv[nt].y);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    mma_f8(acc[nt], wl[d].z, wh[d].z, wl[d].w, wh[d].w, xv[nt].z, xv[nt].w);
                // refill this ring slot with the chunk D ahead (the next task's head once this task is exhausted)
                if (u + D < KCH)
                {
                    wl[d] = ldg_w(wbase + toff + (u + D) * 64);
                    wh[d] = ldg_w(wbase + toff + (u + D) * 64 + hid);
                }
                else if (more)
                {
                    wl[d] = ldg_w(wbase + noff + (u + D - KCH) * 64);
                    wh[d] = ldg_w(wbase + noff + (u + D - KCH) * 64 + hid);
                }
            }
        int const mt = q0 + (w + 8 * kt) * nbs;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            int const col = nt * 8 + 2 * c;
#pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                if (col + j < T)
                {
                    if constexpr (EPI == EPI_GU)
                    {
                        float const gate = bfr(acc[nt][j] * a.sc), up = bfr(acc[nt][2 + j] * a.sc);
                        ((uint8_t*) a.out)[(long) (col + j) * a.ldo + mt * 8 + g]
                            = fp8q(bfr(gelu_tanh(gate) * up) / a.sc2);
                    }
                    else
                    {
                        float* yp = (float*) a.out + (long) ss * TMAX * HID + (long) (col + j) * HID + mt * 16 + g;
                        yp[0] = acc[nt][j];
                        yp[8] = acc[nt][2 + j];
                    }
                }
            }
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
        toff = noff;
    }
}

template <int NT, int R>
__device__ __forceinline__ void run_gemm(Params const& p, int i, int T, char* smem, unsigned int const* wt)
{
    static_assert(R >= 0 && R < 6);
    GemmArgs ga;
    bool const f8 = make_args(p, i, R, ga);
    constexpr int EPI = R == 0 ? EPI_QKV : R == 2 ? EPI_GU : R == 4 ? EPI_PLEGATE : R == 5 ? EPI_PART16 : EPI_PART;
    if constexpr (NT <= 6)
    {
        if constexpr (R == 0)
        {
            if (f8)
                gemm<NT, true, 3, EPI>(ga, T, (float*) smem, wt);
            else
                gemm<NT, false, 3, EPI>(ga, T, (float*) smem, wt);
        }
        else if constexpr (R == 1)
        {
            gemm<NT, true, 1, EPI>(ga, T, (float*) smem, wt);
        }
        else if constexpr (R == 2)
        {
            if constexpr (NT >= WT_MIN_NT)
                gemm_wt<NT, EPI>(ga, T, (uint8_t*) smem, wt);
            else
                gemm<NT, true, 3, EPI>(ga, T, (float*) smem, wt);
        }
        else if constexpr (R == 3)
        {
            gemm<NT, true, 3, EPI>(ga, T, (float*) smem, wt);
        }
        else
        {
            gemm<NT, false, 1, EPI>(ga, T, (float*) smem, wt);
        }
    }
    else
    {
        if constexpr (R == 0)
        {
            if (f8)
                gemm<NT, true, 0, EPI>(ga, T, (float*) smem, wt);
            else
                gemm<NT, false, 0, EPI>(ga, T, (float*) smem, wt);
        }
        else if constexpr (R >= 1 && R <= 3)
        {
            gemm<NT, true, 0, EPI>(ga, T, (float*) smem, wt);
        }
        else
        {
            gemm<NT, false, 0, EPI>(ga, T, (float*) smem, wt);
        }
    }
}

// True when this block owns a non-empty split-KV attention item.  Other blocks can skip the QKV wait and
// continue toward independent work now that every phase uses a distinct barrier counter.
template <int KSTEP>
__device__ __forceinline__ bool attn_busy(Params const& p, int i, int nb)
{
    int kind, j;
    layer_kind(i, kind, j);
    bool const sliding = (kind == 0 || kind == 2);
    int const Q = p.Q, NG = p.ng, NIc = p.nic;
    int const nitems = p.R * NG * NIc;
    for (int item = blockIdx.x; item < nitems; item += nb)
    {
        int const jj = item % NIc, r = item / NIc / NG;
        int const cl = p.cache_lens[r];
        int const kend = cl + Q;
        int const kstart = sliding ? max(0, cl - 511) : 0;
        int const nk = kend - kstart;
        int const csz = (((nk + NIc - 1) / NIc) + KSTEP - 1) & ~(KSTEP - 1);
        if (kend - jj * csz > kstart)
            return true;
    }
    return false;
}

// L2 prefetch (one 128-byte line per thread) of the first 32-key step of this block's attention items.  Only old
// cache rows are touched, which the preceding QKV phase does not write, so it may run before the barrier wait.
template <int KSTEP>
__device__ __noinline__ void attn_l2_prefetch(Params const& p, int i, int nb)
{
    int kind, j;
    layer_kind(i, kind, j);
    bool const sliding = (kind == 0 || kind == 2);
    bool const kvp = kind < 2;
    int const HD = sliding ? 256 : 512;
    uint8_t* kc;
    uint8_t* vc;
    long long kvps;
    kv_layer(p, kind, j, kc, vc, kvps);
    int const Q = p.Q, NG = p.ng, NIc = p.nic;
    int const nitems = p.R * NG * NIc;
    int const LPR = HD / 128;
    for (int item = blockIdx.x; item < nitems; item += nb)
    {
        int const jj = item % NIc, rg = item / NIc;
        int const r = rg / NG;
        int const cl = p.cache_lens[r];
        int const kend = cl + Q;
        int const kstart = sliding ? max(0, cl - 511) : 0;
        int const nk = kend - kstart;
        int const csz = (((nk + NIc - 1) / NIc) + KSTEP - 1) & ~(KSTEP - 1);
        int const ke = kend - jj * csz;
        if (ke <= kstart)
            continue;
        int const kb = max(kstart, ke - csz);
        bool const use_new = kvp && jj == 0;
#ifdef ATTN_EXTERNAL_PF_K_ONLY
        int const PF_LINES = KSTEP * LPR;
#else
        int const PF_LINES = 2 * KSTEP * LPR;
#endif
        for (int li = threadIdx.x; li < PF_LINES; li += NTHR)
        {
            int const isv = li / (KSTEP * LPR), rem = li - isv * KSTEP * LPR;
            int const key = rem / LPR, ln = rem - key * LPR;
            int const kk = kb + key;
            if (kk < ke && !(use_new && kk >= cl))
            {
                uint8_t const* pa = (isv ? vc : kc) + kv_off(kv_rpt(p, sliding, r), kvps, kk, HD) + ln * 128;
#ifdef KV_PERSIST
                bool const kvlast = sliding ? ((KV_PERSIST & 1) && (kind == 2 || (kind == 0 && j == 11)))
                                            : ((KV_PERSIST & 2) && (kind == 3 || (kind == 1 && j == 2)));
#ifndef KV_POL_EF
                if (kvlast)
                    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(pa));
                else
#endif
#endif
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(pa));
            }
        }
    }
}

// L2 bulk prefetch of every weight task this block will own in GEMM r of layer i (same task mapping as
// gemm()).  Issued by warp 0 during latency-bound phases so DRAM keeps streaming while the grid waits.
__device__ __forceinline__ void bulk_pf(void const* ptr, unsigned bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_pf_w(void const* ptr, unsigned bytes)
{
#ifdef WEIGHT_EF2
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(ptr), "r"(bytes), "l"(pol)
                 : "memory");
#else
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
#endif
}

// Bulk L2 prefetch of the old K/V rows of the attention items this block will own in layer i.  Old rows
// [0, cache_len) are never written by this kernel, so this may be issued phases ahead of the attention.
// Head groups of one request share rows; only head group 0 issues.
template <int KSTEP>
__device__ __noinline__ void attn_kv_bulk_prefetch(Params const& p, int i, int nb)
{
    if (threadIdx.x != 0)
        return;
    int kind, j;
    layer_kind(i, kind, j);
    bool const sliding = (kind == 0 || kind == 2);
    int const HD = sliding ? 256 : 512;
    uint8_t* kc;
    uint8_t* vc;
    long long kvps;
    kv_layer(p, kind, j, kc, vc, kvps);
    int const Q = p.Q, NG = p.ng, NIc = p.nic;
    int const nitems = p.R * NG * NIc;
    for (int item = blockIdx.x; item < nitems; item += nb)
    {
        int const jj = item % NIc, rg = item / NIc;
        int const hg = rg % NG, r = rg / NG;
        if (hg != 0)
            continue;
        int const cl = p.cache_lens[r];
        int const kend = cl + Q;
        int const kstart = sliding ? max(0, cl - 511) : 0;
        int const nk = kend - kstart;
        int const csz = (((nk + NIc - 1) / NIc) + KSTEP - 1) & ~(KSTEP - 1);
        int const ke = min(kend - jj * csz, cl);
        int const kb = max(kstart, kend - jj * csz - csz);
        if (ke <= kb)
            continue;
        int const* rpt = kv_rpt(p, sliding, r);
        for (int k0 = kb; k0 < ke;)
        { // one bulk prefetch per (partial) page
            int const k1 = min(ke, ((k0 >> KV_PAGE_SHIFT) + 1) << KV_PAGE_SHIFT);
            const size_t off = kv_off(rpt, kvps, k0, HD);
            unsigned const nbt = (unsigned) ((k1 - k0) * HD);
            bulk_pf(kc + off, nbt);
            bulk_pf(vc + off, nbt);
            k0 = k1;
        }
    }
}

__device__ __noinline__ void own_task_prefetch(Params const& p, int i, int r, int nb_)
{
    if (threadIdx.x >= 32)
        return;
    GemmArgs a;
    make_args(p, i, r, a);
    int const lane = threadIdx.x;
    int const S = a.S, nb = nb_;
    int const ss = (int) blockIdx.x % S, q0 = (int) blockIdx.x / S;
    int const nbs = (nb - ss + S - 1) / S;
    int const Ks = a.Kb / S;
    int const rpm = a.gu_inter ? 8 : 16;
    if (S == 1)
    {
        // rows of a task are contiguous: one bulk prefetch per task (two for gate/up halves)
        int const nh = a.gu_inter ? 2 : 1;
        int const ntl = q0 < a.mtiles ? (a.mtiles - q0 + nbs - 1) / nbs : 0;
        for (int e = lane; e < ntl * nh; e += 32)
        {
            int const k = e / nh, h = e - k * nh;
            long const row = (long) (q0 + k * nbs) * rpm + (h ? a.gu_inter : 0);
            bulk_pf(a.W + row * a.ldw, (unsigned) (rpm * a.ldw));
        }
    }
    else
    {
        int const ntl = q0 < a.mtiles ? (a.mtiles - q0 + nbs - 1) / nbs : 0;
        for (int e = lane; e < ntl * rpm; e += 32)
        {
            int const k = e / rpm, rr = e - k * rpm;
            long const row = (long) (q0 + k * nbs) * rpm + rr;
            bulk_pf(a.W + row * a.ldw + (long) ss * Ks, (unsigned) Ks);
        }
    }
}

// Reset barrier state for the next launch once every block (compute and prefetch) has finished.
__device__ __forceinline__ void kernel_exit()
{
    __shared__ int s_last_blk;
    if (threadIdx.x == 0)
    {
        __threadfence();
        unsigned int const prev = atomicAdd(&g_done, 1u);
        s_last_blk = (prev == gridDim.x - 1);
    }
    __syncthreads();
    if (s_last_blk)
    {
        for (int k = threadIdx.x; k < NBAR; k += NTHR)
            g_bars[k] = 0u;
        for (int k = threadIdx.x; k < NLAYER * 32; k += NTHR)
            g_attn_ready[k] = 0u;
        if (threadIdx.x == 0)
            g_done = 0u;
        __threadfence();
    }
}

#ifndef PF_KVL_LEAD
#define PF_KVL_LEAD 0
#endif
#ifndef PF_KVL_MINT
#define PF_KVL_MINT 17 // small batches: the per-layer KV is small and not DRAM-bound
#endif
#ifndef PF_KVL_SHARED_MINT
#define PF_KVL_SHARED_MINT 1000 // re-fetch the layer-14 cache before the KV-shared full layers from this many tokens
#endif
#ifndef PF_KVL_MIN
#define PF_KVL_MIN 0
#endif
#ifndef PF_KVL_MAX
#define PF_KVL_MAX 14 // KV-producing full layers only: the shared layer-14 cache gains nothing from a re-fetch
#endif
#ifndef PF_KVL_RATE
#define PF_KVL_RATE 1500ull
#endif
#ifndef PF_CH
#define PF_CH 32768 // bytes in flight per prefetch CTA (one chunk)
#endif
// prefetch rate (GB/s == bytes/ns) and window (how far the frontier may run ahead of the weights being consumed),
// separately for the small-T (<= 24 tokens) and large-T builds: large batches keep more KV cache in L2
#ifndef PF_RATE_SMALL
#define PF_RATE_SMALL 1300ull
#endif
#ifndef PF_RATE_LARGE
#define PF_RATE_LARGE 1000ull
#endif
// Very small token batches have enough L2 room to stream every weight region, not only the large projections.
#ifndef PF_ALL_MAXNT
#define PF_ALL_MAXNT 0
#endif
#ifndef PF_RATE_ALL
#define PF_RATE_ALL 1500ull
#endif
#ifndef PF_RATE_ALL_NT1
#define PF_RATE_ALL_NT1 PF_RATE_ALL
#endif
#ifndef PF_RATE_ALL_NT2
#define PF_RATE_ALL_NT2 PF_RATE_ALL
#endif
#ifndef PF_RATE_ALL_NT3
#define PF_RATE_ALL_NT3 PF_RATE_ALL_NT2
#endif
#ifndef PF_KVPROD_ALL
#define PF_KVPROD_ALL 0
#endif
#ifndef PF_WIN_SMALL
#define PF_WIN_SMALL 112
#endif
// Single-request batches have little KV traffic, so a second independent issuing warp leader can
// better sustain the target prefetch rate without disturbing multi-request attention.
#ifndef PF_VT_R1
#define PF_VT_R1 2
#endif
#ifndef PF_VT_LARGE
#define PF_VT_LARGE 1
#endif
#ifndef PF_WIN_NT3
#define PF_WIN_NT3 PF_WIN_SMALL // 17..24 tokens: larger KV footprint competes with prefetched weights
#endif
#ifndef PF_WIN_LARGE
#define PF_WIN_LARGE 64
#endif
// Region (see region()) whose weights the compute grid consumes once barrier target `b` has completed.
__device__ __forceinline__ int consuming_region(int b)
{
    if (b <= 0)
        return 0;
    if (b == 1)
        return 1;
    int const i = (b - 2) / 11, ph = (b - 2) - 11 * i + 1; // phase ph (1..11) of layer i is running
    // phase -> region within layer: qkv 0, attn/comb/oproj 1 (o), P6/gu 2, down 3, P9/pgate 4, pproj 5, P12 next qkv
    int const rk = (int) ((0x654432211100ull >> (4 * ph)) & 0xFull);
    return 1 + 6 * i + rk;
}

// Closed-loop L2 prefetcher: streams the weights in consumption order with PF_CH bytes in flight per CTA,
// never more than PF_WIN_MB ahead of the region the compute grid is consuming (tracked via barrier counters).
// Chunks already behind the consumption point are skipped; the CTA stops once the compute grid is done.
#define LAST_BAR (2 + 11 * NLAYER - 1)
#ifdef PHASE_TIMING
__device__ unsigned long long g_pft[NREG];
#endif
// Barrier completion is monotone (blocks arrive in program order), so one 16-counter probe can advance cb by up
// to 16.  Returns the number of barriers newly known complete.
__device__ __forceinline__ int probe_bars(int& cb, unsigned int ncb)
{
    int const b0 = cb & ~3;
    uint4 v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[q].x), "=r"(v[q].y), "=r"(v[q].z), "=r"(v[q].w)
                     : "l"(&g_bars[b0 + 4 * q])
                     : "memory");
    unsigned int const* a = (unsigned int const*) v;
    int adv = 0;
#pragma unroll
    for (int q = 0; q < 16; ++q)
    {
        if (b0 + q == cb && cb < LAST_BAR && a[q] >= ncb)
        {
            ++cb;
            ++adv;
        }
    }
    return adv;
}

__device__ __noinline__ void weight_prefetcher(
    Params const& p, unsigned int ncb, unsigned long long const win_bytes, unsigned long long const rate_gbps)
{
    int const npf = (int) (gridDim.x - ncb), pid = (int) (blockIdx.x - ncb);
    int const tid = threadIdx.x;
    __shared__ int s_act[2];
    int cb = 0, cons_k = 0;          // thread 0: barriers known complete, region being consumed
    unsigned long long cons_off = 0; // thread 0: stream offset of that region
    unsigned long long reg_off = 0;
    long n0 = 0;
    uint32_t sink = 0;
    int it = 0;
    bool stop = false;
    unsigned long long t_next = 0ull;
    constexpr int PER = PF_CH / NTHR / 16; // uint4 loads per thread
    for (int k = 0; k < NREG; ++k)
    {
        uint8_t const* base;
        unsigned long long sz;
        region(p, k, base, sz);
        bool const gu = (k >= 1) && ((k - 1) % 6 == 2);
        unsigned long long const half = sz / 2;
        long nch = (long) ((sz + PF_CH - 1) / PF_CH);
#ifdef PF_BIG_ONLY
        // only the gate_up / down regions: the compute blocks already preload the small ones before each barrier
        if (k == 0)
            nch = 0;
        else
        {
            int const rr = (k - 1) % 6, li = (k - 1) / 6;
            int lk, lj;
            layer_kind(li, lk, lj);
            bool want = rr == 2 || rr == 3;
#ifdef PF_QO
            // plus the multi-task BF16 q projections of KV-shared layers and the 4096-wide o projections
            want = want || (rr == 0 && lk >= 2) || (rr == 1 && (lk & 1));
#endif
            if (!want)
                nch = 0;
        }
#endif
#if defined(PF_KV) && defined(PF_BULK)
        // Before a full-attention layer's projections, thread 0 also streams that layer's valid KV rows [0, cache_len)
        // into L2 (own cache for layers 4/9/14, layer 14's cache for the KV-shared full layers).
        if (tid == 0 && k >= 1 && (k - 1) % 6 == 0 && !stop)
        {
            int const li = (k - 1) / 6;
            int lk, lj;
            layer_kind(li, lk, lj);
            if (lk & 1)
            {
                bool ok = false;
                for (;;)
                {
                    if (cb >= LAST_BAR || reg_off < cons_off)
                        break;
                    if (reg_off < cons_off + win_bytes)
                    {
                        ok = true;
                        break;
                    }
                    if (probe_bars(cb, ncb))
                    {
                        int const kk = consuming_region(cb);
                        while (cons_k < kk)
                        {
                            uint8_t const* b2;
                            unsigned long long s2;
                            region(p, cons_k, b2, s2);
                            cons_off += s2;
                            ++cons_k;
                        }
                    }
                    else
                    {
                        __nanosleep(200);
                    }
                }
                if (ok)
                {
                    int const jb = lk == 1 ? lj : 2;
                    uint8_t* kfl;
                    uint8_t* vfl;
                    long long kvps;
                    kv_layer(p, 1, jb, kfl, vfl, kvps);
                    int q = 0;
                    for (int r = 0; r < p.R; ++r)
                    {
                        unsigned long long const rb = (unsigned long long) p.cache_lens[r] * 512ull;
                        int const* rpt = kv_rpt(p, false, r);
                        int const nchr = (int) ((rb + KV_PS * 512 - 1) / (KV_PS * 512));
                        for (int isv = 0; isv < 2; ++isv)
                        {
                            uint8_t const* kvb = isv ? vfl : kfl;
                            for (int c = 0; c < nchr; ++c, ++q)
                            {
                                if (q % npf != pid)
                                    continue;
                                unsigned long long const o = (unsigned long long) c * (KV_PS * 512);
                                unsigned long long const nbytes = rb - o < (KV_PS * 512) ? rb - o : (KV_PS * 512);
                                const size_t pgo = (size_t) __ldg(rpt + c) * (size_t) kvps;
                                unsigned long long now;
                                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                                if (t_next == 0ull)
                                    t_next = now;
                                while (now < t_next)
                                {
                                    __nanosleep(100);
                                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                                }
                                bulk_pf(kvb + pgo, (unsigned) nbytes);
                                t_next = (t_next > now ? t_next : now) + (unsigned long long) npf * nbytes / rate_gbps;
                            }
                        }
                    }
                }
            }
        }
#endif
        long j = (long) (((pid - n0) % npf + npf) % npf);
        for (; j < nch; j += npf, ++it)
        {
            unsigned long long const spos = reg_off + (unsigned long long) j * PF_CH;
            if (tid == 0)
            {
                int act = 1;
                for (;;)
                {
                    if (cb >= LAST_BAR)
                    {
                        act = 2;
                        break;
                    }
                    if (spos + PF_CH <= cons_off)
                    {
                        act = 0;
                        break;
                    }          // already consumed
                    if (spos < cons_off + win_bytes)
                        break; // inside the window
                    if (probe_bars(cb, ncb))
                    {
                        int const kk = consuming_region(cb);
                        while (cons_k < kk)
                        {
                            uint8_t const* b2;
                            unsigned long long s2;
                            region(p, cons_k, b2, s2);
                            cons_off += s2;
                            ++cons_k;
                        }
                    }
                    else
                    {
                        __nanosleep(200);
                    }
                }
                s_act[it & 1] = act;
            }
            __syncthreads();
            int const act = s_act[it & 1];
#ifdef PHASE_TIMING
            if (tid == 0 && pid == 0 && j < npf)
            {
                unsigned long long tt;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt));
                g_pft[k] = act == 1 ? tt : 0ull;
            }
#endif
            if (act == 2)
            {
                stop = true;
                break;
            }
            if (act == 0)
                continue;
            // chunk address: gate_up regions interleave gate/up halves in consumption order
            uint8_t const* cp;
            unsigned long long cbytes;
            if (gu)
            {
                unsigned long long const o = (unsigned long long) (j >> 1) * PF_CH;
                cp = base + ((j & 1) ? half : 0) + o;
                cbytes = o < half ? (half - o < PF_CH ? half - o : PF_CH) : 0;
            }
            else
            {
                unsigned long long const o = (unsigned long long) j * PF_CH;
                cp = base + o;
                cbytes = sz - o < PF_CH ? sz - o : PF_CH;
            }
#ifdef PF_BULK
            // TMA bulk L2 prefetch, paced open-loop at PF_RATE_GBPS across all prefetch CTAs
            if (tid == 0 && cbytes > 0)
            {
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (t_next == 0ull)
                    t_next = now;
                while (now < t_next)
                {
                    __nanosleep(100);
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                }
                bulk_pf(cp, (unsigned) cbytes);
                t_next
                    = (t_next > now ? t_next : now) + (unsigned long long) npf * (unsigned long long) PF_CH / rate_gbps;
            }
            uint4 r[1];
            r[0] = make_uint4(0u, 0u, 0u, 0u);
            constexpr int PERX = 1;
#else
            uint4 r[PER];
            constexpr int PERX = PER;
#pragma unroll
            for (int e = 0; e < PER; ++e)
            {
                unsigned long long const off = ((unsigned long long) e * NTHR + tid) * 16;
                if (off < cbytes)
                    r[e] = ldg_w(cp + off);
                else
                    r[e] = make_uint4(0u, 0u, 0u, 0u);
            }
#endif
            // progress probe overlapped with the chunk loads
            if (tid == 0 && cb < LAST_BAR)
            {
                if (probe_bars(cb, ncb))
                {
                    int const kk = consuming_region(cb);
                    while (cons_k < kk)
                    {
                        uint8_t const* b2;
                        unsigned long long s2;
                        region(p, cons_k, b2, s2);
                        cons_off += s2;
                        ++cons_k;
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < PERX; ++e)
                sink ^= r[e].x ^ r[e].w;
        }
        if (stop)
            break;
        n0 += nch;
        reg_off += sz;
    }
    if (sink == 0x9e3779b9u && tid == NTHR)
        g_done = sink; // keeps the loads live; never true for tid < NTHR
    kernel_exit();
}

// Lean single-thread variant of the bulk prefetcher: thread 0 of each prefetch CTA walks its share of the chunks,
// holds each chunk until it is inside the window, paces issue with globaltimer, and overlaps the barrier-progress
// probe with the pacing wait so the probe latency never limits the issue rate.  No CTA-wide syncs in the loop.
__device__ __forceinline__ void advance_cons(Params const& p, int cb, int& cons_k, unsigned long long& cons_off)
{
    int const kk = consuming_region(cb);
    while (cons_k < kk)
    {
        uint8_t const* b2;
        unsigned long long s2;
        region(p, cons_k, b2, s2);
        cons_off += s2;
        ++cons_k;
    }
}

template <bool QO, bool ALL, bool KVALL = false, bool LATEQ = false, bool KVL = false>
__device__ __noinline__ void bulk_prefetcher(Params const& p, unsigned int ncb, unsigned long long const win_bytes,
    unsigned long long const rate_gbps, int const vt, int const hold = 0, unsigned long long const kvcap = 0ull)
{
    // One lane per issuing warp walks a disjoint share of the chunk stream.
    if ((threadIdx.x & 31) == 0 && (int) (threadIdx.x >> 5) < vt)
    {
        int const npf = (int) (gridDim.x - ncb) * vt;
        int const pid = (int) (blockIdx.x - ncb) * vt + (int) (threadIdx.x >> 5);
        int cb = 0, cons_k = 0;
        unsigned long long cons_off = 0, reg_off = 0, t_next = 0ull;
        long n0 = 0;
        bool stop = false;
        while (cb < hold)
        {
            if (probe_bars(cb, ncb))
                advance_cons(p, cb, cons_k, cons_off);
            else
                __nanosleep(200);
        }
        for (int k = 0; k < NREG && !stop; ++k)
        {
            uint8_t const* base;
            unsigned long long sz;
            region(p, k, base, sz);
            bool const gu = (k >= 1) && ((k - 1) % 6 == 2);
            unsigned long long const half = sz / 2;
            long nch = (long) ((sz + PF_CH - 1) / PF_CH);
            if (k == 0)
                nch = 0;
            else
            {
                int const rr = (k - 1) % 6, li = (k - 1) / 6;
                int lk, lj;
                layer_kind(li, lk, lj);
                bool want = rr == 2 || rr == 3;
#ifdef PF_QO
                if constexpr (QO)
                    want = want || (rr == 0 && lk >= 2) || (rr == 1 && (lk & 1));
#endif
                if constexpr (ALL)
                    want = true;
                if constexpr (KVALL)
                    want = want || lk < 2;
                if constexpr (LATEQ)
                    want = want || (lk == 3 && rr == 0);
                if (!want)
                    nch = 0;
            }
#ifdef PF_KVL
            // Before a full-attention layer's projections, stream that layer's old KV rows [0, cache_len) into L2 (its
            // own cache for layers 4/9/14, the layer-14 cache for the KV-shared full layers) once its qkv region is
            // inside the window, paced like the weights: the attention steps then hit L2 instead of waiting on DRAM.
            if (KVL && k >= 1 && (k - 1) % 6 == 0 && !stop)
            {
                int const li = (k - 1) / 6;
                int lk, lj;
                layer_kind(li, lk, lj);
                if ((lk & 1) && li >= PF_KVL_MIN && p.T >= PF_KVL_MINT
                    && (li <= PF_KVL_MAX || p.T >= PF_KVL_SHARED_MINT))
                {
                    bool ok = false;
                    for (;;)
                    {
                        if (cb >= LAST_BAR || reg_off < cons_off)
                            break;
                        if (reg_off < cons_off + win_bytes + (unsigned long long) PF_KVL_LEAD * 1048576ull)
                        {
                            ok = true;
                            break;
                        }
                        if (probe_bars(cb, ncb))
                            advance_cons(p, cb, cons_k, cons_off);
                        else
                            __nanosleep(200);
                    }
                    if (ok)
                    {
                        int const jb = lk == 1 ? lj : 2;
                        uint8_t* kfl;
                        uint8_t* vfl;
                        long long kvps;
                        kv_layer(p, 1, jb, kfl, vfl, kvps);
                        int q = 0;
                        for (int r = 0; r < p.R; ++r)
                        {
                            unsigned long long const rb = (unsigned long long) p.cache_lens[r] * 512ull;
                            int const* rpt = kv_rpt(p, false, r);
                            int const nchr = (int) ((rb + KV_PS * 512 - 1) / (KV_PS * 512));
#ifdef PF_KVL_K_ONLY
                            constexpr int nplane = 1;
#else
                            constexpr int nplane = 2;
#endif
                            for (int isv = 0; isv < nplane; ++isv)
                            {
                                uint8_t const* kvb = isv ? vfl : kfl;
                                for (int c = 0; c < nchr; ++c, ++q)
                                {
                                    if (q % npf != pid)
                                        continue;
                                    unsigned long long const o = (unsigned long long) c * (KV_PS * 512);
                                    unsigned long long const nbytes = rb - o < (KV_PS * 512) ? rb - o : (KV_PS * 512);
                                    const size_t pgo = (size_t) __ldg(rpt + c) * (size_t) kvps;
                                    unsigned long long now;
                                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                                    if (t_next == 0ull)
                                        t_next = now;
                                    while (now < t_next)
                                    {
                                        __nanosleep(64);
                                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                                    }
                                    bulk_pf(kvb + pgo, (unsigned) nbytes);
                                    t_next = (t_next > now ? t_next : now)
                                        + (unsigned long long) npf * nbytes / (unsigned long long) PF_KVL_RATE;
                                }
                            }
                        }
                    }
                }
            }
#endif
            long j = (long) (((pid - n0) % npf + npf) % npf);
            for (; j < nch; j += npf)
            {
                unsigned long long const spos = reg_off + (unsigned long long) j * PF_CH;
                int act = 1;
                for (;;)
                {
                    if (cb >= LAST_BAR)
                    {
                        act = 2;
                        break;
                    }
                    if (spos + PF_CH <= cons_off)
                    {
                        act = 0;
                        break;
                    }          // already consumed
                    if (spos < cons_off + win_bytes)
                        break; // inside the window
                    if (probe_bars(cb, ncb))
                        advance_cons(p, cb, cons_k, cons_off);
                    else
                        __nanosleep(200);
                }
                if (act == 2)
                {
                    stop = true;
                    break;
                }
                if (act == 0)
                    continue;
                uint8_t const* cp;
                unsigned long long cbytes;
                if (gu)
                {
                    unsigned long long const o = (unsigned long long) (j >> 1) * PF_CH;
                    cp = base + ((j & 1) ? half : 0) + o;
                    cbytes = o < half ? (half - o < PF_CH ? half - o : PF_CH) : 0;
                }
                else
                {
                    unsigned long long const o = (unsigned long long) j * PF_CH;
                    cp = base + o;
                    cbytes = sz - o < PF_CH ? sz - o : PF_CH;
                }
                if (cbytes == 0)
                    continue;
#ifdef PF_PAUSE_ATTN
                // hold the stream while a full-attention layer's attention phase streams its KV cache
                for (;;)
                {
                    bool hold = false;
                    if (cb >= 3 && cb < LAST_BAR && (cb - 3) % 11 == 0)
                    {
                        int lk2, lj2;
                        layer_kind((cb - 3) / 11, lk2, lj2);
                        hold = (PF_PAUSE_ATTN >> lk2) & 1;
                    }
                    if (!hold)
                        break;
                    if (probe_bars(cb, ncb))
                        advance_cons(p, cb, cons_k, cons_off);
                    else
                        __nanosleep(100);
                }
#endif
                // progress probe in flight during the pacing wait
                int const b0 = cb & ~3;
                uint4 v[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(v[q].x), "=r"(v[q].y), "=r"(v[q].z), "=r"(v[q].w)
                                 : "l"(&g_bars[b0 + 4 * q])
                                 : "memory");
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (t_next == 0ull)
                    t_next = now;
                while (now < t_next)
                {
                    __nanosleep(64);
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                }
                bulk_pf_w(cp, (unsigned) cbytes);
#ifdef PF_RATE_KVPROD
                unsigned long long const capr = kvcap ? kvcap : PF_RATE_KVPROD;
                unsigned long long const issue_rate = k <= 90 && rate_gbps > capr
#ifdef PF_RATE_KVPROD_MAX_T
                        && p.T <= PF_RATE_KVPROD_MAX_T
#endif
                    ? capr
                    : rate_gbps;
#else
                unsigned long long const issue_rate = rate_gbps;
#endif
                t_next = (t_next > now ? t_next : now) + (unsigned long long) npf * cbytes / issue_rate;
                unsigned int const* a = (unsigned int const*) v;
                int adv = 0;
#pragma unroll
                for (int q = 0; q < 16; ++q)
                {
                    if (b0 + q == cb && cb < LAST_BAR && a[q] >= ncb)
                    {
                        ++cb;
                        ++adv;
                    }
                }
                if (adv)
                    advance_cons(p, cb, cons_k, cons_off);
            }
            n0 += nch;
            reg_off += sz;
        }
    }
    kernel_exit();
}

// One layer's 256-row PLE projection, using two BF16 split-K tasks and a separate partial buffer.
template <int NT, bool VIRTUAL = false>
__device__ __forceinline__ void ple_slice(Params const& p, int i, int T, float* red, int vb = 0, int vnb = 0)
{
    LazyGemmArgs ga;
    // Lazy mode does not materialize the full projection, so its compact fp32 partials reuse g_pproj storage.
    ga.vb = vb;
    ga.vnb = vnb;
    ga.part = (float*) g_pproj;
    ga.W = (uint8_t const*) (p.ple_mp_w + (size_t) i * PLED * HID);
    ga.ldw = 3072;
    ga.X = (uint8_t const*) p.hidden;
    ga.ldx = 3072;
    ga.Kb = 3072;
    ga.mtiles = PLED / 16;
    ga.S = 2;
    ga.gu_inter = 0;
    ga.epi = EPI_QKV;
    ga.sc = 1.f;
    ga.sc2 = 1.f;
    ga.out = g_qkv;
    ga.ldo = PLED;
    ga.aux = nullptr;
    if constexpr (NT <= 6)
        gemm<NT, false, 3, EPI_QKV, VIRTUAL, true, LazyGemmArgs>(ga, T, red, (unsigned int const*) nullptr);
    else
        gemm<NT, false, 0, EPI_QKV, VIRTUAL, true, LazyGemmArgs>(ga, T, red, (unsigned int const*) nullptr);
}

// ---------------------------------------------------------------- the megakernel
template <int NT, int AMT>
__global__ void __launch_bounds__(NTHR, 1) gemma_kernel(const Params p)
{
    extern __shared__ __align__(16) char smem[];
    __shared__ float sred[32];
    float* red = (float*) smem;
    int const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int const T = p.T, Q = p.Q;
    constexpr int ATTN_KSTEP = AMT == 1 ? 32 : 64;
    unsigned int bar_target = 0;
    unsigned int const NCB = gridDim.x - NPF_OF(NT); // compute CTAs
#if NPF_SMALL > 0 || NPF_LARGE > 0
    if (blockIdx.x >= NCB)
    {
#ifdef PF_LEAN
#ifndef PF_HOLD_MINNT
#define PF_HOLD_MINNT 99
#endif
#ifndef PF_KVCAP_MID
#define PF_KVCAP_MID 0ull
#endif
        constexpr int pf_hold = NT >= PF_HOLD_MINNT ? 1 : 0;
        if (NT <= PF_ALL_MAXNT)
            bulk_prefetcher<true, true, false, false, (NT >= 3)>(p, NCB, (unsigned long long) PF_WIN_SMALL * 1048576ull,
                NT <= 1       ? PF_RATE_ALL_NT1
                    : NT <= 2 ? PF_RATE_ALL_NT2
                              : PF_RATE_ALL_NT3,
                1, pf_hold, NT >= 2 ? (unsigned long long) PF_KVCAP_MID : 0ull);
#ifdef PF_LARGE_ALL
        else if (NT > 3)
            bulk_prefetcher<true, true, false, false, true>(
                p, NCB, (unsigned long long) PF_WIN_LARGE * 1048576ull, PF_RATE_LARGE, PF_VT_LARGE, pf_hold);
#endif
        else
            bulk_prefetcher<(NT <= 3), false, (NT > 3 && PF_KVPROD_ALL), (NT >= 6)>(p, NCB,
                (unsigned long long) (NT <= 2 ? PF_WIN_SMALL
                        : NT <= 3             ? PF_WIN_NT3
                                              : PF_WIN_LARGE)
                    * 1048576ull,
                NT <= 3 ? PF_RATE_SMALL : PF_RATE_LARGE, p.R == 1 ? PF_VT_R1 : 1);
        return;
#endif
        weight_prefetcher(p, NCB,
            (unsigned long long) (NT <= 2 ? PF_WIN_SMALL
                    : NT <= 3             ? PF_WIN_NT3
                                          : PF_WIN_LARGE)
                * 1048576ull,
            NT <= 3 ? PF_RATE_SMALL : PF_RATE_LARGE);
        return;
    }
#endif

#ifndef PLE_LAZY_MINNT
#define PLE_LAZY_MINNT 99
#endif
#ifndef PLE_LAZY_MAXNT
#define PLE_LAZY_MAXNT 99
#endif
#ifndef PLE_LAZY_EXTRA_NT
#define PLE_LAZY_EXTRA_NT 99
#endif
#ifndef PLE_P3_MINNT
#define PLE_P3_MINNT 99
#endif
    constexpr bool ple_lazy = (NT >= PLE_LAZY_MINNT && NT <= PLE_LAZY_MAXNT) || NT == PLE_LAZY_EXTRA_NT;
    int const ple_e = p.nic - (511 + p.Q + ATTN_KSTEP - 1) / ATTN_KSTEP;
    bool const ple_p3 = NT >= PLE_P3_MINNT && ple_e > 0 && p.R * p.ng * ple_e >= 2 && p.R * p.ng * p.nic <= (int) NCB;
    if constexpr (ple_lazy)
    {
        grid_arrive(bar_target, NCB);
    }
    else
    {
        // ---------------- PLE projection GEMM
        {
            GemmArgs ga;
            ga.W = (uint8_t const*) p.ple_mp_w;
            ga.ldw = 3072;
            ga.X = (uint8_t const*) p.hidden;
            ga.ldx = 3072;
            ga.Kb = 3072;
            ga.mtiles = 8960 / 16;
            ga.S = 1;
            ga.gu_inter = 0;
            ga.epi = EPI_PLEMP;
            ga.sc = 1.f;
            ga.sc2 = 1.f;
            ga.out = g_pproj;
            ga.ldo = 8960;
            ga.aux = nullptr;
            gemm<NT, false, 0, EPI_PLEMP>(ga, T, red, (unsigned int const*) nullptr);
        }
        grid_sync(bar_target, NCB);
    }
    // ---------------- PLE norm/combine + layer-0 input
    {
        if constexpr (!ple_lazy)
            for (int pr = blockIdx.x * 8 + w; pr < T * NLAYER; pr += NCB * 8)
            {
                int const t = pr / NLAYER, l = pr - t * NLAYER;
                bf16 const* src = g_pproj + (size_t) t * 8960 + l * PLED;
                bf16 const* tk = p.ple_tok + ((size_t) t * NLAYER + l) * PLED;
                float x[8], pn[8], tv[8];
                float ss = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    int const d = lane + 32 * e;
                    x[e] = bf2f(__ldcg(src + d));
                    pn[e] = bf2f(p.ple_pn_w[d]);
                    tv[e] = bf2f(tk[d]);
                    ss += x[e] * x[e];
                }
                ss = warp_sum(ss);
                float const rs = rsqrtf(ss / 256.f + EPSV);
                bf16* dst = g_pli + ((size_t) t * NLAYER + l) * PLED;
#pragma unroll
                for (int e = 0; e < 8; ++e)
                {
                    int const d = lane + 32 * e;
                    float const y = bfr((x[e] * rs) * pn[e]);
                    float const s2 = bfr(y + tv[e]);
                    dst[d] = __float2bfloat16_rn(s2 * 0.70710678118654752f);
                }
            }
        for (int e = blockIdx.x * NTHR + tid; e < T * 192; e += NCB * NTHR)
        {
            int const t = e / 192, f = e - t * 192;
            int const r = t / Q;
            int const pos = p.cache_lens[r] + (t - r * Q);
            float inv;
            if (f < 128)
                inv = 1.0f / powf(10000.f, (float) (2 * f) / 256.f);
            else
                inv = 1.0f / powf(1000000.f, (float) (2 * (f - 128)) / 512.f);
            float const ang = (float) pos * inv;
            float sn, cs;
            sincosf(ang, &sn, &cs);
            g_rope[e] = make_float2(cs, sn);
        }
        float const xs0 = p.a_sc[1];
        for (int t = blockIdx.x; t < T; t += NCB)
        {
            float hv[6], wv[6];
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < 6; ++e)
            {
                int const n = tid + 256 * e;
                hv[e] = bf2f(p.hidden[(size_t) t * HID + n]);
                wv[e] = bf2f(p.ln_w[n]);
                g_h[t * HID + n] = __float2bfloat16_rn(hv[e]);
                ss += hv[e] * hv[e];
            }
            ss = block_sum(ss, sred);
            float const rs = rsqrtf(ss / (float) HID + EPSV);
#pragma unroll
            for (int e = 0; e < 6; ++e)
            {
                int const n = tid + 256 * e;
                float const x = bfr((hv[e] * rs) * wv[e]);
                g_xa[t * HID + n] = fp8q(x / xs0);
            }
        }
    }
    grid_arrive(bar_target, NCB);

#pragma unroll 1
    for (int i = 0; i < NLAYER; ++i)
    {
        int kind, j;
        layer_kind(i, kind, j);
        bool const kvp = kind < 2;
        bool const sliding = (kind == 0 || kind == 2);
        int const hd = sliding ? 256 : 512;
        float const* sc = kind == 0 ? p.a_sc + j * 8
            : kind == 1             ? p.b_sc + j * 8
            : kind == 2             ? p.c_sc + j * 6
                                    : p.d_sc + j * 6;
        int const so = kvp ? 2 : 0; // offset of the o-proj scale pair
        bf16 const* nw = p.ln_w + (size_t) i * 5 * HID;
        int const inter = kvp ? 6144 : 12288;
        int const N1 = kvp ? 10 * hd : 8 * hd;

#if defined(KVPF_MASK) && KVPF_AT == 2
        if ((KVPF_MASK >> (kind & 1)) & 1)
            attn_kv_bulk_prefetch<ATTN_KSTEP>(p, i, NCB);
#endif
        if constexpr (ple_lazy)
        {
            if ((int) blockIdx.x < 16 && tid == 0)
                bulk_pf((uint8_t const*) (p.ple_mp_w + ((size_t) i * PLED + blockIdx.x * 16) * HID), 16u * HID * 2u);
        }
        // ---- P1: qkv / q projection
        run_gemm<NT, 0>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
        attn_l2_prefetch<ATTN_KSTEP>(
            p, i, NCB);                  // first key step of this block's attention items, overlapping the barrier
#ifdef PF_GU
        own_task_prefetch(p, i, 2, NCB); // this block's gate_up weights stream into L2 during attention
#endif
#ifdef ATTN_PREBAR_V
        unsigned int const attn_wneed = NCB;
#else
        if (attn_busy<ATTN_KSTEP>(p, i, NCB))
            grid_wait(bar_target, NCB);
        unsigned int const attn_wneed = 0;
#endif
        // ---- P3: attention partials
#ifdef ATTN_TIMING
        if (tid == 0)
            s_curlayer = i;
#endif
        {
            bf16 const* qn;
            bf16 const* kn = nullptr;
            uint8_t* kc;
            uint8_t* vc;
            long long kvps;
            kv_layer(p, kind, j, kc, vc, kvps);
            if (kind == 0)
            {
                qn = p.a_qkn + j * 512;
                kn = qn + 256;
            }
            else if (kind == 1)
            {
                qn = p.b_qkn + j * 1024;
                kn = qn + 512;
            }
            else if (kind == 2)
            {
                qn = p.c_qn + j * 256;
            }
            else
            {
                qn = p.d_qn + j * 512;
            }
            int const nitems = p.R * p.ng * p.nic;
#ifdef KV_PERSIST
            // KV read again by later KV-shared layers (layer 13 sliding / layer 14 full) stays in L2
            bool const kvlast = sliding ? ((KV_PERSIST & 1) && (kind == 2 || (kind == 0 && j == 11)))
                                        : ((KV_PERSIST & 2) && (kind == 3 || (kind == 1 && j == 2)));
#else
            bool const kvlast = false;
#endif
            for (int item = blockIdx.x; item < nitems; item += NCB)
            {
                bool did;
                if (sliding)
                {
                    if (kvp)
                        did = attn_item<256, AMT, true, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, attn_wneed, kvlast);
                    else
                        did = attn_item<256, AMT, false, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, attn_wneed, kvlast);
                }
                else
                {
                    if (kvp)
                        did = attn_item<512, AMT, true, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, attn_wneed, kvlast);
                    else
                        did = attn_item<512, AMT, false, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, attn_wneed, kvlast);
                }
#if !defined(PRUNE_TAIL_BARRIERS) || defined(ATTN_TWICE)
                __syncthreads();
#endif
#ifdef ATTN_TWICE
                if (sliding)
                {
                    if (kvp)
                        did |= attn_item<256, AMT, true, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, 0, kvlast);
                    else
                        did |= attn_item<256, AMT, false, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, 0, kvlast);
                }
                else
                {
                    if (kvp)
                        did |= attn_item<512, AMT, true, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, 0, kvlast);
                    else
                        did |= attn_item<512, AMT, false, (AMT == 2 || (NT == 3 && AMT == 1))>(
                            p, item, N1, qn, kn, kc, vc, kvps, smem, bar_target, 0, kvlast);
                }
                __syncthreads();
#endif
                if (did && tid == 0)
                {
                    int const rg = item / p.nic;
                    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_attn_ready[i * 32 + rg])
                                 : "memory");
                }
            }
            if constexpr (ple_lazy)
            {
                if (ple_p3 && sliding && (int) blockIdx.x < p.R * p.ng * p.nic)
                {
                    int const jj = (int) blockIdx.x % p.nic;
                    if (jj >= p.nic - ple_e)
                        ple_slice<NT, true>(p, i, T, red, ((int) blockIdx.x / p.nic) * ple_e + (jj - (p.nic - ple_e)),
                            p.R * p.ng * ple_e);
                }
            }
        }
        ATT_STAMP(i, 3);
        grid_arrive(bar_target, NCB);
        // ---- P4: combine partials -> fp8 o-proj input  (one block per (token, head, 128-wide d chunk);
        //      warp w folds partials w, w+8, ...; fixed-order cross-warp sum through smem)
        {
            float const xs_o = sc[so + 1];
            int const NG = p.ng, NIc = p.nic, HG = 8 / NG;
            int const ndq = hd >> 7;
            int const nunits = T * 8 * ndq;
            float* cred = (float*) smem; // [8][132]
            if (nunits > (int) NCB)
            {                            // many units: one warp per unit, sequential over partials
                for (int u = blockIdx.x * 8 + w; u < nunits; u += NCB * 8)
                {
                    int const t = u / (8 * ndq), rem = u - t * 8 * ndq;
                    int const h = rem / ndq, dq = rem - h * ndq;
                    int const r = t / Q, tl = t - r * Q;
                    int const hg = h / HG;
                    int const row = tl * HG + (h - hg * HG);
                    int const cl = p.cache_lens[r];
                    int const kend = cl + Q;
                    int const kstart = sliding ? max(0, cl - 511) : 0;
                    int const nk = kend - kstart;
                    int const csz = (((nk + NIc - 1) / NIc) + ATTN_KSTEP - 1) & ~(ATTN_KSTEP - 1);
                    int const nused = (nk + csz - 1) / csz;
                    int const ib = (r * NG + hg) * NIc;
                    attn_wait_warp(&g_attn_ready[i * 32 + r * NG + hg], nused);
                    float M = -1e30f;
#ifndef SINGLE_PASS_COMBINE
                    for (int it = lane; it < nused; it += 32)
                        M = fmaxf(M, __ldcg(&g_ml[(ib + it) * AROWS + row].x));
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
#endif
                    float O[4] = {0.f, 0.f, 0.f, 0.f}, Ls = 0.f;
                    __half const* opb = g_opart + ((size_t) ib * AROWS + row) * hd + dq * 128 + lane * 4;
#if defined(COMB_B) && defined(SINGLE_PASS_COMBINE)
                    if constexpr (NT <= COMB_B_MAXNT)
                    {
                        // Partials stream in predicated batches of COMB_B (all loads of a batch in flight, no remainder
                        // loop); accumulation order is unchanged.
#ifndef COMB_B_GE2
#define COMB_B_GE2 COMB_B
#endif
                        constexpr int CB = NT == 1 ? COMB_B : COMB_B_GE2;
                        for (int it0 = 0; it0 < nused; it0 += CB)
                        {
                            float4 vb[CB];
                            float2 mb[CB];
#pragma unroll
                            for (int k = 0; k < CB; ++k)
                            {
                                int const itc = min(it0 + k, nused - 1);
                                vb[k] = ldg_h4(opb + (size_t) itc * AROWS * hd);
                                mb[k] = __ldcg(&g_ml[(ib + itc) * AROWS + row]);
                            }
#pragma unroll
                            for (int k = 0; k < CB; ++k)
                            {
                                if (it0 + k >= nused)
                                    break;
                                const float4 v = vb[k];
                                const float2 ml = mb[k];
                                float wgt;
                                if (ml.x > M)
                                {
                                    float const a = expf(M - ml.x);
                                    O[0] *= a;
                                    O[1] *= a;
                                    O[2] *= a;
                                    O[3] *= a;
                                    Ls *= a;
                                    M = ml.x;
                                    wgt = 1.f;
                                }
                                else
                                {
                                    wgt = expf(ml.x - M);
                                }
                                O[0] += v.x * wgt;
                                O[1] += v.y * wgt;
                                O[2] += v.z * wgt;
                                O[3] += v.w * wgt;
                                Ls += ml.y * wgt;
                            }
                        }
                    }
                    else
#endif
                    {
#pragma unroll 4
                        for (int it = 0; it < nused; ++it)
                        {
                            const float4 v = ldg_h4(opb + (size_t) it * AROWS * hd);
                            const float2 ml = __ldcg(&g_ml[(ib + it) * AROWS + row]);
#ifdef SINGLE_PASS_COMBINE
                            float wgt;
                            if (ml.x > M)
                            {
                                float const a = expf(M - ml.x);
                                O[0] *= a;
                                O[1] *= a;
                                O[2] *= a;
                                O[3] *= a;
                                Ls *= a;
                                M = ml.x;
                                wgt = 1.f;
                            }
                            else
                            {
                                wgt = expf(ml.x - M);
                            }
#else
                            float const wgt = expf(ml.x - M);
#endif
                            O[0] += v.x * wgt;
                            O[1] += v.y * wgt;
                            O[2] += v.z * wgt;
                            O[3] += v.w * wgt;
                            Ls += ml.y * wgt;
                        }
                    }
                    uint32_t packed = 0;
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        packed |= (uint32_t) fp8q(bfr(O[q] / Ls) / xs_o) << (8 * q);
                    *(uint32_t*) (g_xb + (size_t) t * 8 * hd + h * hd + dq * 128 + lane * 4) = packed;
                }
            }
            else
                for (int u = blockIdx.x; u < nunits; u += NCB)
                {
                    int const t = u / (8 * ndq), rem = u - t * 8 * ndq;
                    int const h = rem / ndq, dq = rem - h * ndq;
                    int const r = t / Q, tl = t - r * Q;
                    int const hg = h / HG;
                    int const row = tl * HG + (h - hg * HG);
                    int const cl = p.cache_lens[r];
                    int const kend = cl + Q;
                    int const kstart = sliding ? max(0, cl - 511) : 0;
                    int const nk = kend - kstart;
                    int const csz = (((nk + NIc - 1) / NIc) + ATTN_KSTEP - 1) & ~(ATTN_KSTEP - 1);
                    int const nused = (nk + csz - 1) / csz;
                    int const ib = (r * NG + hg) * NIc;
                    attn_wait_cta(&g_attn_ready[i * 32 + r * NG + hg], nused);
#ifdef CTA_SP
                    if constexpr (NT <= COMB_B_MAXNT)
                    {
                        // Single pass: warp w loads its first three partials (w, w + 8, w + 16) in one batch and folds
                        // them against its own running max; warp 0 merges the eight (max, sum, O) states with
                        // rescaling.
                        float M = -1e30f;
                        float O[4] = {0.f, 0.f, 0.f, 0.f}, Ls = 0.f;
                        __half const* opb = g_opart + ((size_t) ib * AROWS + row) * hd + dq * 128 + lane * 4;
                        {
                            float4 vb[3];
                            float2 mb[3];
#pragma unroll
                            for (int k = 0; k < 3; ++k)
                            {
                                int const itc = min(w + 8 * k, nused - 1);
                                vb[k] = ldg_h4(opb + (size_t) itc * AROWS * hd);
                                mb[k] = __ldcg(&g_ml[(ib + itc) * AROWS + row]);
                            }
#pragma unroll
                            for (int k = 0; k < 3; ++k)
                                if (w + 8 * k < nused)
                                    M = fmaxf(M, mb[k].x);
#pragma unroll
                            for (int k = 0; k < 3; ++k)
                            {
                                if (w + 8 * k < nused)
                                {
                                    float const wgt = expf(mb[k].x - M);
                                    O[0] += vb[k].x * wgt;
                                    O[1] += vb[k].y * wgt;
                                    O[2] += vb[k].z * wgt;
                                    O[3] += vb[k].w * wgt;
                                    Ls += mb[k].y * wgt;
                                }
                            }
                            for (int it = w + 24; it < nused; it += 8)
                            { // more than 24 partials: online update
                                const float4 v = ldg_h4(opb + (size_t) it * AROWS * hd);
                                const float2 ml = __ldcg(&g_ml[(ib + it) * AROWS + row]);
                                float wgt = 1.f;
                                if (ml.x > M)
                                {
                                    float const a = expf(M - ml.x);
                                    O[0] *= a;
                                    O[1] *= a;
                                    O[2] *= a;
                                    O[3] *= a;
                                    Ls *= a;
                                    M = ml.x;
                                }
                                else
                                {
                                    wgt = expf(ml.x - M);
                                }
                                O[0] += v.x * wgt;
                                O[1] += v.y * wgt;
                                O[2] += v.z * wgt;
                                O[3] += v.w * wgt;
                                Ls += ml.y * wgt;
                            }
                        }
                        *(float4*) (cred + w * 132 + lane * 4) = make_float4(O[0], O[1], O[2], O[3]);
                        if (lane == 0)
                        {
                            cred[w * 132 + 128] = Ls;
                            cred[w * 132 + 129] = M;
                        }
                        __syncthreads();
                        if (w == 0)
                        {
                            float Mt = -1e30f;
#pragma unroll
                            for (int ww = 0; ww < 8; ++ww)
                                Mt = fmaxf(Mt, cred[ww * 132 + 129]);
                            float o4[4] = {0.f, 0.f, 0.f, 0.f};
                            float Lt = 0.f;
#pragma unroll
                            for (int ww = 0; ww < 8; ++ww)
                            {
                                float const sw = expf(cred[ww * 132 + 129] - Mt);
                                const float4 v = *(float4 const*) (cred + ww * 132 + lane * 4);
                                o4[0] += v.x * sw;
                                o4[1] += v.y * sw;
                                o4[2] += v.z * sw;
                                o4[3] += v.w * sw;
                                Lt += cred[ww * 132 + 128] * sw;
                            }
                            uint32_t packed = 0;
#pragma unroll
                            for (int q = 0; q < 4; ++q)
                                packed |= (uint32_t) fp8q(bfr(o4[q] / Lt) / xs_o) << (8 * q);
                            *(uint32_t*) (g_xb + (size_t) t * 8 * hd + h * hd + dq * 128 + lane * 4) = packed;
                        }
                        __syncthreads();
                        continue;
                    }
#endif
                    float M = -1e30f;
                    for (int it = lane; it < nused; it += 32)
                        M = fmaxf(M, __ldcg(&g_ml[(ib + it) * AROWS + row].x));
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
                    float O[4] = {0.f, 0.f, 0.f, 0.f}, Ls = 0.f;
                    __half const* opb = g_opart + ((size_t) ib * AROWS + row) * hd + dq * 128 + lane * 4;
#pragma unroll 2
                    for (int it = w; it < nused; it += 8)
                    {
                        const float4 v = ldg_h4(opb + (size_t) it * AROWS * hd);
                        const float2 ml = __ldcg(&g_ml[(ib + it) * AROWS + row]);
                        float const wgt = expf(ml.x - M);
                        O[0] += v.x * wgt;
                        O[1] += v.y * wgt;
                        O[2] += v.z * wgt;
                        O[3] += v.w * wgt;
                        Ls += ml.y * wgt;
                    }
                    *(float4*) (cred + w * 132 + lane * 4) = make_float4(O[0], O[1], O[2], O[3]);
                    if (lane == 0)
                        cred[w * 132 + 128] = Ls;
                    __syncthreads();
                    if (w == 0)
                    {
                        float o4[4] = {0.f, 0.f, 0.f, 0.f};
                        float Lt = 0.f;
#pragma unroll
                        for (int ww = 0; ww < 8; ++ww)
                        {
                            const float4 v = *(float4 const*) (cred + ww * 132 + lane * 4);
                            o4[0] += v.x;
                            o4[1] += v.y;
                            o4[2] += v.z;
                            o4[3] += v.w;
                            Lt += cred[ww * 132 + 128];
                        }
                        uint32_t packed = 0;
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            packed |= (uint32_t) fp8q(bfr(o4[q] / Lt) / xs_o) << (8 * q);
                        *(uint32_t*) (g_xb + (size_t) t * 8 * hd + h * hd + dq * 128 + lane * 4) = packed;
                    }
                    __syncthreads();
                }
        }
        grid_arrive(bar_target, NCB);
        // ---- P5: o-proj (split-K partials)
        int const So = hd == 256 ? 4 : 8;
        run_gemm<NT, 1>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
#ifdef PF_DN
        own_task_prefetch(p, i, 3, NCB); // down weights stream into L2 during the norm phase
#endif
        // ---- P6: h += norm(attn); x = quant(norm(h))   (one token per block; NCB >= T)
        {
            float const sco = sc[so] * sc[so + 1];
            float const xs_gu = sc[so + 3];
            int const t = (int) NCB - 1 - (int) blockIdx.x;
            float a[6], hp[6], w1[6], w2[6];
            float ss = 0.f;
            if (t < T)
            {
#pragma unroll
                for (int e = 0; e < 6; ++e)
                { // independent loads before the barrier wait
                    int const n = tid + 256 * e;
                    hp[e] = bf2f(__ldcg(&g_h[t * HID + n]));
                    w1[e] = bf2f(nw[HID + n]);
                    w2[e] = bf2f(nw[2 * HID + n]);
                }
            }
            if (t < T)
            {
                grid_wait(bar_target, NCB);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float v = 0.f;
                    if constexpr (NT >= LOADFIRST_MIN_NT)
                    {
                        if (So == 8)
                        {
                            float z[8];
#pragma unroll
                            for (int s = 0; s < 8; ++s)
                                z[s] = __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
#pragma unroll
                            for (int s = 0; s < 8; ++s)
                                v += z[s];
                        }
                        else
                        {
                            float z[4];
#pragma unroll
                            for (int s = 0; s < 4; ++s)
                                z[s] = __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
#pragma unroll
                            for (int s = 0; s < 4; ++s)
                                v += z[s];
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                            if (s < So)
                                v += __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
                    }
                    a[e] = bfr(v * sco);
                    ss += a[e] * a[e];
                }
                ss = block_sum(ss, sred);
                float const rs = rsqrtf(ss / (float) HID + EPSV);
                float hh[6];
                float ss2 = 0.f;
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float const r1 = bfr((a[e] * rs) * w1[e]);
                    hh[e] = bfr(hp[e] + r1);
                    g_h[t * HID + n] = __float2bfloat16_rn(hh[e]);
                    ss2 += hh[e] * hh[e];
                }
                ss2 = block_sum(ss2, sred + 8);
                float const rs2 = rsqrtf(ss2 / (float) HID + EPSV);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float const x = bfr((hh[e] * rs2) * w2[e]);
                    g_xa[t * HID + n] = fp8q(x / xs_gu);
                }
            }
        }
        if constexpr (ple_lazy)
        {
            if (!(ple_p3 && sliding) && (int) blockIdx.x < 32)
                ple_slice<NT>(p, i, T, red);
        }
        grid_arrive(bar_target, NCB);
        // ---- P7: gate_up + gelu*up -> fp8 down input
        run_gemm<NT, 2>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
        // ---- P8: down (split-K partials)
        int const Sd = kvp ? 4 : 8;
        run_gemm<NT, 3>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
#if defined(KVPF_MASK) && KVPF_AT == 1
        if (i + 1 < NLAYER)
        {
            int nk2, nj2;
            layer_kind(i + 1, nk2, nj2);
            if ((KVPF_MASK >> (nk2 & 1)) & 1)
                attn_kv_bulk_prefetch<ATTN_KSTEP>(p, i + 1, NCB);
        }
#endif
#ifdef PF_NX
        own_task_prefetch(p, i, 4, NCB);
        own_task_prefetch(p, i, 5, NCB);
        if (i + 1 < NLAYER)
        {
            own_task_prefetch(p, i + 1, 0, NCB);
            own_task_prefetch(p, i + 1, 1, NCB);
        }
#endif
        // ---- P9: h += norm(mlp)
        {
            float const scd = sc[so + 4] * sc[so + 5];
            int const t = (int) NCB - 1 - (int) blockIdx.x;
            float a[6], hp[6], w3[6];
            float ss = 0.f;
            float plx = 0.f, plw = 0.f, plt = 0.f;
            if (t < T)
            {
#pragma unroll
                for (int e = 0; e < 6; ++e)
                { // independent loads before the barrier wait
                    int const n = tid + 256 * e;
                    hp[e] = bf2f(__ldcg(&g_h[t * HID + n]));
                    w3[e] = bf2f(nw[3 * HID + n]);
                }
                if constexpr (ple_lazy)
                {
                    float const* ps = (float const*) g_pproj;
                    plx = __ldcg(ps + (size_t) t * PLED + tid) + __ldcg(ps + (size_t) (TMAX + t) * PLED + tid);
                    plx = bfr(bfr(plx) * 0.025515518153991442f);
                    plw = bf2f(p.ple_pn_w[tid]);
                    plt = bf2f(p.ple_tok[((size_t) t * NLAYER + i) * PLED + tid]);
                }
            }
            if (t < T)
            {
                grid_wait(bar_target, NCB);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float v = 0.f;
                    if constexpr (NT >= LOADFIRST_MIN_NT)
                    {
                        if (Sd == 8)
                        {
                            float z[8];
#pragma unroll
                            for (int s = 0; s < 8; ++s)
                                z[s] = __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
#pragma unroll
                            for (int s = 0; s < 8; ++s)
                                v += z[s];
                        }
                        else
                        {
                            float z[4];
#pragma unroll
                            for (int s = 0; s < 4; ++s)
                                z[s] = __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
#pragma unroll
                            for (int s = 0; s < 4; ++s)
                                v += z[s];
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                            if (s < Sd)
                                v += __ldcg(&g_ypart[(size_t) s * TMAX * HID + t * HID + n]);
                    }
                    a[e] = bfr(v * scd);
                    ss += a[e] * a[e];
                }
                ss = block_sum(ss, sred);
                float const rs = rsqrtf(ss / (float) HID + EPSV);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float const r1 = bfr((a[e] * rs) * w3[e]);
                    float const hh = bfr(hp[e] + r1);
                    g_h[t * HID + n] = __float2bfloat16_rn(hh);
                }
                if constexpr (ple_lazy)
                {
                    float const pss = block_sum(plx * plx, sred + 8);
                    float const prs = rsqrtf(pss / 256.f + EPSV);
                    float const py = bfr((plx * prs) * plw);
                    float const ps2 = bfr(py + plt);
                    g_pli[((size_t) t * NLAYER + i) * PLED + tid] = __float2bfloat16_rn(ps2 * 0.70710678118654752f);
                }
            }
        }
        grid_arrive(bar_target, NCB);
        // ---- P10: PLE gate GEMM (split-K) + gelu * pli
        run_gemm<NT, 4>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
        // ---- P11: PLE proj GEMM
        run_gemm<NT, 5>(p, i, T, smem, &bar_target);
        grid_arrive(bar_target, NCB);
        // ---- P12: h = (h + norm(ple)) * scalar ; next-layer input / final norm
        {
            float const lsc = p.lscalar[i];
            int nkind = 0, nj = 0;
            if (i + 1 < NLAYER)
                layer_kind(i + 1, nkind, nj);
            bf16 const* nw0 = (i + 1 < NLAYER) ? p.ln_w + (size_t) (i + 1) * 5 * HID : p.final_w;
            float xs_n = 1.f;
            if (i + 1 < NLAYER && nkind < 2)
                xs_n = (nkind == 0 ? p.a_sc + nj * 8 : p.b_sc + nj * 8)[1];
            int const t = (int) NCB - 1 - (int) blockIdx.x;
            float a[6], hp[6], w4[6], w0[6];
            float ss = 0.f;
            if (t < T)
            {
#pragma unroll
                for (int e = 0; e < 6; ++e)
                { // independent loads before the barrier wait
                    int const n = tid + 256 * e;
                    hp[e] = bf2f(__ldcg(&g_h[t * HID + n]));
                    w4[e] = bf2f(nw[4 * HID + n]);
                    w0[e] = bf2f(nw0[n]);
                }
            }
            if (t < T)
            {
                grid_wait(bar_target, NCB);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    a[e] = bfr(__half2float(__ldcg(&g_ypart_h[t * HID + n])));
                    ss += a[e] * a[e];
                }
                ss = block_sum(ss, sred);
                float const rs = rsqrtf(ss / (float) HID + EPSV);
                float hh[6];
                float ss2 = 0.f;
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float const r1 = bfr((a[e] * rs) * w4[e]);
                    float hv = bfr(hp[e] + r1);
                    hv = bfr(hv * lsc);
                    hh[e] = hv;
                    g_h[t * HID + n] = __float2bfloat16_rn(hv);
                    ss2 += hv * hv;
                }
                ss2 = block_sum(ss2, sred + 8);
                float const rs2 = rsqrtf(ss2 / (float) HID + EPSV);
#pragma unroll
                for (int e = 0; e < 6; ++e)
                {
                    int const n = tid + 256 * e;
                    float const x = bfr((hh[e] * rs2) * w0[e]);
                    if (i + 1 == NLAYER)
                        p.out[(size_t) t * HID + n] = __float2bfloat16_rn(x);
                    else if (nkind < 2)
                        g_xa[t * HID + n] = fp8q(x / xs_n);
                    else
                        g_xbf[t * HID + n] = __float2bfloat16_rn(x);
                }
            }
        }
        if (i + 1 < NLAYER)
            grid_arrive(bar_target, NCB);
    }
    kernel_exit();
}

// ---------------------------------------------------------------- host
template <int NT, int AMT>
static cudaError_t launch(Params const& prm, int nsm, cudaStream_t stream)
{
#ifdef FORCE_SMEM
    static constexpr int smem_bytes = FORCE_SMEM;
#else
    static constexpr int smem_gemm = 2 * 8 * GemmRed<NT>::TS * sizeof(float);
    static constexpr int smem_attn = AttnSmem<512, AMT, (AMT == 2 || (NT == 3 && AMT == 1))>::BYTES;
    static constexpr int smem_wt = (NT >= WT_MIN_NT && NT <= 6) ? NT * 8 * WT_XS : 0;
    static constexpr int smem_ag = smem_attn > smem_gemm ? smem_attn : smem_gemm;
    static constexpr int smem_bytes = smem_ag > smem_wt ? smem_ag : smem_wt;
#endif
    static bool configured = false;
    if (!configured)
    {
        cudaError_t const status
            = cudaFuncSetAttribute(gemma_kernel<NT, AMT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (status != cudaSuccess)
            return status;
        configured = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(nsm + NPF_OF(NT));
    cfg.blockDim = dim3(NTHR);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, gemma_kernel<NT, AMT>, prm);
}

template <int NT>
static cudaError_t launch_attn_variant(Params const& prm, int nsm, cudaStream_t stream)
{
    int const rows = prm.Q * (8 / prm.ng);
    if (rows <= 16)
        return launch<NT, 1>(prm, nsm, stream);
    return launch<NT, 2>(prm, nsm, stream);
}

} // namespace gemma4_e2b_decode

cudaError_t launchGemma4E2bDecodeStack(gemma4_e2b_decode::Params& prm, int numSms, cudaStream_t stream)
{
    using namespace gemma4_e2b_decode;
    if (prm.R < 1 || prm.T % prm.R != 0 || prm.T > TMAX || prm.T / prm.R > 8)
        return cudaErrorInvalidValue;
    prm.Q = prm.T / prm.R;
    prm.L = 0;
    int const nsm_dev = numSms > MAXITEMS ? MAXITEMS : numSms;
    int nsm = prm.T <= NSM_T_SPLIT ? NSM_SMALL : NSM_LARGE;
    if (nsm > nsm_dev - NPF_OF((prm.T + 7) / 8))
        nsm = nsm_dev - NPF_OF((prm.T + 7) / 8);
    if (nsm < prm.T)
        nsm = prm.T;
    prm.ng = (prm.Q * 8 > 32) ? (prm.R <= 4 ? 4 : 2) : 1;
    prm.nic = nsm / (prm.R * prm.ng);
    if (prm.nic < 1)
        prm.nic = 1;
    int const nt = (prm.T + 7) / 8;
    if (nt <= 1)
        return launch_attn_variant<1>(prm, nsm, stream);
    if (nt <= 2)
        return launch_attn_variant<2>(prm, nsm, stream);
    if (nt <= 3)
        return launch_attn_variant<3>(prm, nsm, stream);
    if (nt <= 4)
        return launch_attn_variant<4>(prm, nsm, stream);
    if (nt <= 5)
        return launch_attn_variant<5>(prm, nsm, stream);
    if (nt <= 6)
        return launch_attn_variant<6>(prm, nsm, stream);
    return launch_attn_variant<8>(prm, nsm, stream);
}

TRTLLM_MV2_NAMESPACE_END
