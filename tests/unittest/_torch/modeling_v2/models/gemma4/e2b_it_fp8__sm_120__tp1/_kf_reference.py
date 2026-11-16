# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off
# ruff: noqa
"""Kernel Factory reference of the decoder-stack megakernel (definition gemma4_e2b_fp8_decoder_stack_spec_verify),
kept verbatim as the independent oracle of test_decode_stack.py: contiguous per-request caches, input generator
and statistical checker.

Gemma-4 E2B-it text decoder stack, one MTP spec-verify step (target forward only), FP8 checkpoint semantics.

Semantics follow HF transformers 5.5.4 models/gemma4/modeling_gemma4.py and TRT-LLM 1.4.0rc3
_torch/models/modeling_gemma4.py with the ModelOpt FP8 checkpoint (E2B-fp8-lmhead_fp8-textonly):

  per spec step: R requests, each with Q = k+1 new tokens at positions cache_lens[r] + j (j < Q).
  Input  hidden_states      = embed_tokens(ids) * sqrt(1536)                      [T=R*Q, 1536] bf16
         ple_token_embeds   = embed_tokens_per_layer(ids) * sqrt(256), reshaped   [T, 35, 256]  bf16
  Output out_hidden         = final RMSNorm(last layer output)                    [T, 1536]     bf16
         (+ new K/V of the 15 KV-producing layers written in place into the FP8 caches)
  lm_head (FP8, 262144 x 1536) + final_logit_softcapping are NOT part of this op.

Layer kinds (35 layers, every 5th = full attention, layers >= 15 are KV-shared, double-wide MLP):
  A: KV-producing sliding  layers 0-3,5-8,10-13 (12)  head_dim 256, fused FP8 QKV [2048+256+256]
  B: KV-producing full     layers 4,9,14          (3)   head_dim 512, fused FP8 QKV [4096+512+512]
  C: KV-shared sliding     layers 15-18,20-23,...,30-33 (16) BF16 q_proj, reads KV of layer 13
  D: KV-shared full        layers 19,24,29,34     (4)   BF16 q_proj, reads KV of layer 14
FP8 linear (ModelOpt per-tensor, TRT-LLM FP8QDQLinearMethod): xq = fp8_e4m3(clamp(x / in_scale, +-448)),
  y = bf16((xq @ Wq^T) * in_scale * w_scale). Fused qkv / gate_up use one (max) scale pair as TRT-LLM does.
"""
import math
import torch
import torch.nn.functional as F

H = 1536
NL = 35
PLE = 256
NH = 8
HD_S = 256
HD_F = 512
INTER = 6144
WIN = 512                   # HF: key visible iff 0 <= q_pos - k_pos < 512
EPS = 1e-6
FP8_MAX = 448.0
ROPE_THETA_S = 10000.0
ROPE_THETA_F = 1000000.0
ROPE_ANGLES_F = 64          # proportional rope: int(0.25 * 512 // 2)
LAYERS_A = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
LAYERS_B = [4, 9, 14]
LAYERS_C = [15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33]
LAYERS_D = [19, 24, 29, 34]
SRC_A = 11                  # layer 13 = LAYERS_A[11]  (KV source of all C layers)
SRC_B = 2                   # layer 14 = LAYERS_B[2]   (KV source of all D layers)
BF = torch.bfloat16
F8 = torch.float8_e4m3fn

def _kind(i):
    if i < 15:
        return ("A", LAYERS_A.index(i)) if i % 5 != 4 else ("B", LAYERS_B.index(i))
    return ("C", LAYERS_C.index(i)) if i % 5 != 4 else ("D", LAYERS_D.index(i))


def _rms(x, w=None):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        y = y * w.float()
    return y.to(BF)


def _q8(x, s):
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(F8)


def _lin8(x, w8, ws, xs, calib=None, key=None):
    if calib is not None:                       # calibration pass: dynamic per-tensor amax
        a = x.float().abs().max().clamp_min(1e-12)
        calib[key] = torch.maximum(calib.get(key, a), a)
        xs = a / FP8_MAX
    y = _q8(x, xs).float() @ w8.float().t()
    return (y * (xs * ws)).to(BF)


def _lin16(x, w):
    return (x.float() @ w.float().t()).to(BF)


def _rope(x, pos, head_dim, sliding):
    # x [T, nh, hd] bf16 ; neox rotate_half over the full head (proportional rope for full layers)
    half = head_dim // 2
    if sliding:
        inv = 1.0 / (ROPE_THETA_S ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    else:
        inv = torch.zeros(half, device=x.device, dtype=torch.float32)
        inv[:ROPE_ANGLES_F] = 1.0 / (ROPE_THETA_F ** (torch.arange(0, 2 * ROPE_ANGLES_F, 2, device=x.device,
                                                                    dtype=torch.float32) / head_dim))
    ang = pos.float()[:, None] * inv[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    xf = x.float()
    rot = torch.cat([-xf[..., half:], xf[..., :half]], -1)
    return (xf * cos + rot * sin).to(BF)


def _attend(q, kc, vc, cache_lens, Q, sliding, window):
    # q [T, NH, hd] bf16 ; kc/vc [R, L, hd] fp8 (already holding the new tokens) ; scaling = 1.0.
    # Dense masked form (no host reads of cache_lens -> CUDA-graph capturable).
    R, L, hd = kc.shape
    k = kc.float()
    v = vc.float()
    qr = q.float().reshape(R, Q, NH, hd)
    s = torch.einsum("rqhd,rkd->rhqk", qr, k)
    qp = cache_lens.to(torch.long)[:, None] + torch.arange(Q, device=q.device)[None]      # [R, Q]
    kp = torch.arange(L, device=q.device)
    m = kp[None, None, :] <= qp[:, :, None]                                                 # [R, Q, L]
    if sliding:
        m = m & ((qp[:, :, None] - kp[None, None, :]) < window)
    s = s.masked_fill(~m[:, None], float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("rhqk,rkd->rqhd", p, v)
    return o.reshape(R * Q, NH * hd).to(BF)


def _forward(hidden_states, ple_token_embeds, cache_lens,
             k_cache_swa, v_cache_swa, k_cache_full, v_cache_full,
             ple_model_proj_w, ple_proj_norm_w, layer_norm_w, layer_scalar, ple_gate_w, ple_proj_w, final_norm_w,
             a_qkv_w, a_o_w, a_gu_w, a_down_w, a_qk_norm_w, a_scales,
             b_qkv_w, b_o_w, b_gu_w, b_down_w, b_qk_norm_w, b_scales,
             c_q_w, c_o_w, c_gu_w, c_down_w, c_q_norm_w, c_scales,
             d_q_w, d_o_w, d_gu_w, d_down_w, d_q_norm_w, d_scales,
             calib=None, window=WIN):
    T = hidden_states.shape[0]
    R = cache_lens.shape[0]
    Q = T // R
    dev = hidden_states.device
    pos = (cache_lens.to(torch.long)[:, None] + torch.arange(Q, device=dev)[None]).reshape(-1)
    ridx = torch.arange(R, device=dev).repeat_interleave(Q)
    G = {"A": (a_qkv_w, a_o_w, a_gu_w, a_down_w, a_qk_norm_w, a_scales, HD_S, True),
         "B": (b_qkv_w, b_o_w, b_gu_w, b_down_w, b_qk_norm_w, b_scales, HD_F, False),
         "C": (c_q_w, c_o_w, c_gu_w, c_down_w, c_q_norm_w, c_scales, HD_S, True),
         "D": (d_q_w, d_o_w, d_gu_w, d_down_w, d_q_norm_w, d_scales, HD_F, False)}

    # Per-layer embeddings (PLE): project main embedding, norm, combine with token PLE.
    proj = (_lin16(hidden_states, ple_model_proj_w) * (H ** -0.5)).reshape(T, NL, PLE)
    proj = _rms(proj, ple_proj_norm_w)
    pli = (proj + ple_token_embeds) * (2.0 ** -0.5)

    h = hidden_states
    for i in range(NL):
        kind, j = _kind(i)
        qkv_w, o_w, gu_w, down_w, qkn, sc, hd, sliding = G[kind]
        sc = sc[j]
        ck = lambda name: (i, name)
        nw = layer_norm_w[i]
        residual = h
        x = _rms(h, nw[0])
        if kind in "AB":
            qkv = _lin8(x, qkv_w[j], sc[0, 0], sc[0, 1], calib, ck("qkv"))
            q = qkv[:, :NH * hd].reshape(T, NH, hd)
            k = qkv[:, NH * hd:NH * hd + hd].reshape(T, 1, hd)
            v = qkv[:, NH * hd + hd:].reshape(T, 1, hd)
            v = _rms(v)
            q = _rope(_rms(q, qkn[j, 0]), pos, hd, sliding)
            k = _rope(_rms(k, qkn[j, 1]), pos, hd, sliding)
            kc = (k_cache_swa if sliding else k_cache_full)[j]
            vc = (v_cache_swa if sliding else v_cache_full)[j]
            kc[ridx, pos] = k[:, 0].to(F8)
            vc[ridx, pos] = v[:, 0].to(F8)
        else:
            q = _lin16(x, qkv_w[j]).reshape(T, NH, hd)
            q = _rope(_rms(q, qkn[j]), pos, hd, sliding)
            src = SRC_A if sliding else SRC_B
            kc = (k_cache_swa if sliding else k_cache_full)[src]
            vc = (v_cache_swa if sliding else v_cache_full)[src]
        a = _attend(q, kc, vc, cache_lens, Q, sliding, window)
        o_row = 1 if kind in "AB" else 0
        a = _lin8(a, o_w[j], sc[o_row, 0], sc[o_row, 1], calib, ck("o"))
        h = residual + _rms(a, nw[1])
        residual = h
        x = _rms(h, nw[2])
        gu = _lin8(x, gu_w[j], sc[o_row + 1, 0], sc[o_row + 1, 1], calib, ck("gu"))
        inter = gu.shape[-1] // 2
        act = (F.gelu(gu[:, :inter].float(), approximate="tanh") * gu[:, inter:].float()).to(BF)
        d = _lin8(act, down_w[j], sc[o_row + 2, 0], sc[o_row + 2, 1], calib, ck("down"))
        h = residual + _rms(d, nw[3])
        residual = h
        g = F.gelu(_lin16(h, ple_gate_w[i]), approximate="tanh") * pli[:, i]
        p = _lin16(g, ple_proj_w[i])
        h = residual + _rms(p, nw[4])
        h = (h.float() * layer_scalar[i].float()).to(BF)
    return _rms(h, final_norm_w)


@torch.no_grad()
def run(hidden_states, ple_token_embeds, cache_lens,
        k_cache_swa, v_cache_swa, k_cache_full, v_cache_full,
        ple_model_proj_w, ple_proj_norm_w, layer_norm_w, layer_scalar, ple_gate_w, ple_proj_w, final_norm_w,
        a_qkv_w, a_o_w, a_gu_w, a_down_w, a_qk_norm_w, a_scales,
        b_qkv_w, b_o_w, b_gu_w, b_down_w, b_qk_norm_w, b_scales,
        c_q_w, c_o_w, c_gu_w, c_down_w, c_q_norm_w, c_scales,
        d_q_w, d_o_w, d_gu_w, d_down_w, d_q_norm_w, d_scales):
    torch.backends.cuda.matmul.allow_tf32 = False
    return _forward(hidden_states, ple_token_embeds, cache_lens,
                    k_cache_swa, v_cache_swa, k_cache_full, v_cache_full,
                    ple_model_proj_w, ple_proj_norm_w, layer_norm_w, layer_scalar, ple_gate_w, ple_proj_w,
                    final_norm_w,
                    a_qkv_w, a_o_w, a_gu_w, a_down_w, a_qk_norm_w, a_scales,
                    b_qkv_w, b_o_w, b_gu_w, b_down_w, b_qk_norm_w, b_scales,
                    c_q_w, c_o_w, c_gu_w, c_down_w, c_q_norm_w, c_scales,
                    d_q_w, d_o_w, d_gu_w, d_down_w, d_q_norm_w, d_scales)


INPUT_NAMES = ["hidden_states", "ple_token_embeds", "cache_lens",
               "k_cache_swa", "v_cache_swa", "k_cache_full", "v_cache_full",
               "ple_model_proj_w", "ple_proj_norm_w", "layer_norm_w", "layer_scalar", "ple_gate_w", "ple_proj_w",
               "final_norm_w",
               "a_qkv_w", "a_o_w", "a_gu_w", "a_down_w", "a_qk_norm_w", "a_scales",
               "b_qkv_w", "b_o_w", "b_gu_w", "b_down_w", "b_qk_norm_w", "b_scales",
               "c_q_w", "c_o_w", "c_gu_w", "c_down_w", "c_q_norm_w", "c_scales",
               "d_q_w", "d_o_w", "d_gu_w", "d_down_w", "d_q_norm_w", "d_scales"]


@torch.no_grad()
def generate_inputs(axes_and_scalars, device, seed, ctx_lens=None, **kwargs):
    """Random weights with the real checkpoint's shapes, dtypes and FP8 weight-code statistics, in a numerically
    well-conditioned regime (see below). Static FP8 input scales are calibrated on the generated batch exactly
    like ModelOpt (amax / 448). Past KV is random FP8 with the real K/V magnitudes. ctx_lens (custom_inputs_kwargs)
    gives each request's cache length; default: max_ctx - Q."""
    R = int(axes_and_scalars["num_requests"])
    Q = int(axes_and_scalars["tokens_per_request"])
    L = int(axes_and_scalars["max_ctx"])
    T = R * Q
    g = torch.Generator(device=device).manual_seed(int(seed))
    if ctx_lens is None:
        ctx_lens = [L - Q] * R
    assert len(ctx_lens) == R and all(1 <= c <= L - Q for c in ctx_lens), (ctx_lens, R, L, Q)

    def randn(*shape, std=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g, device=device) * std).to(dtype)

    def fp8w(*shape):
        # ModelOpt per-tensor FP8 weights use the full e4m3 range (amax -> 448); real std ~20-60, heavy tails.
        std = 18.0 + 22.0 * torch.rand(shape[0], 1, 1, generator=g, device=device)
        t = torch.empty(*shape, device=device, dtype=F8)
        for n in range(shape[0]):
            w = torch.randn(*shape[1:], generator=g, device=device)
            w = w * (1.0 + 0.6 * w.abs())          # heavier tails than gaussian
            t[n] = (w * std[n]).clamp(-FP8_MAX, FP8_MAX).to(F8)
        return t

    def norm_w(n, mean, std):
        return (mean + std * torch.randn(n, generator=g, device=device)).clamp(mean - 3 * std, mean + 3 * std)

    def unit_gain(w8, rows):
        # weight_scale such that the dequantized weight has ~unit gain (std 1/sqrt(K)); lands in the real
        # checkpoint's range (5e-4..2.3e-3) for the generated FP8 code distribution.
        n, _, K = w8.shape
        sd = torch.stack([w8[i].float().std() for i in range(n)])
        return (0.8 + 0.4 * torch.rand(n, generator=g, device=device)) / (sd * math.sqrt(K))

    # Numerically well-conditioned regime: pre-norm weights ~1, small post-branch norms (residual-dominated
    # stream), layer_scalar ~0.9-1.  Random weights with the real checkpoint's norm/scalar values make the
    # 35-layer stack chaotic (any 1-ulp difference decorrelates the output), which would make correctness
    # checking meaningless.
    inp = {}
    inp["hidden_states"] = randn(T, H, std=0.0316 * math.sqrt(H), dtype=BF)
    inp["ple_token_embeds"] = randn(T, NL, PLE, std=0.0687 * math.sqrt(PLE), dtype=BF)
    inp["cache_lens"] = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    knorm_a = torch.tensor([0.127, 0.122, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.126], device=device)
    knorm_b = torch.tensor([0.065, 0.062, 0.061], device=device)
    # Past KV (random beyond cache_len + Q as well; only [0, cache_len) is meaningful): K ~ k_norm weight
    # magnitude (real 0.06-0.13), V ~ unit RMS (v_norm has no scale).
    inp["k_cache_swa"] = (randn(12, R, L, HD_S) * knorm_a[:, None, None, None]).to(F8)
    inp["v_cache_swa"] = randn(12, R, L, HD_S).to(F8)
    inp["k_cache_full"] = (randn(3, R, L, HD_F) * knorm_b[:, None, None, None]).to(F8)
    inp["v_cache_full"] = randn(3, R, L, HD_F).to(F8)
    inp["ple_model_proj_w"] = randn(NL * PLE, H, std=0.0343, dtype=BF)
    inp["ple_proj_norm_w"] = norm_w(PLE, 1.0, 0.1).to(BF)
    ln = torch.empty(NL, 5, H, device=device)
    stats = [(1.0, 0.1), (0.15, 0.05), (1.0, 0.1), (0.15, 0.05), (0.15, 0.05)]
    for i in range(NL):
        for s_, (m, sd) in enumerate(stats):
            ln[i, s_] = norm_w(H, m, sd)
    inp["layer_norm_w"] = ln.to(BF)
    inp["layer_scalar"] = 0.9 + 0.1 * torch.rand(NL, generator=g, device=device)
    inp["ple_gate_w"] = randn(NL, PLE, H, std=0.0255, dtype=BF)
    inp["ple_proj_w"] = randn(NL, H, PLE, std=0.0625, dtype=BF)
    inp["final_norm_w"] = norm_w(H, 1.0, 0.1).to(BF)

    def mk(prefix, n, qkv_rows, q_bf16_rows, o_in, gu_rows, down_in, nrows):
        sc = torch.ones(n, nrows, 2, device=device)
        r0 = 0
        if qkv_rows:
            inp[prefix + "_qkv_w"] = w = fp8w(n, qkv_rows, H); sc[:, 0, 0] = unit_gain(w, qkv_rows); r0 = 1
        else:
            inp[prefix + "_q_w"] = randn(n, q_bf16_rows, H, std=1.0 / math.sqrt(H), dtype=BF)
        for r, (name, rows_, k) in enumerate([("_o_w", H, o_in), ("_gu_w", gu_rows, H), ("_down_w", H, down_in)]):
            inp[prefix + name] = w = fp8w(n, rows_, k); sc[:, r0 + r, 0] = unit_gain(w, rows_)
        inp[prefix + "_scales"] = sc

    mk("a", 12, NH * HD_S + 2 * HD_S, None, NH * HD_S, 2 * INTER, INTER, 4)
    mk("b", 3, NH * HD_F + 2 * HD_F, None, NH * HD_F, 2 * INTER, INTER, 4)
    mk("c", 16, None, NH * HD_S, NH * HD_S, 4 * INTER, 2 * INTER, 3)
    mk("d", 4, None, NH * HD_F, NH * HD_F, 4 * INTER, 2 * INTER, 3)
    inp["a_qk_norm_w"] = torch.stack([torch.full((12, HD_S), 0.99, device=device),
                                      knorm_a[:, None].expand(12, HD_S)], 1).to(BF)
    inp["b_qk_norm_w"] = torch.stack([torch.full((3, HD_F), 1.0, device=device),
                                      knorm_b[:, None].expand(3, HD_F)], 1).to(BF)
    inp["c_q_norm_w"] = torch.full((16, HD_S), 0.99, device=device, dtype=BF)
    inp["d_q_norm_w"] = torch.full((4, HD_F), 1.0, device=device, dtype=BF)

    # ModelOpt-style static input-scale calibration (amax/448) on this batch, on a scratch copy of the cache.
    calib = {}
    args = [inp[n] for n in INPUT_NAMES]
    for ci in range(3, 7):
        args[ci] = args[ci].clone()
    _forward(*args, calib=calib)
    for i in range(NL):
        kind, j = _kind(i)
        sc = inp[{"A": "a_scales", "B": "b_scales", "C": "c_scales", "D": "d_scales"}[kind]]
        names = ["qkv", "o", "gu", "down"] if kind in "AB" else ["o", "gu", "down"]
        for r, nm in enumerate(names):
            sc[j, r, 1] = calib[(i, nm)] / FP8_MAX
    return inp


def check_outputs(ref_outputs, user_outputs, *, ctx_lens, tokens_per_request,
                  hidden_rel_l2=0.35, hidden_mean_cos=0.95, hidden_min_cos=0.90,
                  kv_layer0_rel_l2=0.06, kv_rel_l2=0.15, **kwargs):
    """Statistical check. FP8 static activation quantization makes the 35-layer output chaotic at the 1-ulp level:
    legitimate re-associations (bf16 softmax probabilities, fused norm+quant, fp32 intermediates, GEMM
    accumulation order) move out_hidden by 14-24% relative L2 (min per-token cosine >= 0.959) on generated
    inputs and ~10% on the real checkpoint, so element-wise tolerances cannot be used.
      out_hidden : finite; relative L2 <= hidden_rel_l2; mean / min per-token cosine >= hidden_mean_cos / hidden_min_cos
      KV caches  : rows outside [ctx, ctx+Q) bit-identical to the input; new rows finite; relative L2 of the new
                   rows per layer <= kv_layer0_rel_l2 for layer 0 (no upstream noise) and <= kv_rel_l2 otherwise."""
    res = {}
    r = ref_outputs["out_hidden"].float()
    u = user_outputs["out_hidden"].float()
    fin = bool(torch.isfinite(u).all())
    if fin:
        rel = ((u - r).norm() / r.norm().clamp_min(1e-12)).item()
        cos = F.cosine_similarity(u, r, dim=-1)
        mc, mn = cos.mean().item(), cos.min().item()
    else:
        rel, mc, mn = float("inf"), -1.0, -1.0
    res["out_hidden"] = {"passed": fin and rel <= hidden_rel_l2 and mc >= hidden_mean_cos and mn >= hidden_min_cos,
                         "metadata": {"rel_l2": rel, "mean_token_cos": mc, "min_token_cos": mn, "finite": fin}}
    Q = int(tokens_per_request)
    for name in ["k_cache_swa", "v_cache_swa", "k_cache_full", "v_cache_full"]:
        rc, uc = ref_outputs[name], user_outputs[name]
        newmask = torch.zeros(rc.shape[1], rc.shape[2], dtype=torch.bool, device=rc.device)
        for b, c in enumerate(ctx_lens):
            newmask[b, c:c + Q] = True
        old_ok = bool((rc.view(torch.uint8) == uc.view(torch.uint8))[:, ~newmask].all())
        rels, ok = [], old_ok
        for l in range(rc.shape[0]):
            rn, un = rc[l][newmask].float(), uc[l][newmask].float()
            if not bool(torch.isfinite(un).all()):
                rels.append(float("inf")); ok = False; continue
            e = ((un - rn).norm() / rn.norm().clamp_min(1e-12)).item()
            rels.append(e)
            lim = kv_layer0_rel_l2 if (l == 0 and name.endswith("swa")) else kv_rel_l2
            ok = ok and e <= lim
        res[name] = {"passed": ok, "metadata": {"old_rows_identical": old_ok, "new_rows_rel_l2_per_layer": rels}}
    return res
