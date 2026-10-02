# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise tests for VLLM_MAMBA2_FUSED_DECODE (mamba2_fused_decode.cu).

Stock reference = what MambaMixer2 runs for a decode batch (align cache mode,
FlashInfer SSU backend): vLLM's Triton causal_conv1d_update (in place) ->
flashinfer.mamba.selective_state_update(rand_seed, philox_rounds,
algorithm="auto") -> the Inductor gated-RMSNorm + FP8-quant kernel vLLM
generates (inductor_gated_rmsnorm_fp8_ref.py). Compared bit for bit: the FP8
out_proj input, the SSM output, projected_states (conv output written in
place) and the whole raw conv/SSM cache buffers in vLLM's page layout.
"""

import importlib.util
import os

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(100)),
    reason="needs an SM100 (B200) GPU",
)

DEV = "cuda"
BF16, FP16, FP8 = torch.bfloat16, torch.float16, torch.float8_e4m3fn
NH, HD, NS, NG, XD, CD, PW = 32, 64, 128, 2, 2048, 2560, 4640
CPU_GEN = torch.Generator().manual_seed(2024)


def _fd():
    from vllm.model_executor.layers.mamba.ops import mamba2_fused_decode as fd

    return fd


def _null():
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    return NULL_BLOCK_ID


_REF = {}


def inductor_norm_quant(ssm_out, proj, weight, scale):
    if "k" not in _REF:
        path = os.path.join(os.path.dirname(__file__), "inductor_gated_rmsnorm_fp8_ref.py")
        spec = importlib.util.spec_from_file_location("inductor_gated_rmsnorm_fp8_ref", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REF["k"] = mod.triton_per_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_silu_view_0
    m = ssm_out.shape[0]
    out = torch.empty(m, XD, dtype=FP8, device=DEV)
    _REF["k"].run(ssm_out, proj, weight, scale.reshape(1), out, m, 2 * m, 1024,
                  stream=torch.cuda.current_stream().cuda_stream)
    return out


def stock_chain(proj, conv_cache, ssm_cache, sidx, cu, nacc, w, seed, ntok, nrows):
    from flashinfer.mamba import selective_state_update as fi_ssu

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    xbc, dt = proj[:, XD:XD + CD], proj[:, XD + CD:]
    hbc = causal_conv1d_update(xbc[:ntok], conv_cache.transpose(-1, -2), w["conv_w"], w["conv_b"], "silu",
                               conv_state_indices=sidx, num_accepted_tokens=nacc, query_start_loc=cu,
                               max_query_len=sidx.size(-1))
    hs, b, c = torch.split(hbc, [XD, 256, 256], dim=-1)
    y = torch.zeros(nrows, XD, dtype=BF16, device=DEV)
    fi_ssu(ssm_cache, hs.view(-1, NH, HD), dt[:ntok][:, :, None].expand(-1, -1, HD),
           w["A"][:, None, None].expand(-1, HD, NS).float(), b.view(-1, NG, NS), c.view(-1, NG, NS),
           D=w["D"][:, None].expand(-1, HD), z=None, dt_bias=w["dt_bias"][:, None].expand(-1, HD),
           dt_softplus=True, state_batch_indices=sidx, dst_state_batch_indices=sidx, cu_seqlens=cu,
           num_accepted_tokens=nacc, cache_steps=sidx.size(-1) if cu is not None else 0,
           pad_slot_id=_null(), out=y[:ntok].view(ntok, -1, HD), rand_seed=seed if w["philox"] else None,
           philox_rounds=w["philox"] or 10, algorithm="auto")
    return inductor_norm_quant(y, proj, w["norm_w"], w["scale"]), y


def fused_chain(proj, conv_cache, ssm_cache, sidx, cu, nacc, w, seed, ntok, nrows, hpc=1, xq=None, y=None):
    fd = _fd()
    xq = xq if xq is not None else torch.zeros(nrows, XD, dtype=FP8, device=DEV)
    spec = cu is not None
    args = fd.pack_args(proj=proj, conv_w=w["conv_w"], conv_b=w["conv_b"], conv_state=conv_cache.transpose(-1, -2),
                        sidx=sidx, didx=sidx, cu=cu, nacc=nacc, ssm=ssm_cache, A=w["A"], D=w["D"],
                        dt_bias=w["dt_bias"], seed=seed if w["philox"] else None, norm_w=w["norm_w"],
                        scale=w["scale"], out=xq, y=y, nseq=sidx.shape[0], pad=_null(),
                        nt=sidx.shape[-1] if spec else ntok)
    if spec:
        fd.fused_decode_mtp(args, sidx.shape[0], nt=sidx.shape[-1], hpc=hpc, philox=w["philox"])
    else:
        fd.fused_decode_stp(args, sidx.shape[0], hpc=hpc, philox=w["philox"])
    return xq


def make_caches(nslots, state_len, gen):
    """MambaSpec page layout: per block [conv (state_len, 2560) bf16 | ssm (32, 64, 128) fp16]."""
    conv_b, page = state_len * CD * 2, state_len * CD * 2 + NH * HD * NS * 2
    raw = torch.empty(nslots * page, dtype=torch.uint8, device=DEV)
    conv = torch.as_strided(raw.view(BF16), (nslots, state_len, CD), (page // 2, CD, 1), 0)
    ssm = torch.as_strided(raw.view(FP16), (nslots, NH, HD, NS), (page // 2, HD * NS, NS, 1), conv_b // 2)
    conv.copy_((torch.randn(conv.shape, generator=gen, device=DEV) * 0.8).to(BF16))
    ssm.copy_((torch.randn(ssm.shape, generator=gen, device=DEV) * 0.5).to(FP16))
    return raw, conv, ssm


def views(raw, conv, ssm):
    return (torch.as_strided(raw.view(BF16), conv.shape, conv.stride(), conv.storage_offset()),
            torch.as_strided(raw.view(FP16), ssm.shape, ssm.stride(), ssm.storage_offset()))


def weights(gen, philox):
    return {
        "conv_w": (torch.randn(CD, 4, generator=gen, device=DEV) * 0.3).to(BF16),
        "conv_b": (torch.randn(CD, generator=gen, device=DEV) * 0.1).to(BF16),
        "A": -torch.exp(torch.rand(NH, generator=gen, device=DEV) * 3),
        "D": torch.randn(NH, generator=gen, device=DEV).to(BF16),
        "dt_bias": (torch.randn(NH, generator=gen, device=DEV) - 3).to(BF16),
        "norm_w": (torch.rand(XD, generator=gen, device=DEV) + 0.5).to(BF16),
        "scale": torch.tensor(0.05, device=DEV), "philox": philox,
    }


def rand_proj(n, gen, scale=1.0):
    p = torch.randn(n, PW, generator=gen, device=DEV) * scale
    p[:, XD + CD:] = p[:, XD + CD:] * 2 - 1
    return p.to(BF16)


def case_stp(gen, m, n_null_tail=0, nslots=64):
    raw, conv, ssm = make_caches(nslots, 3, gen)
    sidx = (torch.randperm(nslots - 1, generator=CPU_GEN)[:m] + 1).to(torch.int32).view(m, 1).to(DEV)
    if n_null_tail:
        sidx[m - n_null_tail:] = _null()
    return raw, conv, ssm, sidx, None, None, m - n_null_tail


def case_mtp(gen, nseq, nt=8, lens=None, nacc=None, null_seqs=0, null_dst=False, nslots=160):
    raw, conv, ssm = make_caches(nslots, 3 + nt - 1, gen)
    sidx = (torch.randperm(nslots - 1, generator=CPU_GEN)[: nseq * nt] + 1).view(nseq, nt).to(torch.int32).to(DEV)
    lens = list(lens or [nt] * nseq)
    nacc = list(nacc or torch.randint(1, nt + 1, (nseq,), generator=CPU_GEN).tolist())
    for s in range(nseq - null_seqs, nseq):
        sidx[s], lens[s], nacc[s] = _null(), 0, 1
    if null_dst:
        sidx[0, nt - 1] = _null()
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.int32, device=DEV)
    return raw, conv, ssm, sidx, cu, torch.tensor(nacc, dtype=torch.int32, device=DEV), int(cu[-1])


def assert_same(a, b, what):
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    assert a.shape == b.shape and torch.equal(a8, b8), f"{what}: {int((a8 != b8).sum())} bytes differ"


def run_case(case, w, gen, hpc=1):
    raw, conv, ssm, sidx, cu, nacc, ntok = case
    nrows = max(ntok, 1)
    proj = rand_proj(nrows, gen)
    seed = torch.randint(0, 2**32, (1,), generator=CPU_GEN).to(DEV)
    raw2, proj2 = raw.clone(), proj.clone()
    conv2, ssm2 = views(raw2, conv, ssm)
    xq_s, y_s = stock_chain(proj, conv, ssm, sidx, cu, nacc, w, seed, ntok, nrows)
    y_f = torch.zeros(nrows, XD, dtype=BF16, device=DEV)
    xq_f = fused_chain(proj2, conv2, ssm2, sidx, cu, nacc, w, seed, ntok, nrows, hpc, y=y_f)
    torch.cuda.synchronize()
    assert_same(xq_f[:ntok], xq_s[:ntok], "fp8 out_proj input")
    assert_same(y_f[:ntok], y_s[:ntok], "ssm output")
    assert_same(proj2, proj, "projected_states")
    assert_same(raw2, raw, "conv/ssm cache buffers")


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 9])
@pytest.mark.parametrize("hpc", [1, 2])
@pytest.mark.parametrize("philox", [5, 0])
def test_single_token_decode_bitwise(m, hpc, philox):
    gen = torch.Generator(device=DEV).manual_seed(m * 10 + hpc + philox)
    w = weights(gen, philox)
    for _ in range(2):
        run_case(case_stp(gen, m), w, gen, hpc)
    run_case(case_stp(gen, m, n_null_tail=min(2, m - 1)), w, gen, hpc)


@pytest.mark.parametrize("nseq", [1, 2, 4, 8])
@pytest.mark.parametrize("philox", [5, 0])
def test_spec_verify_decode_bitwise(nseq, philox):
    gen = torch.Generator(device=DEV).manual_seed(100 + nseq + philox)
    w = weights(gen, philox)
    for _ in range(2):
        run_case(case_mtp(gen, nseq), w, gen)
    run_case(case_mtp(gen, 4, lens=[8, 3, 1, 6], nacc=[8, 2, 1, 4]), w, gen)
    run_case(case_mtp(gen, 8, null_seqs=3, null_dst=True), w, gen)
    if philox == 5:
        run_case(case_mtp(gen, 2, nt=6), w, gen, hpc=2)


@pytest.mark.parametrize("m", [1, 2, 5, 8, 33, 4096, 4097, 10240])
@pytest.mark.parametrize("scale", [0.02, 1.0, 30.0])
def test_gated_norm_fp8_quant_matches_inductor(m, scale):
    gen = torch.Generator(device=DEV).manual_seed(m)
    w = weights(gen, 5)
    proj = rand_proj(m, gen, scale)
    y = (torch.randn(m, XD, generator=gen, device=DEV) * scale).to(BF16)
    ref = inductor_norm_quant(y, proj, w["norm_w"], w["scale"])
    out = torch.empty(m, XD, dtype=FP8, device=DEV)
    _fd().gated_norm_fp8_quant(y, proj[:, :XD], w["norm_w"], w["scale"], out, m)
    assert_same(out, ref, "gated norm + fp8 quant")


@pytest.mark.parametrize("mode", ["single_token", "spec_verify"])
@pytest.mark.parametrize("batch", [1, 4])
def test_cuda_graph_replay_bitwise(mode, batch):
    """3 chained fused launches (PDL) in one CUDA graph, replayed on fresh inputs vs stock eager."""
    gen = torch.Generator(device=DEV).manual_seed(7 + batch)
    w = weights(gen, 5)
    raw, conv, ssm, sidx, cu, nacc, ntok = case_stp(gen, batch) if mode == "single_token" else case_mtp(gen, batch)
    sproj = torch.zeros(ntok, PW, dtype=BF16, device=DEV)
    sraw = raw.clone()
    sconv, sssm = views(sraw, conv, ssm)
    sseed = torch.zeros(1, dtype=torch.int64, device=DEV)
    sxq = torch.zeros(ntok, XD, dtype=FP8, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(3):
                fused_chain(sproj, sconv, sssm, sidx, cu, nacc, w, sseed, ntok, ntok, xq=sxq)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        proj = rand_proj(ntok, gen)
        seed = torch.randint(0, 2**32, (1,), generator=CPU_GEN).to(DEV)
        sproj.copy_(proj)
        sraw.copy_(raw)
        sseed.copy_(seed)
        g.replay()
        rraw, rproj = raw.clone(), proj.clone()
        rconv, rssm = views(rraw, conv, ssm)
        for _ in range(3):
            xq_s, _ = stock_chain(rproj, rconv, rssm, sidx, cu, nacc, w, seed, ntok, ntok)
        torch.cuda.synchronize()
        assert_same(sxq, xq_s, "fp8 out_proj input")
        assert_same(sproj, rproj, "projected_states")
        assert_same(sraw, rraw, "conv/ssm cache buffers")
