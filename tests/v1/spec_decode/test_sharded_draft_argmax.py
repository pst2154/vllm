# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness of the sharded draft argmax (VLLM_SPEC_DRAFT_SHARDED_ARGMAX).

The kernels are checked against torch.argmax on the concatenated logits by
simulating the TP shards on one GPU: per-shard candidates are concatenated in
the layout of a dim-0 all-gather and reduced by the pick kernel. Covers heavy
ties within and across shards, -inf rows, +inf ties, NaN (first NaN wins),
-0.0/+0.0 ties, bf16/fp16/fp32, and CUDA graph replay with changing inputs.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.sharded_argmax import (
    local_argmax_candidates,
    pick_from_candidates,
    sharded_draft_argmax_unsupported_reason,
)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernels require CUDA."
)


def _select(logits: torch.Tensor, tp_size: int) -> torch.Tensor:
    num_rows, vocab_size = logits.shape
    vocab_local = vocab_size // tp_size
    candidates = [
        local_argmax_candidates(
            logits[:, r * vocab_local : (r + 1) * vocab_local], r * vocab_local
        )
        for r in range(tp_size)
    ]
    # Same layout as all_gather(dim=0): [tp_size * num_rows, SPLIT * 2].
    gathered = torch.cat(candidates, 0)
    return pick_from_candidates(gathered, num_rows, tp_size)


def _cases(num_rows, vocab_size, dtype, gen):
    device = "cuda"
    yield (
        "randn",
        torch.randn(num_rows, vocab_size, generator=gen, device=device).to(dtype),
    )
    yield (
        "ties-int",
        torch.randint(-3, 4, (num_rows, vocab_size), generator=gen, device=device).to(
            dtype
        ),
    )
    x = torch.randn(num_rows, vocab_size, generator=gen, device=device).to(dtype)
    x[:, vocab_size // 3] = 50
    x[:, (2 * vocab_size) // 3] = 50
    x[:, vocab_size - 1] = 50
    yield "cross-shard-ties", x
    yield (
        "all-neg-inf",
        torch.full((num_rows, vocab_size), float("-inf"), device=device, dtype=dtype),
    )
    x = torch.randn(num_rows, vocab_size, generator=gen, device=device).to(dtype)
    x[:, 7] = float("inf")
    x[:, vocab_size // 2] = float("inf")
    yield "pos-inf-ties", x
    x = torch.randn(num_rows, vocab_size, generator=gen, device=device).to(dtype)
    x[0, vocab_size // 2 + 5] = float("nan")
    x[0, vocab_size - 2] = float("nan")
    x[:, 100] = 1e4 if dtype != torch.float16 else 6e4
    yield "nan", x
    x = torch.full((num_rows, vocab_size), -1.0, device=device, dtype=dtype)
    x[:, vocab_size // 4 + 3] = -0.0
    x[:, 3 * vocab_size // 4] = 0.0
    yield "signed-zero-ties", x
    yield "zeros", torch.zeros(num_rows, vocab_size, device=device, dtype=dtype)


@requires_cuda
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab_kind", ["131072", "32000", "4096_per_rank"])
def test_sharded_argmax_matches_torch_argmax(tp_size, dtype, vocab_kind):
    vocab_size = {
        "131072": 131072,
        "32000": 32000,
        "4096_per_rank": 4096 * tp_size,
    }[vocab_kind]
    assert vocab_size % tp_size == 0
    gen = torch.Generator(device="cuda").manual_seed(0)
    for num_rows in (1, 2, 3, 4, 8, 13, 64):
        for name, x in _cases(num_rows, vocab_size, dtype, gen):
            ref = torch.argmax(x, dim=-1)
            got = _select(x, tp_size)
            assert torch.equal(ref, got), (
                f"{name}: B={num_rows}, rows "
                f"{(ref != got).nonzero().flatten().tolist()}"
            )


@requires_cuda
def test_sharded_argmax_cuda_graph_replay():
    tp_size, num_rows, vocab_size = 4, 8, 131072
    gen = torch.Generator(device="cuda").manual_seed(0)
    static = torch.randn(num_rows, vocab_size, device="cuda").to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _select(static, tp_size)  # warm-up / Triton compile
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _select(static, tp_size)
    for it in range(200):
        if it % 2:
            fresh = torch.randint(
                -2, 3, (num_rows, vocab_size), generator=gen, device="cuda"
            )
        else:
            fresh = torch.randn(num_rows, vocab_size, generator=gen, device="cuda")
        static.copy_(fresh.to(torch.bfloat16))
        graph.replay()
        assert torch.equal(out, torch.argmax(static, -1)), it


def _fake_speculator(**overrides):
    vocab_size = 131072
    logits_processor = SimpleNamespace(
        logits_as_input=False,
        soft_cap=None,
        scale=1.0,
        org_vocab_size=vocab_size,
        use_all_gather=True,
    )
    lm_head = SimpleNamespace(
        shard_indices=object(),
        bias=None,
        org_vocab_size=vocab_size,
        num_embeddings_padded=vocab_size,
        tp_size=4,
    )
    speculator = SimpleNamespace(
        draft_logits=None,
        use_local_argmax_reduction=False,
        use_acceptance_estimator=False,
        draft_watermarker=None,
        model=SimpleNamespace(logits_processor=logits_processor, lm_head=lm_head),
    )
    for path, value in overrides.items():
        obj = speculator
        *parents, name = path.split(".")
        for parent in parents:
            obj = getattr(obj, parent)
        setattr(obj, name, value)
    return speculator


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, None),
        ({"draft_logits": torch.empty(1)}, "probabilistic"),
        ({"use_local_argmax_reduction": True}, "use_local_argmax_reduction"),
        ({"use_acceptance_estimator": True}, "acceptance estimator"),
        ({"draft_watermarker": object()}, "watermarker"),
        ({"model": SimpleNamespace()}, "lm_head"),
        ({"model.logits_processor.soft_cap": 30.0}, "soft cap"),
        ({"model.logits_processor.scale": 0.5}, "scaling"),
        ({"model.logits_processor.logits_as_input": True}, "scaling"),
        ({"model.lm_head.bias": torch.zeros(1)}, "bias"),
        ({"model.lm_head.num_embeddings_padded": 131136}, "padding"),
        ({"model.logits_processor.org_vocab_size": 131000}, "padding"),
        (
            {
                "model.lm_head.org_vocab_size": 1 << 24,
                "model.lm_head.num_embeddings_padded": 1 << 24,
                "model.logits_processor.org_vocab_size": 1 << 24,
            },
            "fp32",
        ),
        ({"model.lm_head.tp_size": 1}, "tp_size 1"),
        ({"model.lm_head.tp_size": 6}, "power of two"),
        ({"model.logits_processor.use_all_gather": False}, "gathers"),
    ],
)
def test_sharded_draft_argmax_eligibility(overrides, expected):
    reason = sharded_draft_argmax_unsupported_reason(_fake_speculator(**overrides))
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason
