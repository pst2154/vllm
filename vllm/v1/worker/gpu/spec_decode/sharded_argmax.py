# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact greedy draft-token selection over a vocab-parallel lm_head.

Greedy drafting computes the logits shard on every TP rank, all-gathers the
full-vocab logits and takes ``argmax``. This module produces the same token
without gathering the logits:

1. ``_local_argmax_kernel``: one pass over the rank's logits shard with
   ``SPLIT`` CTAs per row. Each CTA writes its best ``(value, global id)``
   candidate, both as fp32 (fp32 holds every bf16/fp16/fp32 value and every
   id below 2**24 exactly).
2. An all-gather of the ``[num_rows, SPLIT * 2]`` fp32 candidates along dim 0
   (64 bytes per row and rank).
3. ``_pick_kernel``: per row, the best of the ``tp_size * SPLIT`` candidates.

Every comparison uses the total order of ``torch.argmax``: NaN beats every
number (the first NaN wins), otherwise the larger value wins, ties go to the
lower vocab id, and -0.0 == +0.0. The argmax under a total order with an index
tie-break composes over any partition of the vocabulary, so the result equals
``argmax(all_gather(logits))`` bit for bit.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

# CTAs per row. SPLIT * 2 fp32 = 64 bytes per row keeps the gathered rows
# 16-byte aligned, as required by the one-shot custom all-gather.
SPLIT = 8
BLOCK = 4096
# Global vocab ids are carried as fp32.
_MAX_EXACT_FP32_ID = 1 << 24


@triton.jit
def _better(v, i, bv, bi):
    # torch.argmax order: NaN beats everything (first NaN wins), else larger
    # value, ties -> lower index.
    nv = v != v
    nb = bv != bv
    return (
        (nv & (~nb))
        | (nv & nb & (i < bi))
        | ((~nv) & (~nb) & ((v > bv) | ((v == bv) & (i < bi))))
    )


@triton.jit
def _combine(v1, i1, v2, i2):
    t = _better(v2, i2, v1, i1)
    return tl.where(t, v2, v1), tl.where(t, i2, i1)


@triton.jit
def _local_argmax_kernel(
    logits_ptr,
    stride_row,
    n_valid,
    chunk,
    vocab_start,
    out_ptr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # grid (rows, SPLIT): CTA (row, s) scans [s * chunk, min((s + 1) * chunk,
    # n_valid)) once and writes its best (value as fp32, global id as fp32) to
    # out[row, s, :]. Empty chunks write (-inf, 2**31), which loses every
    # comparison against a real entry.
    row = tl.program_id(0)
    s = tl.program_id(1)
    base = logits_ptr + row.to(tl.int64) * stride_row
    lo = s * chunk
    hi = tl.minimum(lo + chunk, n_valid)
    offs0 = tl.arange(0, BLOCK)
    BIG: tl.constexpr = 2147483647
    bv = tl.full([BLOCK], float("-inf"), tl.float32)
    bi = tl.full([BLOCK], BIG, tl.int32)
    for start in range(lo, hi, BLOCK):
        offs = start + offs0
        msk = offs < hi
        x = tl.load(base + offs, mask=msk, other=float("-inf")).to(tl.float32)
        # Per lane, indices only grow, so "better" means NaN-first or strictly
        # greater; unset lanes take any entry.
        nx = x != x
        nb = bv != bv
        take = msk & ((bi == BIG) | (nx & (~nb)) | ((~nx) & (~nb) & (x > bv)))
        bv = tl.where(take, x, bv)
        bi = tl.where(take, offs, bi)
    v, i = tl.reduce((bv, bi), 0, _combine)
    o = out_ptr + (row * SPLIT + s) * 2
    tl.store(o, v)
    tl.store(o + 1, tl.where(i == BIG, 2147483648.0, (i + vocab_start).to(tl.float32)))


@triton.jit
def _pick_kernel(g_ptr, num_rows, out_ptr, NCAND: tl.constexpr, SPLIT: tl.constexpr):
    # g: [tp_size * num_rows, SPLIT * 2] fp32, the dim-0 all-gather of the
    # per-rank [num_rows, SPLIT * 2] candidates.
    row = tl.program_id(0)
    c = tl.arange(0, NCAND)  # candidate = r * SPLIT + s
    r = c // SPLIT
    s = c % SPLIT
    p = g_ptr + ((r * num_rows + row) * SPLIT + s) * 2
    v = tl.load(p)
    i = tl.load(p + 1)
    bv, bi = tl.reduce((v, i), 0, _combine)
    tl.store(out_ptr + row, bi.to(tl.int64))


def local_argmax_candidates(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    """[B, V_local] logits shard -> [B, SPLIT * 2] fp32 (value, global id)
    candidates."""
    assert logits.dim() == 2 and logits.stride(1) == 1
    num_rows, vocab_local = logits.shape
    out = torch.empty(num_rows, SPLIT * 2, dtype=torch.float32, device=logits.device)
    if num_rows:
        chunk = triton.cdiv(vocab_local, SPLIT)
        _local_argmax_kernel[(num_rows, SPLIT)](
            logits,
            logits.stride(0),
            vocab_local,
            chunk,
            vocab_start,
            out,
            SPLIT=SPLIT,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return out


def pick_from_candidates(
    gathered: torch.Tensor, num_rows: int, tp_size: int
) -> torch.Tensor:
    """[tp_size * B, SPLIT * 2] gathered candidates -> [B] int64 token ids."""
    num_candidates = tp_size * SPLIT
    assert num_candidates & (num_candidates - 1) == 0, (
        "tp_size * SPLIT must be a power of two"
    )
    out = torch.empty(num_rows, dtype=torch.int64, device=gathered.device)
    if num_rows:
        _pick_kernel[(num_rows,)](
            gathered,
            num_rows,
            out,
            NCAND=num_candidates,
            SPLIT=SPLIT,
            num_warps=1,
        )
    return out


def sharded_draft_argmax_unsupported_reason(
    speculator: "DraftModelSpeculator",
) -> str | None:
    """None if `sharded_draft_argmax` reproduces the greedy draft token of
    `DraftModelSpeculator.sample_draft` exactly, else the reason it does not.

    Assumes the draft model's `compute_logits` is
    `logits_processor(lm_head, hidden_states)`.
    """
    if speculator.draft_logits is not None:
        return "probabilistic drafting needs the full logits"
    if speculator.use_local_argmax_reduction:
        return "use_local_argmax_reduction is enabled"
    if speculator.use_acceptance_estimator:
        return "the acceptance estimator needs the full logits"
    if speculator.draft_watermarker is not None:
        return "the draft watermarker needs the full logits"
    model = speculator.model
    logits_processor = getattr(model, "logits_processor", None)
    lm_head = getattr(model, "lm_head", None)
    if (
        logits_processor is None
        or lm_head is None
        or not hasattr(lm_head, "shard_indices")
    ):
        return "the draft model has no logits_processor / vocab-parallel lm_head"
    if (
        getattr(logits_processor, "logits_as_input", False)
        or getattr(logits_processor, "soft_cap", None) is not None
        or getattr(logits_processor, "scale", 1.0) != 1.0
    ):
        return "logits scaling or soft cap"
    if getattr(lm_head, "bias", None) is not None:
        return "lm_head bias"
    # Checked on the whole vocabulary rather than on this rank's shard, so
    # that every TP rank takes the same path.
    vocab_size = lm_head.org_vocab_size
    if (
        lm_head.num_embeddings_padded != vocab_size
        or getattr(logits_processor, "org_vocab_size", vocab_size) != vocab_size
    ):
        return "vocab padding or added vocab"
    if vocab_size >= _MAX_EXACT_FP32_ID:
        return "vocab ids do not fit exactly in fp32"
    tp_size = lm_head.tp_size
    if tp_size <= 1:
        return "tp_size 1"
    if tp_size & (tp_size - 1):
        return "tp_size is not a power of two"
    if not getattr(logits_processor, "use_all_gather", True):
        return "the platform gathers logits instead of all-gathering them"
    return None


def sharded_draft_argmax(
    lm_head: torch.nn.Module,
    logits_processor: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Greedy draft tokens, identical to
    `logits_processor(lm_head, hidden_states).argmax(-1)` for configurations
    accepted by `sharded_draft_argmax_unsupported_reason`."""
    # Same lm_head GEMM as LogitsProcessor._get_logits, so the local logits
    # are bit-identical to the stock path's shard.
    logits = logits_processor._apply_head(lm_head, hidden_states, None)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    num_rows = logits.shape[0]
    shard = lm_head.shard_indices
    assert logits.shape[1] == shard.num_org_elements, (logits.shape, shard)
    candidates = local_argmax_candidates(logits, shard.org_vocab_start_index)
    gathered = tensor_model_parallel_all_gather(candidates, 0)
    return pick_from_candidates(gathered, num_rows, lm_head.tp_size)
