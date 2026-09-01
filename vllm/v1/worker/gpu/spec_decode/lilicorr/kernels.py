# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

_TILE_SIZE = 1024
_TILES_PER_PROGRAM = 8
_MAX_GREEDY_TOP_K = 16
_NEGATIVE_INFINITY = -3.0e38


@triton.jit
def _scan_tiles_kernel(
    logits_ptr,
    tile_max_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    vocab_size,
    num_tiles,
    logits_stride,
    tile_max_stride,
    partial_stride,
    TILE_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    negative_infinity = -3.0e38
    row = tl.program_id(0)
    group = tl.program_id(1)
    tile_offsets = group * TILES_PER_PROGRAM + tl.arange(0, TILES_PER_PROGRAM)
    offsets = tile_offsets[:, None] * TILE_SIZE + tl.arange(0, TILE_SIZE)[None, :]
    mask = (tile_offsets[:, None] < num_tiles) & (offsets < vocab_size)
    values = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=negative_infinity,
    ).to(tl.float32)
    maxima = tl.max(values, axis=1)
    tl.store(
        tile_max_ptr + row * tile_max_stride + tile_offsets,
        maxima,
        mask=tile_offsets < num_tiles,
    )
    partial_max = tl.max(maxima, axis=0)
    partial_sum = tl.sum(tl.sum(tl.exp(values - partial_max), axis=0), axis=0)
    tl.store(partial_max_ptr + row * partial_stride + group, partial_max)
    tl.store(partial_sum_ptr + row * partial_stride + group, partial_sum)


@triton.jit
def _select_tiles_kernel(
    logits_ptr,
    tile_ids_ptr,
    out_values_ptr,
    out_ids_ptr,
    vocab_size,
    logits_stride,
    tile_ids_stride,
    values_stride,
    ids_stride,
    TILE_SIZE: tl.constexpr,
    NUM_SELECTED_TILES: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_SELECTED: tl.constexpr,
):
    negative_infinity = -3.0e38
    row = tl.program_id(0)
    tile_ids = tl.load(
        tile_ids_ptr + row * tile_ids_stride + tl.arange(0, NUM_SELECTED_TILES)
    )
    offsets = tile_ids[:, None] * TILE_SIZE + tl.arange(0, TILE_SIZE)[None, :]
    mask = offsets < vocab_size
    values = tl.reshape(
        tl.load(
            logits_ptr + row * logits_stride + offsets,
            mask=mask,
            other=negative_infinity,
        ).to(tl.float32),
        (NUM_SELECTED,),
    )
    token_ids = tl.reshape(offsets, (NUM_SELECTED,))
    lanes = tl.arange(0, NUM_SELECTED)
    for index in range(TOP_K):
        maximum = tl.max(values, axis=0)
        position = tl.argmax(values, axis=0)
        token_id = tl.sum(tl.where(lanes == position, token_ids, 0), axis=0)
        tl.store(out_values_ptr + row * values_stride + index, maximum)
        tl.store(out_ids_ptr + row * ids_stride + index, token_id)
        values = tl.where(lanes == position, negative_infinity, values)


@triton.jit
def _greedy_path_kernel(
    start_ptr,
    pair_ptr,
    candidate_ptr,
    output_ptr,
    start_batch_stride,
    pair_batch_stride,
    pair_slot_stride,
    pair_from_stride,
    candidate_batch_stride,
    candidate_slot_stride,
    output_batch_stride,
    NUM_SLOTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    negative_infinity = -3.0e38
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < TOP_K
    scores = tl.load(
        start_ptr + row * start_batch_stride + offsets,
        mask=mask,
        other=negative_infinity,
    )
    current = tl.argmax(tl.where(mask, scores, negative_infinity), axis=0).to(tl.int32)
    tl.store(
        output_ptr + row * output_batch_stride,
        tl.load(candidate_ptr + row * candidate_batch_stride + current),
    )
    for slot in range(1, NUM_SLOTS):
        scores = tl.load(
            pair_ptr
            + row * pair_batch_stride
            + (slot - 1) * pair_slot_stride
            + current * pair_from_stride
            + offsets,
            mask=mask,
            other=negative_infinity,
        )
        current = tl.argmax(tl.where(mask, scores, negative_infinity), axis=0).to(
            tl.int32
        )
        tl.store(
            output_ptr + row * output_batch_stride + slot,
            tl.load(
                candidate_ptr
                + row * candidate_batch_stride
                + slot * candidate_slot_stride
                + current
            ),
        )


@triton.jit
def _cache_one_hot_logits_kernel(
    draft_logits_ptr,
    cached_ids_ptr,
    selected_ids_ptr,
    req_state_ptr,
    draft_batch_stride,
    draft_step_stride,
    num_steps: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    valid = req_state >= 0
    step = flat % num_steps
    base = draft_logits_ptr + req_state * draft_batch_stride + step * draft_step_stride
    old_token_id = tl.load(cached_ids_ptr + req_state * num_steps + step)
    new_token_id = tl.load(selected_ids_ptr + flat, mask=valid, other=0)
    tl.store(base + old_token_id, -float("inf"), mask=valid)
    tl.store(base + new_token_id, 0.0, mask=valid)
    tl.store(cached_ids_ptr + req_state * num_steps + step, new_token_id, mask=valid)


def _topk_log_probs_torch(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    values, ids = torch.topk(logits, top_k, dim=-1)
    row_max = values[:, :1]
    log_partition = (
        row_max.squeeze(-1).float()
        + (logits - row_max).exp().sum(dim=-1, dtype=torch.float32).log()
    )
    return values.float() - log_partition[:, None], ids.to(torch.int64)


def lilicorr_topk_log_probs(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not logits.is_cuda:
        return _topk_log_probs_torch(logits, top_k)

    logits = logits.contiguous()
    num_rows, vocab_size = logits.shape
    num_tiles = triton.cdiv(vocab_size, _TILE_SIZE)
    selected_tiles = min(top_k, num_tiles)
    if selected_tiles & (selected_tiles - 1):
        return _topk_log_probs_torch(logits, top_k)

    num_programs = triton.cdiv(num_tiles, _TILES_PER_PROGRAM)
    tile_max = torch.full(
        (num_rows, num_tiles),
        _NEGATIVE_INFINITY,
        dtype=torch.float32,
        device=logits.device,
    )
    partial_max = torch.empty(
        (num_rows, num_programs), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty_like(partial_max)
    _scan_tiles_kernel[(num_rows, num_programs)](
        logits,
        tile_max,
        partial_max,
        partial_sum,
        vocab_size,
        num_tiles,
        logits.stride(0),
        tile_max.stride(0),
        partial_max.stride(0),
        TILE_SIZE=_TILE_SIZE,
        TILES_PER_PROGRAM=_TILES_PER_PROGRAM,
        num_warps=4,
    )

    tile_ids = torch.topk(tile_max, selected_tiles, dim=-1).indices
    tile_ids = torch.sort(tile_ids, dim=-1).values.to(torch.int64)
    values = torch.empty((num_rows, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_rows, top_k), dtype=torch.int64, device=logits.device)
    _select_tiles_kernel[(num_rows,)](
        logits,
        tile_ids,
        values,
        ids,
        vocab_size,
        logits.stride(0),
        tile_ids.stride(0),
        values.stride(0),
        ids.stride(0),
        TILE_SIZE=_TILE_SIZE,
        NUM_SELECTED_TILES=selected_tiles,
        TOP_K=top_k,
        NUM_SELECTED=selected_tiles * _TILE_SIZE,
        num_warps=4,
    )
    global_max = partial_max.max(dim=-1, keepdim=True).values
    log_partition = (
        global_max.squeeze(-1)
        + (partial_sum * (partial_max - global_max).exp()).sum(dim=-1).log()
    )
    return values - log_partition[:, None], ids


def _greedy_path_torch(
    start_scores: torch.Tensor,
    pair_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    top_k = start_scores.shape[-1]
    current = start_scores.argmax(dim=-1)
    path = [current]
    for slot in range(1, candidate_ids.shape[1]):
        transitions = torch.gather(
            pair_scores[:, slot - 1],
            1,
            current[:, None, None].expand(-1, 1, top_k),
        ).squeeze(1)
        current = transitions.argmax(dim=-1)
        path.append(current)
    indices = torch.stack(path, dim=-1)
    return torch.gather(candidate_ids, 2, indices.unsqueeze(-1)).squeeze(-1)


def lilicorr_greedy_path(
    start_scores: torch.Tensor,
    pair_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    top_k = start_scores.shape[-1]
    if not start_scores.is_cuda or top_k > _MAX_GREEDY_TOP_K:
        return _greedy_path_torch(start_scores, pair_scores, candidate_ids)

    start_scores = start_scores.float().contiguous()
    pair_scores = pair_scores.float().contiguous()
    candidate_ids = candidate_ids.contiguous()
    batch_size, num_slots, _ = candidate_ids.shape
    output = torch.empty(
        (batch_size, num_slots), dtype=candidate_ids.dtype, device=candidate_ids.device
    )
    _greedy_path_kernel[(batch_size,)](
        start_scores,
        pair_scores,
        candidate_ids,
        output,
        start_scores.stride(0),
        pair_scores.stride(0),
        pair_scores.stride(1),
        pair_scores.stride(2),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        output.stride(0),
        NUM_SLOTS=num_slots,
        TOP_K=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    return output


def cache_one_hot_draft_logits(
    draft_logits: torch.Tensor,
    cached_ids: torch.Tensor,
    selected_ids: torch.Tensor,
    req_state: torch.Tensor,
    num_steps: int,
) -> None:
    _cache_one_hot_logits_kernel[(selected_ids.numel(),)](
        draft_logits,
        cached_ids,
        selected_ids,
        req_state,
        draft_logits.stride(0),
        draft_logits.stride(1),
        num_steps=num_steps,
    )
