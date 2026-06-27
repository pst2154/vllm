# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class DSparkSpeculator(DFlashSpeculator):
    """Parallel DSpark block proposer with sequential Markov correction."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps
        target_hidden_size = self.hidden_size
        self.target_hidden_states = self.hidden_states
        projected_hidden_size = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            projected_hidden_size,
            dtype=self.dtype,
            device=device,
        )
        assert target_hidden_size == projected_hidden_size * len(
            self.draft_model_config.hf_config.dspark_target_layer_ids
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dspark_model(target_model, self.vllm_config)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden_states).view(
            num_reqs, self.num_speculative_steps, -1
        )
        previous_tokens = self.input_buffers.input_ids[
            self.sample_indices[: num_sample : self.num_speculative_steps]
        ]
        req_state_mapping = self.sample_idx_mapping[
            : num_sample : self.num_speculative_steps
        ]

        for step in range(self.num_speculative_steps):
            step_logits, _ = self.model.apply_markov_logits(
                base_logits[:, step], previous_tokens
            )
            if self.draft_logits is None:
                sampled = step_logits.argmax(dim=-1)
            else:
                step_positions = self.sample_pos[
                    step : num_sample : self.num_speculative_steps
                ]
                sampled = gumbel_sample(
                    step_logits,
                    req_state_mapping,
                    self.temperature,
                    self.seeds,
                    step_positions + 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self.sample_col[step],
                    use_fp64=self.use_fp64_gumbel,
                )
            self.draft_tokens[:num_reqs, step] = sampled
            previous_tokens = sampled

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del aux_hidden_states, mm_inputs
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        self.target_hidden_states[:num_target_tokens].copy_(
            last_hidden_states[:num_target_tokens]
        )
        projected = self.model.combine_hidden_states(
            self.target_hidden_states[:num_target_tokens]
        )
        self.hidden_states[:num_target_tokens].copy_(projected)
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
            )
            return self.draft_tokens[:num_reqs]

        assert self.draft_kv_cache_group_id >= 0
        query_slot_mapping = self.block_tables.slot_mappings[
            self.draft_kv_cache_group_id
        ]
        prepare_dspark_inputs(
            self.input_buffers,
            query_slot_mapping,
            self.context_positions,
            self.context_slot_mapping,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables.input_block_tables[self.draft_kv_cache_group_id],
            self.draft_block_size,
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.max_num_reqs,
            self.max_num_tokens,
        )
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slot_mapping=(
                None if dummy_run else self.context_slot_mapping[:num_target_tokens]
            ),
        )

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=False,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs_padded,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp,
                batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_dspark_inputs_kernel(
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    block_table_ptr,
    block_table_stride,
    noise_token_id,
    block_size,
    num_query_per_req,
    max_num_reqs,
    max_num_tokens,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - rejected
    sampled = tl.load(num_sampled_ptr + req_idx)
    if sampled > 0:
        anchor_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        anchor_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = tl.minimum(ctx_pos // block_size, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    input_id = tl.where(query_off == 0, anchor_token, noise_token_id)
    query_block_num = tl.minimum(query_pos // block_size, block_table_stride - 1)
    query_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + query_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    query_slot = query_block_id * block_size + (query_pos % block_size)
    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_query_positions_ptr + query_idx, query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, query_slot, mask=is_query)

    sample_idx = req_idx * num_query_per_req + query_off
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_query)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos, mask=is_query)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_query)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_query_per_req
            pad_end = max_num_reqs * num_query_per_req
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, 0, mask=mask)
            for i in range(last_query_end, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dspark_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    noise_token_id: int,
    num_query_per_req: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> None:
    num_reqs = input_batch.num_reqs
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    block = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, block)
    _prepare_dspark_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        noise_token_id,
        block_size,
        num_query_per_req,
        max_num_reqs,
        max_num_tokens,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=block,
    )
