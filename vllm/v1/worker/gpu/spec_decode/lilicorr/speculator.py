# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from .kernels import (
    cache_one_hot_draft_logits,
    lilicorr_greedy_path,
    lilicorr_topk_log_probs,
)


class LiLiCorrSpeculator(DFlashSpeculator):
    _speculator_name = "LiLiCorr"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config.dflash_config
        self.candidate_topk = int(config["lilicorr_candidate_topk"])
        self.anchor_hidden = torch.zeros(
            self.max_num_reqs,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.anchor_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self.cached_selected_ids = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        return torch.float32, -float("inf")

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = super().load_draft_model(target_model, target_attn_layer_names)
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = getattr(target_language_model, "model", target_language_model)
        target_input_embeddings = getattr(
            target_inner, "embed_tokens", None
        ) or getattr(target_inner, "embedding", None)
        if target_input_embeddings is None:
            raise ValueError("LiLiCorr requires the target model's token embeddings.")
        self.target_input_embeddings: nn.Module = target_input_embeddings
        return draft_model

    def _set_context_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
    ) -> None:
        num_reqs = input_batch.num_reqs
        ends = (
            input_batch.query_start_loc[1 : num_reqs + 1].to(torch.int64)
            - num_rejected[:num_reqs].to(torch.int64)
            - 1
        )
        valid = ends >= 0
        safe_ends = ends.clamp(min=0, max=max(hidden_states.shape[0] - 1, 0))
        self.anchor_hidden[:num_reqs].copy_(hidden_states.index_select(0, safe_ends))
        self.anchor_valid[:num_reqs].copy_(valid)
        if num_reqs < self.max_num_reqs:
            self.anchor_hidden[num_reqs:].zero_()
            self.anchor_valid[num_reqs:].zero_()

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
        pass_hidden = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        logits = self.model.compute_logits(pass_hidden.flatten(0, 1))
        assert logits is not None
        log_probs, candidate_ids = lilicorr_topk_log_probs(logits, self.candidate_topk)
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.candidate_topk
        )
        log_probs = log_probs.view_as(candidate_ids)
        token_embeddings = self.target_input_embeddings(candidate_ids).detach()
        start_scores, pair_scores = self.model.model.lilicorr.score(
            token_embeddings=token_embeddings,
            candidate_log_probs=log_probs,
            pass_hidden=pass_hidden,
            anchor_hidden=self.anchor_hidden[:num_reqs],
            anchor_valid=self.anchor_valid[:num_reqs],
        )
        selected = lilicorr_greedy_path(start_scores, pair_scores, candidate_ids).to(
            torch.int64
        )
        self.draft_tokens[:num_reqs].copy_(selected)
        if self.draft_logits is not None:
            cache_one_hot_draft_logits(
                self.draft_logits,
                self.cached_selected_ids,
                selected.flatten(),
                self.sample_idx_mapping[:num_sample],
                self.num_speculative_steps,
            )
