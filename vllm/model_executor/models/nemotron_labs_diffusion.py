# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Labs Diffusion's single-mask decision-scoring path.

The caller supplies a causal prefix followed by exactly one mask token. At
that final position, causal attention is equivalent to the native scorer's
bidirectional one-token block attending to its causal prefix cache. This is
not an implementation of multi-token diffusion generation.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class NemotronLabsDiffusionModel(MistralForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        rope = config.rope_parameters
        if "llama_4_scaling_beta" in rope:
            config.llama_4_scaling = {
                "beta": rope["llama_4_scaling_beta"],
                "original_max_position_embeddings": rope[
                    "original_max_position_embeddings"
                ],
            }
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("Nemotron decision scoring requires dictionary config")
        token_ids = additional.get("decision_token_ids")
        self.decision_token_ids = None
        self._decision_weight = None
        if token_ids is not None:
            if get_tensor_model_parallel_world_size() != 1:
                raise ValueError("Candidate-only scoring currently requires TP=1")
            if vllm_config.quant_config is not None:
                raise ValueError("Candidate-only scoring currently requires BF16/FP16")
            if (
                not token_ids
                or any(
                    type(i) is not int or not 0 <= i < config.vocab_size
                    for i in token_ids
                )
                or len(set(token_ids)) != len(token_ids)
            ):
                raise ValueError("decision_token_ids must be unique vocabulary ids")
            self.decision_token_ids = tuple(token_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.decision_token_ids is None:
            return super().compute_logits(hidden_states)
        # This head is inference-only and immutable after checkpoint loading.
        # Keep the engine's vocabulary indexing while avoiding the full GEMM.
        if self._decision_weight is None:
            self._decision_indices = torch.tensor(
                self.decision_token_ids, device=self.lm_head.weight.device
            )
            self._decision_weight = self.lm_head.weight.index_select(
                0, self._decision_indices
            ).contiguous()
        selected = F.linear(hidden_states, self._decision_weight).float()
        logits = selected.new_full(
            (*selected.shape[:-1], self.config.vocab_size), -torch.inf
        )
        return logits.index_copy_(-1, self._decision_indices, selected)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._decision_weight = None
        # These are HF-format weights: do not apply Mistral-format Q/K permutation.
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
