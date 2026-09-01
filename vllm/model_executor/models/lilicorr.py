# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model


@dataclass(frozen=True)
class LiLiCorrConfig:
    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, model_hidden_size: int) -> int:
        return self.hidden_size or model_hidden_size


def parse_lilicorr_config(config) -> LiLiCorrConfig:
    dflash_config = getattr(config, "dflash_config", None) or {}

    def required(name: str, cast, *, positive: bool = True):
        key = f"lilicorr_{name}"
        if key not in dflash_config:
            raise ValueError(
                f"LiLiCorr requires dflash_config.{key} to reconstruct its "
                "trained head."
            )
        try:
            value = cast(dflash_config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid dflash_config.{key}={dflash_config[key]!r}."
            ) from exc
        if positive and value <= 0:
            raise ValueError(f"dflash_config.{key} must be positive, got {value}.")
        return value

    candidate_topk = required("candidate_topk", int)
    if candidate_topk & (candidate_topk - 1):
        raise ValueError(
            "dflash_config.lilicorr_candidate_topk must be a power of two, got "
            f"{candidate_topk}."
        )
    hidden_size = required("hidden_size", int, positive=False)
    if hidden_size < 0:
        raise ValueError(
            "dflash_config.lilicorr_hidden_size must be non-negative, got "
            f"{hidden_size}."
        )
    return LiLiCorrConfig(
        candidate_topk=candidate_topk,
        hidden_size=hidden_size,
        num_layers=required("num_layers", int),
        num_heads=required("num_heads", int),
        mlp_ratio=required("mlp_ratio", float),
        factor_dim=required("factor_dim", int),
        vector_eps=required("vector_eps", float),
        logit_scale=required("logit_scale", float),
    )


class LiLiCorrRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            self.normalized_shape,
            self.weight,
            self.variance_epsilon,
        )


class LiLiCorrLatticeAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"LiLiCorr hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = F.linear(
            hidden_states, self.in_proj_weight, self.in_proj_bias
        ).chunk(3, dim=-1)
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
        )
        return self.out_proj(
            output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        )


class LiLiCorrLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        self.attn = LiLiCorrLatticeAttention(hidden_size, num_heads)
        self.mlp_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, hidden_size),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states), attention_bias
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class LiLiCorrHead(nn.Module):
    num_candidate_features = 5

    def __init__(
        self,
        *,
        model_hidden_size: int,
        block_size: int,
        rms_norm_eps: float,
        config: LiLiCorrConfig,
    ) -> None:
        super().__init__()
        hidden_size = config.resolve_hidden_size(model_hidden_size)
        self.block_size = block_size
        self.num_candidate_slots = block_size - 1
        self.candidate_topk = config.candidate_topk
        self.hidden_size = hidden_size
        self.num_heads = config.num_heads
        self.factor_dim = config.factor_dim
        self.vector_eps = config.vector_eps
        self.logit_scale = config.logit_scale

        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(self.num_candidate_features),
            nn.Linear(self.num_candidate_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, 1, self.num_candidate_slots, 1, hidden_size)
        )
        self.rank_embedding = nn.Parameter(
            torch.empty(1, 1, 1, self.candidate_topk, hidden_size)
        )
        self.relative_slot_bias = nn.Parameter(
            torch.empty(self.num_heads, 2 * self.block_size - 1)
        )
        self.same_slot_bias = nn.Parameter(torch.empty(self.num_heads))
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                LiLiCorrLayer(
                    hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    rms_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        self.anchor_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.out_head = nn.Linear(hidden_size, self.factor_dim)
        self.in_head = nn.Linear(hidden_size, self.factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, self.factor_dim)

        self._attention_bias: torch.Tensor | None = None
        self._rank_frac: torch.Tensor | None = None
        self._is_top1: torch.Tensor | None = None

    @torch.no_grad()
    def materialize_inference_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        topk = self.candidate_topk
        slot_ids = torch.arange(
            self.num_candidate_slots, device=device, dtype=torch.long
        ).repeat_interleave(topk)
        relative = slot_ids[:, None] - slot_ids[None, :]
        relative = relative.clamp(min=-(self.block_size - 1), max=self.block_size - 1)
        bias = self.relative_slot_bias[:, relative + self.block_size - 1]
        same_slot = slot_ids[:, None] == slot_ids[None, :]
        bias = (
            bias + same_slot[None].to(bias.dtype) * self.same_slot_bias[:, None, None]
        )
        self._attention_bias = bias.to(device=device, dtype=dtype).contiguous()

        if topk == 1:
            rank_frac = torch.zeros(topk, device=device)
        else:
            rank_frac = torch.arange(topk, device=device) / (topk - 1)
        is_top1 = torch.zeros(topk, device=device)
        is_top1[0] = 1
        self._rank_frac = rank_frac.view(1, 1, topk)
        self._is_top1 = is_top1.view(1, 1, topk)

    def score(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._attention_bias is None:
            raise RuntimeError(
                "LiLiCorr inference buffers were not materialized after weight load."
            )
        assert self._rank_frac is not None and self._is_top1 is not None
        batch_size, num_slots, topk = candidate_log_probs.shape
        if topk != self.candidate_topk or num_slots != self.num_candidate_slots:
            raise ValueError(
                "LiLiCorr candidate lattice does not match the checkpoint geometry: "
                f"got slots={num_slots}, topk={topk}; expected "
                f"slots={self.num_candidate_slots}, topk={self.candidate_topk}."
            )

        dtype = self.pass_hidden_proj.weight.dtype
        token_states = self.token_proj(token_embeddings.to(dtype))
        pass_states = self.pass_hidden_proj(pass_hidden.to(dtype)).unsqueeze(-2)
        log_probs = candidate_log_probs.float()
        features = torch.stack(
            (
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.max(dim=-1, keepdim=True).values,
                self._rank_frac.expand_as(log_probs),
                self._is_top1.expand_as(log_probs),
            ),
            dim=-1,
        )
        hidden_states = token_states + pass_states
        hidden_states = hidden_states + self.feature_mlp(features.to(dtype))
        hidden_states = hidden_states + self.slot_embedding[:, 0]
        hidden_states = hidden_states + self.rank_embedding[:, 0]
        hidden_states = hidden_states.reshape(
            batch_size, num_slots * topk, self.hidden_size
        )

        attention_bias = self._attention_bias[None].expand(batch_size, -1, -1, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        hidden_states = self.output_norm(hidden_states).view(
            batch_size, num_slots, topk, self.hidden_size
        )

        anchor_state = self.context_proj(anchor_hidden.to(dtype))
        anchor_state = anchor_state * anchor_valid[:, None].to(dtype)
        anchor_state = self.anchor_norm(anchor_state)
        anchor_row = anchor_state[:, None, None].expand(-1, num_slots, topk, -1)
        factor_hidden = F.silu(
            self.factor_input_proj(
                torch.cat(
                    (hidden_states, anchor_row, hidden_states * anchor_row), dim=-1
                )
            )
        )
        out_vectors = F.normalize(
            self.out_head(factor_hidden), dim=-1, eps=self.vector_eps
        )
        in_vectors = F.normalize(
            self.in_head(factor_hidden), dim=-1, eps=self.vector_eps
        )
        anchor_out = F.normalize(
            self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps
        )

        start_scores = (anchor_out[:, None] * in_vectors[:, 0]).sum(dim=-1)
        pair_scores = torch.matmul(
            out_vectors[:, :-1], in_vectors[:, 1:].transpose(-1, -2)
        )
        return (
            self.logit_scale * start_scores.float(),
            self.logit_scale * pair_scores.float(),
        )


class LiLiCorrQwen3Model(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        block_size = speculative_config.num_speculative_tokens + 1
        with set_model_tag("lilicorr_head"):
            self.lilicorr = LiLiCorrHead(
                model_hidden_size=self.config.hidden_size,
                block_size=block_size,
                rms_norm_eps=self.config.rms_norm_eps,
                config=parse_lilicorr_config(self.config),
            )


def check_head_weight_coverage(head: LiLiCorrHead, seen: set[str]) -> None:
    expected = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        raise ValueError(
            f"LiLiCorr checkpoint is missing {len(missing)} head parameters "
            f"(for example, {missing[:5]})."
        )
    if unexpected:
        raise ValueError(
            f"LiLiCorr checkpoint contains {len(unexpected)} unexpected head "
            f"parameters (for example, {unexpected[:5]})."
        )


class LiLiCorrQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = LiLiCorrQwen3Model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        seen: set[str] = set()

        def tracking():
            for name, weight in weights:
                normalized = name.removeprefix("model.")
                if normalized.startswith("lilicorr."):
                    seen.add(normalized)
                    name = normalized
                yield name, weight

        super().load_weights(tracking())
        check_head_weight_coverage(self.model.lilicorr, seen)
        parameter = next(self.model.lilicorr.parameters())
        self.model.lilicorr.materialize_inference_buffers(
            parameter.device, parameter.dtype
        )


EntryClass = LiLiCorrQwen3ForCausalLM
