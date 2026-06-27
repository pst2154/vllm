# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark block drafter for DeepSeek V4.

DSpark is not an autoregressive MTP head. It predicts a block from one accepted
token plus mask/noise tokens, then applies a token-conditioned Markov correction
to each position before standard speculative verification.
"""

from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model, DeepseekV4MoE
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.v1.attention.backend import AttentionType

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DeepSeekV4DSparkAttention(nn.Module):
    """Uncompressed sliding-window MLA used by each DSpark stage."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()

        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
        )
        self.attn = Attention(
            self.n_local_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window,
            prefix=f"{prefix}.paged_attn",
            attn_type=AttentionType.DECODER,
            sinks=self.attn_sink,
        )
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _project_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        _, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return self.kv_norm(kv).unsqueeze(1)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        kv = self._project_kv(context_states)
        kv, _ = self.rotary_emb(context_positions, kv, None)
        if context_slot_mapping is None:
            return
        self.attn.impl.do_kv_cache_update(
            self.attn,
            kv,
            kv,
            self.attn.kv_cache,
            context_slot_mapping,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q *= torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(
            q.dtype
        )
        kv = kv.unsqueeze(1)
        q, kv = self.rotary_emb(positions, q, kv)
        assert kv is not None
        output = self.attn(q, kv, kv).view(-1, self.n_local_heads, self.head_dim)
        return deep_gemm_fp8_o_proj(
            output,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )


class DSparkMarkovHead(nn.Module):
    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.markov_w1(token_ids)
        logits = self.logits_processor(self.markov_w2, embeddings)
        return logits, embeddings


class DSparkConfidenceHead(nn.Module):
    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            config.hidden_size + config.dspark_markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self, hidden_states: torch.Tensor, markov_embeddings: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([hidden_states.float(), markov_embeddings.float()], dim=-1)
        return self.proj(features).squeeze(-1)


class DeepSeekV4DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        layer_idx: int,
        stage_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = get_draft_quant_config(vllm_config)
        self.config = config
        self.stage_idx = stage_idx
        self.attn = DeepSeekV4DSparkAttention(vllm_config, f"{prefix}.attn")
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

        if stage_idx == 0:
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if stage_idx == config.n_mtp_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config, prefix=f"{prefix}.markov_head")
            self.confidence_head = DSparkConfidenceHead(
                config, prefix=f"{prefix}.confidence_head"
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            post_mix, res_mix, hidden_states = mhc_pre_tilelang(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.config.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=self.attn_norm.weight.data,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            residual, post_mix, res_mix, hidden_states = mhc_fused_post_pre_tilelang(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.config.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=self.attn_norm.weight.data,
                norm_eps=self.attn_norm.variance_epsilon,
            )

        hidden_states = self.attn(positions, hidden_states)
        residual, post_mix, res_mix, hidden_states = mhc_fused_post_pre_tilelang(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.config.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=self.ffn_norm.weight.data,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        hidden_states = self.ffn(hidden_states, input_ids)
        return hidden_states, residual, post_mix, res_mix


class DeepSeekV4DSparkBackbone(nn.Module):
    # Reuse the production DeepSeek V4 weight loader for attention, MoE, and
    # hyper-connection tensors. The DSpark module deliberately keeps the same
    # per-layer names and layouts.
    load_weights = DeepseekV4Model.load_weights
    get_expert_mapping = DeepseekV4Model.get_expert_mapping
    _pad_shared_expert_weight = DeepseekV4Model._pad_shared_expert_weight
    finalize_mega_moe_weights = DeepseekV4Model.finalize_mega_moe_weights

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)
        self.parallel_config = vllm_config.parallel_config
        self.start_layer = self.config.num_hidden_layers
        self.end_layer = self.start_layer + self.config.n_mtp_layers
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [PPMissingLayer() for _ in range(self.start_layer)]
            + [
                DeepSeekV4DSparkDecoderLayer(
                    vllm_config,
                    layer_idx=layer_idx,
                    stage_idx=layer_idx - self.start_layer,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(self.start_layer, self.end_layer)
            ]
        )

    @property
    def first_layer(self) -> DeepSeekV4DSparkDecoderLayer:
        return self.layers[self.start_layer]  # type: ignore[return-value]

    @property
    def last_layer(self) -> DeepSeekV4DSparkDecoderLayer:
        return self.layers[self.end_layer - 1]  # type: ignore[return-value]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.first_layer.main_norm(self.first_layer.main_proj(hidden_states))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.attn.precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mapping
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = (
            inputs_embeds.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
        )
        residual = post_mix = res_mix = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        last_layer = self.last_layer
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            last_layer.hc_head_fn,
            last_layer.hc_head_scale,
            last_layer.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        return last_layer.norm(hidden_states)


class DeepSeekV4DSpark(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert get_pp_group().world_size == 1, (
            "DSpark does not support pipeline parallel"
        )
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DeepSeekV4DSparkBackbone(
            vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def apply_markov_logits(
        self, base_logits: torch.Tensor, previous_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias, embeddings = self.model.last_layer.markov_head(previous_token_ids)
        return base_logits + bias, embeddings

    def predict_confidence(
        self, hidden_states: torch.Tensor, markov_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.model.last_layer.confidence_head(hidden_states, markov_embeddings)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        layer_weights: list[tuple[str, torch.Tensor]] = []
        head_weights: list[tuple[str, torch.Tensor]] = []
        start_layer = self.config.num_hidden_layers
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )
        for name, loaded_weight in weights:
            if not name.startswith("mtp."):
                continue
            parts = name.split(".", 2)
            stage_idx = int(parts[1])
            name = f"layers.{start_layer + stage_idx}.{parts[2]}"
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix
            name = name.replace(
                ".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"
            )
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            if ".markov_head." in name or ".confidence_head." in name:
                head_weights.append((name, loaded_weight))
            else:
                layer_weights.append((name, loaded_weight))

        loaded = self.model.load_weights(layer_weights)
        params_dict = dict(self.model.named_parameters())
        for name, loaded_weight in head_weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        self.model.finalize_mega_moe_weights()
        return {f"model.{name}" for name in loaded}
