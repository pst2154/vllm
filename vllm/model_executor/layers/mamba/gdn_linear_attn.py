# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import ast
import json
import os
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    autotune as flashinfer_autotune,
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode fast-path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)


@triton.jit
def _qwen_gdn_small_m_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k_start in range(0, K, BK):
        k = k_start + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _qwen_gdn_short_prefill_conv_prep_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    conv_state_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    state_idx_ptr,
    has_initial_state_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    L: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INITIAL_STATE_TENSOR: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_cs_block: tl.constexpr,
    stride_cs_d: tl.constexpr,
    stride_cs_s: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_w_s: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_v_t: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    pid_t = tl.program_id(0)
    pid_head = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L
    state_idx = tl.load(state_idx_ptr).to(tl.int64)
    has_initial = True
    if HAS_INITIAL_STATE_TENSOR:
        has_initial = tl.load(has_initial_state_ptr).to(tl.int1)

    HK: tl.constexpr = H * K
    V_OFFSET: tl.constexpr = 2 * H * K

    if pid_head < H:
        offs_k = tl.arange(0, BK)
        mask_k = offs_k < K
        mask_2d = mask_t[:, None] & mask_k[None, :]
        q_dim = pid_head * K + offs_k
        k_dim = HK + pid_head * K + offs_k

        q_acc = tl.zeros((BLOCK_T, BK), dtype=tl.float32)
        k_acc = tl.zeros((BLOCK_T, BK), dtype=tl.float32)
        if HAS_BIAS:
            q_acc += tl.load(conv_bias_ptr + q_dim, mask=mask_k, other=0.0)[None, :]
            k_acc += tl.load(conv_bias_ptr + k_dim, mask=mask_k, other=0.0)[None, :]

        for s in tl.static_range(0, STATE_LEN + 1):
            src_t = offs_t + s - STATE_LEN
            q_w = tl.load(
                conv_weight_ptr + q_dim * stride_w_d + s * stride_w_s,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            k_w = tl.load(
                conv_weight_ptr + k_dim * stride_w_d + s * stride_w_s,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            q_x = tl.load(
                mixed_qkv_ptr + src_t[:, None] * stride_x_t + q_dim[None, :] * stride_x_d,
                mask=(src_t[:, None] >= 0) & mask_2d,
                other=0.0,
            ).to(tl.float32)
            k_x = tl.load(
                mixed_qkv_ptr + src_t[:, None] * stride_x_t + k_dim[None, :] * stride_x_d,
                mask=(src_t[:, None] >= 0) & mask_2d,
                other=0.0,
            ).to(tl.float32)
            if has_initial:
                state_col = src_t + STATE_LEN
                q_state = tl.load(
                    conv_state_ptr
                    + state_idx * stride_cs_block
                    + q_dim[None, :] * stride_cs_d
                    + state_col[:, None] * stride_cs_s,
                    mask=(src_t[:, None] < 0) & mask_2d,
                    other=0.0,
                ).to(tl.float32)
                k_state = tl.load(
                    conv_state_ptr
                    + state_idx * stride_cs_block
                    + k_dim[None, :] * stride_cs_d
                    + state_col[:, None] * stride_cs_s,
                    mask=(src_t[:, None] < 0) & mask_2d,
                    other=0.0,
                ).to(tl.float32)
                q_x = tl.where(src_t[:, None] < 0, q_state, q_x)
                k_x = tl.where(src_t[:, None] < 0, k_state, k_x)
            q_acc += q_x * q_w[None, :]
            k_acc += k_x * k_w[None, :]

        q_acc = q_acc / (1.0 + tl.exp(-q_acc))
        k_acc = k_acc / (1.0 + tl.exp(-k_acc))
        q_norm = tl.rsqrt(tl.sum(q_acc * q_acc, axis=1) + 1e-6)
        k_norm = tl.rsqrt(tl.sum(k_acc * k_acc, axis=1) + 1e-6)
        q_acc *= q_norm[:, None]
        k_acc *= k_norm[:, None]

        tl.store(
            q_ptr + offs_t[:, None] * stride_q_t + pid_head * K + offs_k[None, :],
            q_acc.to(q_ptr.dtype.element_ty),
            mask=mask_2d,
        )
        tl.store(
            k_ptr + offs_t[:, None] * stride_k_t + pid_head * K + offs_k[None, :],
            k_acc.to(k_ptr.dtype.element_ty),
            mask=mask_2d,
        )

        for st in tl.static_range(0, STATE_LEN):
            src_t = L - STATE_LEN + st
            q_last = tl.load(
                mixed_qkv_ptr + src_t * stride_x_t + q_dim * stride_x_d,
                mask=mask_k,
                other=0.0,
            )
            k_last = tl.load(
                mixed_qkv_ptr + src_t * stride_x_t + k_dim * stride_x_d,
                mask=mask_k,
                other=0.0,
            )
            tl.store(
                conv_state_ptr
                + state_idx * stride_cs_block
                + q_dim * stride_cs_d
                + st * stride_cs_s,
                q_last,
                mask=mask_k,
            )
            tl.store(
                conv_state_ptr
                + state_idx * stride_cs_block
                + k_dim * stride_cs_d
                + st * stride_cs_s,
                k_last,
                mask=mask_k,
            )
    else:
        i_hv = pid_head - H
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]
        v_dim = V_OFFSET + i_hv * V + offs_v
        v_acc = tl.zeros((BLOCK_T, BV), dtype=tl.float32)
        if HAS_BIAS:
            v_acc += tl.load(conv_bias_ptr + v_dim, mask=mask_v, other=0.0)[None, :]

        for s in tl.static_range(0, STATE_LEN + 1):
            src_t = offs_t + s - STATE_LEN
            v_w = tl.load(
                conv_weight_ptr + v_dim * stride_w_d + s * stride_w_s,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            v_x = tl.load(
                mixed_qkv_ptr + src_t[:, None] * stride_x_t + v_dim[None, :] * stride_x_d,
                mask=(src_t[:, None] >= 0) & mask_2d,
                other=0.0,
            ).to(tl.float32)
            if has_initial:
                state_col = src_t + STATE_LEN
                v_state = tl.load(
                    conv_state_ptr
                    + state_idx * stride_cs_block
                    + v_dim[None, :] * stride_cs_d
                    + state_col[:, None] * stride_cs_s,
                    mask=(src_t[:, None] < 0) & mask_2d,
                    other=0.0,
                ).to(tl.float32)
                v_x = tl.where(src_t[:, None] < 0, v_state, v_x)
            v_acc += v_x * v_w[None, :]

        v_acc = v_acc / (1.0 + tl.exp(-v_acc))
        tl.store(
            v_ptr + offs_t[:, None] * stride_v_t + i_hv * V + offs_v[None, :],
            v_acc.to(v_ptr.dtype.element_ty),
            mask=mask_2d,
        )

        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        a_vals = tl.load(a_ptr + offs_t * stride_a_t + i_hv, mask=mask_t, other=0.0).to(
            tl.float32
        )
        b_vals = tl.load(b_ptr + offs_t * stride_b_t + i_hv, mask=mask_t, other=0.0).to(
            tl.float32
        )
        x = a_vals + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= 20.0, sp, x)
        g_vals = -tl.exp(A_log_val) * sp
        beta_vals = tl.sigmoid(b_vals)
        gb_offsets = offs_t * HV + i_hv
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)

        for st in tl.static_range(0, STATE_LEN):
            src_t = L - STATE_LEN + st
            v_last = tl.load(
                mixed_qkv_ptr + src_t * stride_x_t + v_dim * stride_x_d,
                mask=mask_v,
                other=0.0,
            )
            tl.store(
                conv_state_ptr
                + state_idx * stride_cs_block
                + v_dim * stride_cs_d
                + st * stride_cs_s,
                v_last,
                mask=mask_v,
            )


def qwen_gdn_short_prefill_conv_prep(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    L = mixed_qkv.shape[0]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    D = mixed_qkv.shape[1]
    state_len = conv_weights.shape[1] - 1

    q = torch.empty(L, H, K, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty(L, H, K, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v = torch.empty(L, HV, V, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    g = torch.empty(L, HV, dtype=torch.float32, device=mixed_qkv.device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=mixed_qkv.device)

    block_t = 16
    grid = (triton.cdiv(L, block_t), H + HV)
    _qwen_gdn_short_prefill_conv_prep_kernel[grid](
        mixed_qkv,
        a,
        b,
        conv_state,
        conv_weights,
        conv_bias,
        state_indices,
        has_initial_state,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        L,
        D,
        H,
        HV,
        K,
        V,
        state_len,
        conv_bias is not None,
        has_initial_state is not None,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        a.stride(0),
        b.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weights.stride(0),
        conv_weights.stride(1),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        block_t,
        triton.next_power_of_2(K),
        triton.next_power_of_2(V),
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta


@triton.jit
def _qwen_gdn_t16_commit1_unpaired_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    a_log_neg_exp_ptr,
    dt_bias_ptr,
    state_in_ptr,
    state_out_ptr,
    out_ptr,
    accepted_tokens: tl.constexpr,
    scale: tl.constexpr,
    h_count: tl.constexpr,
    hv_count: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    bk: tl.constexpr,
    bv: tl.constexpr,
) -> None:
    pid_v = tl.program_id(0)
    hv_idx = tl.program_id(1)
    h_idx = hv_idx // 2

    offs_k = tl.arange(0, bk)
    offs_v = pid_v * bv + tl.arange(0, bv)

    out_base = (hv_idx * v_dim) + offs_v
    state_base = ((hv_idx * v_dim + offs_v[:, None]) * k_dim) + offs_k[None, :]
    h_state = tl.load(state_in_ptr + state_base).to(tl.float32)

    a_decay = tl.load(a_log_neg_exp_ptr + hv_idx).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + hv_idx).to(tl.float32)

    for token_idx in range(0, 16):
        q_base = ((token_idx * h_count + h_idx) * k_dim)
        q_vals = tl.load(q_ptr + q_base + offs_k).to(tl.float32)
        k_vals = tl.load(k_ptr + q_base + offs_k).to(tl.float32)
        q_vals = q_vals * tl.rsqrt(tl.sum(q_vals * q_vals, axis=0) + 1.0e-6)
        k_vals = k_vals * tl.rsqrt(tl.sum(k_vals * k_vals, axis=0) + 1.0e-6)
        q_vals = q_vals * scale

        v_base = ((token_idx * hv_count + hv_idx) * v_dim)
        v_vals = tl.load(v_ptr + v_base + offs_v).to(tl.float32)

        gate_base = token_idx * hv_count + hv_idx
        a_val = tl.load(a_ptr + gate_base).to(tl.float32)
        b_val = tl.load(b_ptr + gate_base).to(tl.float32)
        softplus_x = tl.log(1.0 + tl.exp(a_val + dt_bias))
        decay = tl.exp(a_decay * softplus_x)
        beta = tl.sigmoid(b_val).to(tl.float32)

        h_state *= decay
        projected = tl.sum(h_state * k_vals[None, :], axis=1)
        delta_v = (v_vals - projected) * beta
        h_state += delta_v[:, None] * k_vals[None, :]
        out_vals = tl.sum(h_state * q_vals[None, :], axis=1)
        tl.store(
            out_ptr + out_base + token_idx * hv_count * v_dim,
            out_vals.to(out_ptr.dtype.element_ty),
        )

        if token_idx == accepted_tokens - 1:
            tl.store(
                state_out_ptr + state_base,
                h_state.to(state_out_ptr.dtype.element_ty),
            )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        additional_config = get_current_vllm_config().additional_config
        assert isinstance(additional_config, dict)
        backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
        backend = str(backend_cfg).strip().lower()

        supports_flashinfer = (
            current_platform.is_cuda() and current_platform.is_device_capability(90)
        )

        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
            if not use_flashinfer:
                logger.warning_once(
                    "GDN prefill backend 'flashinfer' is selected but "
                    "cannot use this kernel on the current platform. "
                    "Falling back to Triton/FLA."
                )
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer

        if use_flashinfer:
            logger.info_once("Using FlashInfer GDN prefill kernel")
            logger.info_once(
                "FlashInfer GDN prefill kernel is JIT-compiled; first run may "
                "take a while to compile. Set `--gdn-prefill-backend triton` to "
                "avoid JIT compile time.",
            )
        else:
            logger.info_once("Using Triton/FLA GDN prefill kernel")

        self._forward_method = (
            self.forward_cuda if use_flashinfer else self.forward_native
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )


@PluggableLayer.register("gated_delta_net_attention")
class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout=False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix
        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.enable_dflash_ddtree_branch_gdn = (
            os.environ.get("VLLM_QWEN_GDN_DDTREE_BRANCH", "0") == "1"
        )
        self.enable_dflash_ddtree_packed_gdn = (
            os.environ.get("VLLM_QWEN_GDN_DDTREE_PACKED", "0") == "1"
        )
        self.enable_dflash_ddtree_spine_sidecar_gdn = (
            os.environ.get("VLLM_QWEN_GDN_DDTREE_SPINE_SIDECAR", "0") == "1"
        )
        self._dflash_ddtree_gdn_plan_cache = None
        self._dflash_ddtree_gdn_plan_tensor_cache = None
        self._dflash_ddtree_spine_sidecar_plan_tensor_cache = None
        self._dflash_gdn_row_trace_count = 0
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.has_lora_projections = not create_in_proj_qkvz
        if create_in_proj_qkvz:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
        else:
            # LoRA case (Qwen3.5 only): keep q/k/v and z as separate modules
            # so that LoRA adapters can be applied independently.
            self.in_proj_qkv = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            self.in_proj_z = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_z",
            )
        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.enable_qwen_gdn_fused_in_proj = (
            os.environ.get("VLLM_QWEN_GDN_FUSED_IN_PROJ", "0") == "1"
        )
        self.enable_qwen_gdn_out_proj_out_mm = (
            os.environ.get("VLLM_QWEN_GDN_OUT_PROJ_OUT_MM", "0") == "1"
        )
        self.enable_qwen_gdn_empty_core_out = (
            os.environ.get("VLLM_QWEN_GDN_EMPTY_CORE_OUT", "0") == "1"
        )
        self.enable_qwen_gdn_fp8_proj = (
            os.environ.get("VLLM_QWEN_GDN_FP8_PROJ", "0") == "1"
        )
        self.enable_qwen_gdn_nvfp4_proj = (
            os.environ.get("VLLM_QWEN_GDN_NVFP4_PROJ", "0") == "1"
        )
        self.enable_qwen_gdn_nvfp4_autotune = (
            os.environ.get("VLLM_QWEN_GDN_NVFP4_AUTOTUNE", "1") == "1"
        )
        self.enable_qwen_gdn_triton_proj = (
            os.environ.get("VLLM_QWEN_GDN_TRITON_PROJ", "0") == "1"
        )
        self.enable_qwen_gdn_flashinfer_mtp = (
            os.environ.get("VLLM_QWEN_GDN_FLASHINFER_MTP", "0") == "1"
        )
        self.enable_qwen_gdn_t16_commit1_unpaired = (
            os.environ.get("VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED", "0") == "1"
        )
        self.enable_qwen_gdn_short_prefill_recurrent = (
            os.environ.get("VLLM_QWEN_GDN_SHORT_PREFILL_RECURRENT", "0") == "1"
        )
        self.enable_qwen_gdn_fused_conv_prep = (
            os.environ.get("VLLM_QWEN_GDN_FUSED_CONV_PREP", "0") == "1"
        )
        self.qwen_gdn_short_prefill_max_tokens = int(
            os.environ.get("VLLM_QWEN_GDN_SHORT_PREFILL_MAX_TOKENS", "64")
        )
        self.qwen_gdn_fp8_proj_max_tokens = int(
            os.environ.get("VLLM_QWEN_GDN_FP8_PROJ_MAX_TOKENS", "64")
        )
        self.qwen_gdn_fp8_proj_targets = {
            target.strip()
            for target in os.environ.get(
                "VLLM_QWEN_GDN_FP8_PROJ_TARGETS",
                "in_proj_qkvz,out_proj",
            ).split(",")
            if target.strip()
        }
        self.qwen_gdn_nvfp4_proj_targets = {
            target.strip()
            for target in os.environ.get(
                "VLLM_QWEN_GDN_NVFP4_PROJ_TARGETS",
                "in_proj_qkvz,out_proj",
            ).split(",")
            if target.strip()
        }
        self.qwen_gdn_triton_proj_targets = {
            target.strip()
            for target in os.environ.get(
                "VLLM_QWEN_GDN_TRITON_PROJ_TARGETS",
                "in_proj_qkvz,out_proj",
            ).split(",")
            if target.strip()
        }
        self._qwen_gdn_fused_in_proj_weight: torch.Tensor | None = None
        self._qwen_gdn_fp8_weight_cache: dict[
            str,
            tuple[int, tuple[int, ...], torch.dtype, torch.device, torch.Tensor, torch.Tensor],
        ] = {}
        self._qwen_gdn_nvfp4_weight_cache: dict[
            str,
            tuple[
                int,
                tuple[int, ...],
                torch.dtype,
                torch.device,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
        ] = {}
        self._qwen_gdn_flashinfer_mtp_zero_index: torch.Tensor | None = None
        self._qwen_gdn_t16_a_log_neg_exp_cache: tuple[
            tuple[int, tuple[int, ...], torch.device], torch.Tensor
        ] | None = None
        self._qwen_gdn_t16_debug_count = 0

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _maybe_qwen_gdn_triton_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        cache_key: str,
    ) -> torch.Tensor | None:
        if not self.enable_qwen_gdn_triton_proj:
            return None
        if cache_key not in self.qwen_gdn_triton_proj_targets:
            return None
        if cache_key == "in_proj_qkvz":
            bm, bn, bk = 16, 64, 128
        elif cache_key == "out_proj":
            bm, bn, bk = 64, 32, 64
        else:
            return None
        if (
            x.dim() != 2
            or not x.is_cuda
            or x.shape[0] > self.qwen_gdn_fp8_proj_max_tokens
            or x.dtype != torch.bfloat16
        ):
            return None
        if (
            linear.bias is not None
            or linear.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(linear, "weight", None)
        if (
            weight is None
            or weight.dim() != 2
            or not weight.is_cuda
            or weight.dtype != x.dtype
            or weight.shape[1] != x.shape[-1]
        ):
            return None

        x = x.contiguous()
        b = weight.t()
        m_dim, k_dim = x.shape
        b_k_dim, n_dim = b.shape
        if k_dim != b_k_dim or k_dim % bk != 0:
            return None
        out = torch.empty((m_dim, n_dim), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(m_dim, bm), triton.cdiv(n_dim, bn))
        _qwen_gdn_small_m_matmul_kernel[grid](
            x,
            b,
            out,
            m_dim,
            n_dim,
            k_dim,
            x.stride(0),
            x.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=3,
        )
        return out

    def _maybe_qwen_gdn_fp8_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        cache_key: str,
    ) -> torch.Tensor | None:
        if not self.enable_qwen_gdn_fp8_proj:
            return None
        if cache_key not in self.qwen_gdn_fp8_proj_targets:
            return None
        if (
            x.dim() != 2
            or not x.is_cuda
            or x.shape[0] > self.qwen_gdn_fp8_proj_max_tokens
            or x.shape[-1] % 128 != 0
            or x.dtype not in (torch.bfloat16, torch.float16)
        ):
            return None
        if (
            linear.bias is not None
            or linear.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(linear, "weight", None)
        if (
            weight is None
            or weight.dim() != 2
            or not weight.is_cuda
            or weight.dtype != x.dtype
            or weight.shape[1] != x.shape[-1]
            or weight.shape[0] % 128 != 0
        ):
            return None

        weight_id = weight.data_ptr()
        weight_shape = tuple(weight.shape)
        cached = self._qwen_gdn_fp8_weight_cache.get(cache_key)
        if (
            cached is None
            or cached[0] != weight_id
            or cached[1] != weight_shape
            or cached[2] != weight.dtype
            or cached[3] != weight.device
        ):
            block = 128
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            b = weight.t()
            k_dim, n_dim = b.shape
            if k_dim % block != 0 or n_dim % block != 0:
                return None
            b_view = b.float().reshape(k_dim // block, block, n_dim // block, block)
            b_scale = (
                b_view.abs().amax(dim=(1, 3)).div(fp8_max).clamp_min(1e-12).contiguous()
            )
            b_q = (b_view / b_scale[:, None, :, None]).reshape(k_dim, n_dim)
            # CUTLASS expects B as column-major [K, N], i.e. stride(0) == 1.
            b_q = b_q.to(torch.float8_e4m3fn).t().contiguous().t()
            cached = (weight_id, weight_shape, weight.dtype, weight.device, b_q, b_scale)
            self._qwen_gdn_fp8_weight_cache[cache_key] = cached

        _, _, _, _, b_q, b_scale = cached
        x_q, x_scale = per_token_group_quant_fp8(
            x.contiguous(),
            128,
            use_ue8m0=False,
        )
        return ops.cutlass_scaled_mm(x_q, b_q, x_scale, b_scale, x.dtype)

    def _maybe_qwen_gdn_nvfp4_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        cache_key: str,
    ) -> torch.Tensor | None:
        if not self.enable_qwen_gdn_nvfp4_proj:
            return None
        if cache_key not in self.qwen_gdn_nvfp4_proj_targets:
            return None
        if (
            x.dim() != 2
            or not x.is_cuda
            or x.shape[0] > self.qwen_gdn_fp8_proj_max_tokens
            or x.shape[-1] % 16 != 0
            or x.dtype not in (torch.bfloat16, torch.float16)
            or not has_flashinfer()
        ):
            return None
        if (
            linear.bias is not None
            or linear.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(linear, "weight", None)
        if (
            weight is None
            or weight.dim() != 2
            or not weight.is_cuda
            or weight.dtype != x.dtype
            or weight.shape[1] != x.shape[-1]
            or weight.shape[1] % 16 != 0
        ):
            return None

        weight_id = weight.data_ptr()
        weight_shape = tuple(weight.shape)
        cached = self._qwen_gdn_nvfp4_weight_cache.get(cache_key)
        if (
            cached is None
            or cached[0] != weight_id
            or cached[1] != weight_shape
            or cached[2] != weight.dtype
            or cached[3] != weight.device
        ):
            weight_f = weight.detach().contiguous()
            weight_absmax = weight_f.abs().amax().clamp_min(1e-6)
            weight_global_scale_inv = (2688.0 / weight_absmax).to(torch.float32)
            weight_q, weight_scale = ops.scaled_fp4_quant(
                weight_f,
                weight_global_scale_inv,
                is_sf_swizzled_layout=False,
                backend="flashinfer-cutlass",
            )
            weight_scale = swizzle_blockscale(weight_scale)
            weight_q, weights_padding_cols = pad_nvfp4_weight_for_cutlass(weight_q)

            input_global_scale_inv = torch.ones(
                (), dtype=torch.float32, device=x.device
            )
            alpha = (1.0 / weight_global_scale_inv).to(torch.float32)
            cached = (
                weight_id,
                weight_shape,
                weight.dtype,
                weight.device,
                weight_q,
                weight_scale,
                input_global_scale_inv,
                alpha,
                weights_padding_cols,
            )
            self._qwen_gdn_nvfp4_weight_cache[cache_key] = cached
            logger.info_once(
                "Qwen GDN NVFP4 projection path active for %s: weight=%s",
                cache_key,
                weight_shape,
            )

        (
            _,
            _,
            _,
            _,
            weight_q,
            weight_scale,
            input_global_scale_inv,
            alpha,
            weights_padding_cols,
        ) = cached
        x_q, x_scale = ops.scaled_fp4_quant(
            x.contiguous(),
            input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
        )
        x_q = pad_nvfp4_activation_for_cutlass(x_q, weights_padding_cols)

        def run_fp4_mm() -> torch.Tensor:
            return flashinfer_scaled_fp4_mm(
                x_q,
                weight_q,
                x_scale,
                weight_scale,
                alpha,
                x.dtype,
                backend="cutlass",
            )

        if self.enable_qwen_gdn_nvfp4_autotune:
            with flashinfer_autotune(
                True,
                tuning_buckets=(8, 16, 32, 64),
                round_up=True,
            ):
                out = run_fp4_mm()
        else:
            out = run_fp4_mm()
        return slice_nvfp4_output(out, weight_shape[0])

    def _maybe_qwen_gdn_fused_in_proj(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self.enable_qwen_gdn_fused_in_proj
            or self.has_lora_projections
            or self.gqa_interleaved_layout
        ):
            return None

        qkvz_weight = getattr(self.in_proj_qkvz, "weight", None)
        ba_weight = getattr(self.in_proj_ba, "weight", None)
        qkvz_bias = getattr(self.in_proj_qkvz, "bias", None)
        ba_bias = getattr(self.in_proj_ba, "bias", None)
        if (
            qkvz_weight is None
            or ba_weight is None
            or qkvz_bias is not None
            or ba_bias is not None
            or qkvz_weight.dim() != 2
            or ba_weight.dim() != 2
            or qkvz_weight.shape[1] != ba_weight.shape[1]
        ):
            return None

        fused_weight = self._qwen_gdn_fused_in_proj_weight
        fused_out_dim = qkvz_weight.shape[0] + ba_weight.shape[0]
        if (
            fused_weight is None
            or fused_weight.device != qkvz_weight.device
            or fused_weight.dtype != qkvz_weight.dtype
            or fused_weight.shape != (fused_out_dim, qkvz_weight.shape[1])
        ):
            fused_weight = torch.cat((qkvz_weight, ba_weight), dim=0)
            self._qwen_gdn_fused_in_proj_weight = fused_weight

        projected = F.linear(hidden_states, fused_weight)
        return projected.split([qkvz_weight.shape[0], ba_weight.shape[0]], dim=-1)

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    @torch.compile(fullgraph=True)
    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def _get_dflash_ddtree_gdn_plan(self):
        cached = self._dflash_ddtree_gdn_plan_cache
        profile = os.environ.get("VLLM_DFLASH_DDTREE_PROFILE", "b2seq_s12")
        sidecars_env = os.environ.get("VLLM_DFLASH_DDTREE_SIDECARS", "")
        spec_tree = (
            self.speculative_config.speculative_token_tree
            if self.speculative_config is not None
            else None
        )
        cache_key = (self.num_spec, profile, sidecars_env, spec_tree)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]

        if self.num_spec <= 0:
            parent_rows = []
            depth_groups = []
        else:
            if spec_tree is not None:
                choices = [tuple(path) for path in ast.literal_eval(spec_tree)]
            else:
                profile = profile.strip().lower()
                choices: list[tuple[int, ...]] = []
                if profile in ("root2_chain", "root2-chain", "root2"):
                    depth = 1
                    while len(choices) < self.num_spec:
                        choices.append(depth * (0,))
                        if len(choices) == self.num_spec:
                            break
                        choices.append((1,) + (depth - 1) * (0,))
                        depth += 1
                elif profile in (
                    "spine_sidecar",
                    "spine-sidecar",
                    "sidecar",
                    "spine2",
                ):
                    max_sidecars = max(0, (self.num_spec - 1) // 2)
                    if sidecars_env:
                        try:
                            sidecars = int(sidecars_env)
                        except ValueError as exc:
                            raise ValueError(
                                "VLLM_DFLASH_DDTREE_SIDECARS must be an integer"
                            ) from exc
                    else:
                        sidecars = min(4, max_sidecars)
                    sidecars = max(0, min(sidecars, max_sidecars))
                    spine_len = self.num_spec - sidecars
                    for depth in range(1, spine_len + 1):
                        choices.append(depth * (0,))
                        if depth <= sidecars and len(choices) < self.num_spec:
                            choices.append((depth - 1) * (0,) + (1,))
                else:
                    frontier: list[tuple[int, ...]] = [()]
                    depth = 0
                    while len(choices) < self.num_spec:
                        width = 2 if depth < 2 else 1
                        next_frontier: list[tuple[int, ...]] = []
                        for parent in frontier:
                            for rank in range(width):
                                path = parent + (rank,)
                                choices.append(path)
                                next_frontier.append(path)
                                if len(choices) == self.num_spec:
                                    break
                            if len(choices) == self.num_spec:
                                break
                        frontier = next_frontier
                        depth += 1
            if not choices:
                raise RuntimeError(
                    "DFlash DDTree GDN plan requires at least one choice"
                )

            path_to_row: dict[tuple[int, ...], int] = {(): 0}
            parent_rows = [-1]
            depths = [0]
            for row, path in enumerate(choices, start=1):
                path = tuple(path)
                path_to_row[path] = row
                parent_rows.append(path_to_row[path[:-1]])
                depths.append(len(path))

            depth_groups = []
            for depth in range(max(depths) + 1):
                group = [row for row, row_depth in enumerate(depths) if row_depth == depth]
                if group:
                    depth_groups.append(group)

        self._dflash_ddtree_gdn_plan_cache = (cache_key, parent_rows, depth_groups)
        return parent_rows, depth_groups

    def _get_dflash_ddtree_gdn_plan_tensors(self, device: torch.device):
        profile = os.environ.get("VLLM_DFLASH_DDTREE_PROFILE", "b2seq_s12")
        sidecars_env = os.environ.get("VLLM_DFLASH_DDTREE_SIDECARS", "")
        spec_tree = (
            self.speculative_config.speculative_token_tree
            if self.speculative_config is not None
            else None
        )
        cache_key = (self.num_spec, profile, sidecars_env, spec_tree, device)
        cached = self._dflash_ddtree_gdn_plan_tensor_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2], cached[3]

        parent_rows, depth_groups = self._get_dflash_ddtree_gdn_plan()
        group_plans = []
        all_contiguous = True
        for group in depth_groups:
            start = group[0]
            length = len(group)
            contiguous = group == list(range(start, start + length))
            all_contiguous = all_contiguous and contiguous
            group_parent_rows = [parent_rows[row] for row in group]
            group_plans.append(
                (
                    start,
                    length,
                    torch.tensor(
                        group_parent_rows,
                        dtype=torch.long,
                        device=device,
                    ),
                    group_parent_rows[0] < 0,
                )
            )

        cached = (cache_key, depth_groups, group_plans, all_contiguous)
        self._dflash_ddtree_gdn_plan_tensor_cache = cached
        return depth_groups, group_plans, all_contiguous

    def _get_dflash_ddtree_spine_sidecar_plan_tensors(
        self,
        device: torch.device,
    ):
        profile = os.environ.get("VLLM_DFLASH_DDTREE_PROFILE", "b2seq_s12")
        sidecars_env = os.environ.get("VLLM_DFLASH_DDTREE_SIDECARS", "")
        spec_tree = (
            self.speculative_config.speculative_token_tree
            if self.speculative_config is not None
            else None
        )
        cache_key = (self.num_spec, profile, sidecars_env, spec_tree, device)
        cached = self._dflash_ddtree_spine_sidecar_plan_tensor_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        if self.num_spec <= 0 or spec_tree is None:
            self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (
                cache_key,
                None,
            )
            return None

        choices = [tuple(path) for path in ast.literal_eval(spec_tree)]
        if not choices:
            self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (
                cache_key,
                None,
            )
            return None

        path_to_row: dict[tuple[int, ...], int] = {(): 0}
        for row, path in enumerate(choices, start=1):
            path_to_row[path] = row

        spine_rows = [0]
        sidecar_rows: list[int] = []
        sidecar_parent_rows: list[int] = []
        for row, path in enumerate(choices, start=1):
            if all(rank == 0 for rank in path):
                spine_rows.append(row)
                continue
            if path and all(rank == 0 for rank in path[:-1]):
                parent_row = path_to_row.get(path[:-1])
                if parent_row is None:
                    self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (
                        cache_key,
                        None,
                    )
                    return None
                sidecar_rows.append(row)
                sidecar_parent_rows.append(parent_row)
                continue
            self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (
                cache_key,
                None,
            )
            return None

        if len(spine_rows) + len(sidecar_rows) != len(choices) + 1:
            self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (
                cache_key,
                None,
            )
            return None

        plan = (
            torch.tensor(spine_rows, dtype=torch.long, device=device),
            torch.tensor(sidecar_rows, dtype=torch.long, device=device),
            torch.tensor(sidecar_parent_rows, dtype=torch.long, device=device),
        )
        self._dflash_ddtree_spine_sidecar_plan_tensor_cache = (cache_key, plan)
        return plan

    def _use_dflash_ddtree_branch_gdn(
        self,
        attn_metadata: GDNAttentionMetadata,
    ) -> bool:
        return (
            self.enable_dflash_ddtree_branch_gdn
            and os.environ.get("VLLM_DFLASH_DDTREE", "0") == "1"
            and self.num_spec > 0
            and attn_metadata.num_spec_decodes > 0
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.spec_state_indices_tensor is not None
            and attn_metadata.spec_query_start_loc is not None
            and attn_metadata.num_accepted_tokens is not None
        )

    def _maybe_log_dflash_gdn_row_trace(
        self,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        trace_path = os.environ.get("VLLM_DFLASH_TELEMETRY_PATH", "")
        if (
            not trace_path
            or self.layer_idx != 0
            or os.environ.get("VLLM_DFLASH_DDTREE", "0") != "1"
            or self.num_spec <= 0
            or attn_metadata.num_spec_decodes <= 0
        ):
            return

        try:
            max_rows = max(
                0, int(os.environ.get("VLLM_DFLASH_TELEMETRY_MAX_ROWS", "0") or "0")
            )
        except ValueError:
            max_rows = 0
        if max_rows > 0 and self._dflash_gdn_row_trace_count >= max_rows:
            return

        old_rows_per_spec_request = self.num_spec + 1
        if attn_metadata.num_spec_decodes > 0:
            actual_rows_per_spec_request = (
                attn_metadata.num_spec_decode_tokens
                // attn_metadata.num_spec_decodes
            )
        else:
            actual_rows_per_spec_request = 0
        old_gdn_rows = old_rows_per_spec_request * attn_metadata.num_spec_decodes
        avoided_rows = max(0, old_gdn_rows - attn_metadata.num_spec_decode_tokens)
        record = {
            "type": "dflash_gdn_row_trace",
            "ts": time.time(),
            "code_path": "GatedDeltaNetAttention._forward_core",
            "layer_idx": self.layer_idx,
            "gdn_verifier_rows_per_pass": int(attn_metadata.num_spec_decode_tokens),
            "old_gdn_verifier_rows_per_pass": int(old_gdn_rows),
            "actual_rows_per_spec_request": int(actual_rows_per_spec_request),
            "old_rows_per_spec_request": int(old_rows_per_spec_request),
            "gdn_sidecar_rows_avoided_before_forward": int(avoided_rows),
            "num_spec_decodes": int(attn_metadata.num_spec_decodes),
            "num_prefills": int(attn_metadata.num_prefills),
            "num_decodes": int(attn_metadata.num_decodes),
            "gdn_num_actual_tokens": int(attn_metadata.num_actual_tokens),
            "branch_gdn_enabled": bool(
                self._use_dflash_ddtree_branch_gdn(attn_metadata)
            ),
            "cpu_sync_introduced": False,
        }
        try:
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._dflash_gdn_row_trace_count += 1
        except OSError:
            logger.warning_once(
                "Failed to write DFlash GDN row trace to %s",
                trace_path,
            )

    def _forward_dflash_ddtree_spine_sidecar_gdn_spec(
        self,
        mixed_qkv_spec: torch.Tensor,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spine_rows: torch.Tensor,
        sidecar_rows: torch.Tensor,
        sidecar_parent_rows: torch.Tensor,
    ) -> torch.Tensor:
        rows_per_req = mixed_qkv_spec.shape[0]
        state_rows = spec_state_indices_tensor[0].to(torch.long)
        prev_state_col = (num_accepted_tokens[:1].to(torch.long) - 1).clamp(
            min=0,
            max=rows_per_req - 1,
        )
        root_parent_block = state_rows.gather(0, prev_state_col)
        core_attn_out_spec = mixed_qkv_spec.new_empty(
            (
                1,
                rows_per_req,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
        )

        spine_blocks = state_rows.index_select(0, spine_rows)
        spine_inputs = mixed_qkv_spec.index_select(0, spine_rows)
        spine_conv_outputs = []
        parent_block = root_parent_block
        for spine_index in range(spine_rows.numel()):
            current_block = spine_blocks.narrow(0, spine_index, 1)
            conv_state[current_block] = conv_state[parent_block]
            x_row = causal_conv1d_update(
                spine_inputs.narrow(0, spine_index, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=current_block,
                validate_data=False,
            )
            spine_conv_outputs.append(x_row)
            parent_block = current_block

        x_spine = torch.cat(spine_conv_outputs, dim=0)
        ssm_state[spine_blocks.narrow(0, 0, 1)] = ssm_state[root_parent_block]
        query_spine, key_spine, value_spine = self.rearrange_mixed_qkv(x_spine)
        spine_state_indices = spine_blocks.view(1, -1)
        spine_num_accepted = num_accepted_tokens.new_ones((1,))
        core_spine, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a_spec.index_select(0, spine_rows).unsqueeze(0),
            b=b_spec.index_select(0, spine_rows).unsqueeze(0),
            dt_bias=self.dt_bias,
            q=query_spine,
            k=key_spine,
            v=value_spine,
            initial_state=ssm_state,
            inplace_final_state=True,
            ssm_state_indices=spine_state_indices,
            num_accepted_tokens=spine_num_accepted,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out_spec[:, spine_rows] = core_spine

        if sidecar_rows.numel() > 0:
            sidecar_blocks = state_rows.index_select(0, sidecar_rows)
            sidecar_parent_blocks = state_rows.index_select(0, sidecar_parent_rows)
            conv_state[sidecar_blocks] = conv_state[sidecar_parent_blocks]
            x_sidecar = mixed_qkv_spec.index_select(0, sidecar_rows)
            x_sidecar = causal_conv1d_update(
                x_sidecar,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=sidecar_blocks,
                validate_data=False,
            )

            ssm_state[sidecar_blocks] = ssm_state[sidecar_parent_blocks]
            core_sidecar = mixed_qkv_spec.new_empty(
                (
                    sidecar_rows.numel(),
                    1,
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                )
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=x_sidecar,
                a=a_spec.index_select(0, sidecar_rows),
                b=b_spec.index_select(0, sidecar_rows),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                out=core_sidecar,
                ssm_state_indices=sidecar_blocks,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out_spec[:, sidecar_rows] = core_sidecar.transpose(0, 1)

        return core_attn_out_spec

    def _forward_dflash_ddtree_branch_gdn_spec(
        self,
        mixed_qkv_spec: torch.Tensor,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = spec_state_indices_tensor.shape[0]
        if num_reqs <= 0 or mixed_qkv_spec.shape[0] % num_reqs != 0:
            raise RuntimeError(
                "DFlash DDTree GDN branch path expects equal compact rows "
                f"per request; got {mixed_qkv_spec.shape[0]} rows for "
                f"{num_reqs} requests"
            )
        rows_per_req = mixed_qkv_spec.shape[0] // num_reqs
        total_rows = num_reqs * rows_per_req
        if mixed_qkv_spec.shape[0] != total_rows:
            raise RuntimeError(
                "DFlash DDTree GDN branch path expects fixed tree rows; "
                f"got {mixed_qkv_spec.shape[0]} rows for {num_reqs} requests"
            )

        parent_rows, depth_groups = self._get_dflash_ddtree_gdn_plan()
        if len(parent_rows) != rows_per_req:
            raise RuntimeError(
                f"DFlash DDTree GDN plan has {len(parent_rows)} rows, "
                f"expected {rows_per_req}"
            )

        device = mixed_qkv_spec.device
        req_offsets = (
            torch.arange(num_reqs, device=device, dtype=torch.long) * rows_per_req
        )
        prev_state_cols = (num_accepted_tokens.to(torch.long) - 1).clamp(
            min=0,
            max=rows_per_req - 1,
        )
        root_parent_blocks = spec_state_indices_tensor.gather(
            1,
            prev_state_cols[:, None],
        ).squeeze(1)
        core_attn_out_spec = mixed_qkv_spec.new_empty(
            (
                1,
                total_rows,
                self.num_v_heads // self.tp_size,
                self.head_v_dim,
            )
        )
        if self.enable_dflash_ddtree_spine_sidecar_gdn and num_reqs == 1:
            spine_sidecar_plan = (
                self._get_dflash_ddtree_spine_sidecar_plan_tensors(device)
            )
            if spine_sidecar_plan is not None:
                spine_rows, sidecar_rows, sidecar_parent_rows = spine_sidecar_plan
                return self._forward_dflash_ddtree_spine_sidecar_gdn_spec(
                    mixed_qkv_spec=mixed_qkv_spec,
                    a_spec=a_spec,
                    b_spec=b_spec,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    conv_weights=conv_weights,
                    spec_state_indices_tensor=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    spine_rows=spine_rows,
                    sidecar_rows=sidecar_rows,
                    sidecar_parent_rows=sidecar_parent_rows,
                )

        use_packed_c1_tree = self.enable_dflash_ddtree_packed_gdn and num_reqs == 1
        group_plans = None
        if use_packed_c1_tree:
            _, group_plans, all_contiguous = self._get_dflash_ddtree_gdn_plan_tensors(
                device,
            )
            use_packed_c1_tree = all_contiguous

        if use_packed_c1_tree and group_plans is not None:
            state_rows = spec_state_indices_tensor[0]
            for start, length, parent_group_rows, has_root_parent in group_plans:
                current_blocks = state_rows.narrow(0, start, length)
                if has_root_parent:
                    parent_blocks = root_parent_blocks.expand(length)
                else:
                    parent_blocks = state_rows.index_select(0, parent_group_rows)

                conv_state[current_blocks] = conv_state[parent_blocks]
                x_depth = mixed_qkv_spec.narrow(0, start, length)
                x_depth = causal_conv1d_update(
                    x_depth,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=current_blocks,
                    validate_data=False,
                )

                ssm_state[current_blocks] = ssm_state[parent_blocks]
                core_depth = mixed_qkv_spec.new_empty(
                    (
                        length,
                        1,
                        self.num_v_heads // self.tp_size,
                        self.head_v_dim,
                    )
                )
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=x_depth,
                    a=a_spec.narrow(0, start, length),
                    b=b_spec.narrow(0, start, length),
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    scale=self.head_k_dim**-0.5,
                    initial_state=ssm_state,
                    out=core_depth,
                    ssm_state_indices=current_blocks,
                    use_qk_l2norm_in_kernel=True,
                )
                core_attn_out_spec[
                    :,
                    start : start + length,
                ] = core_depth.transpose(0, 1)

            return core_attn_out_spec

        for group in depth_groups:
            group_rows = torch.tensor(group, dtype=torch.long, device=device)
            global_rows = (req_offsets[:, None] + group_rows[None, :]).reshape(-1)
            current_blocks = spec_state_indices_tensor.index_select(
                1,
                group_rows,
            ).reshape(-1)

            parent_block_columns = []
            for row in group:
                parent_row = parent_rows[row]
                if parent_row < 0:
                    parent_block_columns.append(root_parent_blocks)
                else:
                    parent_block_columns.append(spec_state_indices_tensor[:, parent_row])
            parent_blocks = torch.stack(parent_block_columns, dim=1).reshape(-1)

            conv_state[current_blocks] = conv_state[parent_blocks]
            x_depth = mixed_qkv_spec.index_select(0, global_rows)
            x_depth = causal_conv1d_update(
                x_depth,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=current_blocks,
                validate_data=False,
            )

            query_depth, key_depth, value_depth = self.rearrange_mixed_qkv(x_depth)
            initial_state = ssm_state.index_select(0, parent_blocks).contiguous()
            depth_query_start_loc = torch.arange(
                0,
                current_blocks.numel() + 1,
                dtype=torch.int32,
                device=device,
            )
            core_depth, final_state = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a_spec.index_select(0, global_rows),
                b=b_spec.index_select(0, global_rows),
                dt_bias=self.dt_bias,
                q=query_depth,
                k=key_depth,
                v=value_depth,
                initial_state=initial_state,
                inplace_final_state=False,
                cu_seqlens=depth_query_start_loc,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[current_blocks] = final_state.to(ssm_state.dtype)
            core_attn_out_spec[:, global_rows] = core_depth

        return core_attn_out_spec

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        self._forward_method(hidden_states, output)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ):
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        if self._maybe_qwen_gdn_out_proj_out_mm(core_attn_out, output, num_tokens):
            return
        nvfp4_out = self._maybe_qwen_gdn_nvfp4_linear(
            self.out_proj,
            core_attn_out,
            "out_proj",
        )
        if nvfp4_out is not None:
            output[:num_tokens] = nvfp4_out
            return
        triton_out = self._maybe_qwen_gdn_triton_linear(
            self.out_proj,
            core_attn_out,
            "out_proj",
        )
        if triton_out is not None:
            output[:num_tokens] = triton_out
            return
        fp8_out = self._maybe_qwen_gdn_fp8_linear(
            self.out_proj,
            core_attn_out,
            "out_proj",
        )
        if fp8_out is not None:
            output[:num_tokens] = fp8_out
            return
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _maybe_qwen_gdn_out_proj_out_mm(
        self,
        core_attn_out: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ) -> bool:
        if not self.enable_qwen_gdn_out_proj_out_mm:
            return False
        if (
            self.tp_size != 1
            or self.out_proj.bias is not None
            or (self.out_proj.reduce_results and self.out_proj.tp_size > 1)
            or self.out_proj.quant_method.__class__.__name__
            != "UnquantizedLinearMethod"
        ):
            return False

        output_slice = output[:num_tokens]
        weight = getattr(self.out_proj, "weight", None)
        if (
            weight is None
            or core_attn_out.dim() != 2
            or output_slice.dim() != 2
            or core_attn_out.shape[0] != output_slice.shape[0]
            or core_attn_out.shape[1] != weight.shape[1]
            or output_slice.shape[1] != weight.shape[0]
            or core_attn_out.dtype != weight.dtype
            or output_slice.dtype != weight.dtype
            or not output_slice.is_contiguous()
        ):
            return False

        torch.mm(core_attn_out, weight.t(), out=output_slice)
        return True

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if not self.has_lora_projections and GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                fast_kernel=True,
                layer_name=_encode_layer_name(self.prefix),
            )

            self._output_projection(core_attn_out, z, output, num_tokens)
        else:
            self.forward_cuda(hidden_states, output)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if self.has_lora_projections:
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            fused_in_proj = self._maybe_qwen_gdn_fused_in_proj(hidden_states)
            if fused_in_proj is None:
                mixed_qkvz = self._maybe_qwen_gdn_nvfp4_linear(
                    self.in_proj_qkvz,
                    hidden_states,
                    "in_proj_qkvz",
                )
                if mixed_qkvz is None:
                    mixed_qkvz = self._maybe_qwen_gdn_triton_linear(
                        self.in_proj_qkvz,
                        hidden_states,
                        "in_proj_qkvz",
                    )
                if mixed_qkvz is None:
                    mixed_qkvz = self._maybe_qwen_gdn_fp8_linear(
                        self.in_proj_qkvz,
                        hidden_states,
                        "in_proj_qkvz",
                    )
                if mixed_qkvz is None:
                    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                ba = self._maybe_qwen_gdn_nvfp4_linear(
                    self.in_proj_ba,
                    hidden_states,
                    "in_proj_ba",
                )
                if ba is None:
                    ba, _ = self.in_proj_ba(hidden_states)
            else:
                mixed_qkvz, ba = fused_in_proj

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_out_shape = (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim)
        if self.enable_qwen_gdn_empty_core_out:
            core_attn_out = torch.empty(
                core_out_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            core_attn_out = torch.zeros(
                core_out_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            fast_kernel=False,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        self._output_projection(core_attn_out, z, output, num_tokens)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        assert not self.has_lora_projections, "lora isn't supported on XPU."

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if hasattr(self, "_prefill_kernels_warmed_up"):
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del dummy_mixed_qkv, q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill), dispatches directly to
        ``_forward_core_decode_fast``.  Otherwise unpacks the packed
        layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_fast(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        z_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _maybe_forward_qwen_gdn_flashinfer_mtp_spec(
        self,
        query_spec: torch.Tensor | None,
        key_spec: torch.Tensor | None,
        value_spec: torch.Tensor | None,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        ssm_state: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor | None:
        if not self.enable_qwen_gdn_flashinfer_mtp:
            return None
        if query_spec is None or key_spec is None or value_spec is None:
            return None
        if attn_metadata.num_prefills != 0 or attn_metadata.num_decodes != 0:
            return None
        if attn_metadata.num_spec_decodes != 1:
            return None
        if (
            attn_metadata.flashinfer_mtp_state_base is None
            or attn_metadata.flashinfer_mtp_state_root is None
        ):
            return None
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            return None
        if ssm_state.dim() != 4 or ssm_state.stride(-1) != 1:
            return None

        rows = query_spec.shape[1]
        num_v_heads = self.num_v_heads // self.tp_size
        if rows != self.num_spec + 1:
            return None
        if a_spec.shape[0] != rows or b_spec.shape[0] != rows:
            return None

        base = attn_metadata.flashinfer_mtp_state_base
        root = attn_metadata.flashinfer_mtp_state_root
        if base < 0 or root < 0 or base + rows > ssm_state.shape[0]:
            return None

        state_window = ssm_state.narrow(0, base, rows)
        root_state = ssm_state.narrow(0, root, 1)
        if not state_window.is_contiguous() or not root_state.is_contiguous():
            return None

        zero_index = self._qwen_gdn_flashinfer_mtp_zero_index
        if zero_index is None or zero_index.device != query_spec.device:
            zero_index = torch.zeros(1, dtype=torch.int32, device=query_spec.device)
            self._qwen_gdn_flashinfer_mtp_zero_index = zero_index

        output = core_attn_out[:rows].view(
            1, rows, num_v_heads, self.head_v_dim
        )
        a_mtp = a_spec.detach().contiguous().view(1, rows, num_v_heads)
        b_mtp = b_spec.detach().contiguous().view(1, rows, num_v_heads)
        state_buffer = state_window.detach().view(
            1, rows, num_v_heads, self.head_v_dim, self.head_k_dim
        )

        try:
            if ssm_state.dtype == torch.bfloat16:
                from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                    gated_delta_rule_mtp as flashinfer_gdn_mtp,
                )

                result = flashinfer_gdn_mtp(
                    A_log=self.A_log.detach(),
                    a=a_mtp,
                    dt_bias=self.dt_bias.detach(),
                    q=query_spec.detach().contiguous(),
                    k=key_spec.detach().contiguous(),
                    v=value_spec.detach().contiguous(),
                    b=b_mtp,
                    initial_state_source=root_state.detach(),
                    initial_state_indices=zero_index,
                    intermediate_states_buffer=state_buffer,
                    disable_state_update=True,
                    use_qk_l2norm_in_kernel=True,
                    scale=self.head_k_dim**-0.5,
                    output=output,
                )
                if result is not output:
                    output.copy_(result)
            elif ssm_state.dtype == torch.float32:
                from flashinfer.gdn_decode import gated_delta_rule_mtp

                output, _ = gated_delta_rule_mtp(
                    q=query_spec.detach().contiguous(),
                    k=key_spec.detach().contiguous(),
                    v=value_spec.detach().contiguous(),
                    initial_state=root_state.detach(),
                    initial_state_indices=zero_index,
                    A_log=self.A_log.detach(),
                    a=a_mtp,
                    dt_bias=self.dt_bias.detach(),
                    b=b_mtp,
                    scale=self.head_k_dim**-0.5,
                    output=output,
                    intermediate_states_buffer=state_buffer,
                    disable_state_update=True,
                    use_qk_l2norm=True,
                )
            else:
                return None
        except Exception as exc:
            logger.warning_once(
                "Qwen GDN FlashInfer MTP path failed; falling back to FLA: %s",
                exc,
            )
            return None

        if self.layer_idx == 0:
            logger.info_once(
                "Qwen GDN FlashInfer MTP path active: rows=%s state_dtype=%s",
                rows,
                ssm_state.dtype,
            )
        return output

    def _qwen_gdn_t16_a_log_neg_exp(self) -> torch.Tensor:
        key = (self.A_log.data_ptr(), tuple(self.A_log.shape), self.A_log.device)
        cached = self._qwen_gdn_t16_a_log_neg_exp_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        value = (-torch.exp(self.A_log.detach().float())).contiguous()
        self._qwen_gdn_t16_a_log_neg_exp_cache = (key, value)
        return value

    def _maybe_forward_qwen_gdn_t16_commit1_unpaired_spec(
        self,
        query_spec: torch.Tensor | None,
        key_spec: torch.Tensor | None,
        value_spec: torch.Tensor | None,
        a_spec: torch.Tensor,
        b_spec: torch.Tensor,
        ssm_state: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor | None:
        def debug_skip(reason: str) -> None:
            if self.layer_idx != 0 or self._qwen_gdn_t16_debug_count >= 64:
                return
            accepted_cpu = attn_metadata.num_accepted_tokens_cpu
            accepted = None
            if accepted_cpu is not None and accepted_cpu.numel() > 0:
                accepted = int(accepted_cpu[0].item())
            state_indices = attn_metadata.spec_state_indices_tensor
            logger.info(
                "Qwen GDN T16 skip %s: spec_decodes=%s prefills=%s decodes=%s "
                "accepted=%s num_spec=%s q=%s k=%s v=%s a=%s b=%s "
                "state_indices=%s ssm=%s ssm_stride=%s",
                reason,
                attn_metadata.num_spec_decodes,
                attn_metadata.num_prefills,
                attn_metadata.num_decodes,
                accepted,
                self.num_spec,
                tuple(query_spec.shape) if query_spec is not None else None,
                tuple(key_spec.shape) if key_spec is not None else None,
                tuple(value_spec.shape) if value_spec is not None else None,
                tuple(a_spec.shape),
                tuple(b_spec.shape),
                tuple(state_indices.shape) if state_indices is not None else None,
                tuple(ssm_state.shape),
                ssm_state.stride(),
            )
            self._qwen_gdn_t16_debug_count += 1

        if not self.enable_qwen_gdn_t16_commit1_unpaired:
            return None
        if query_spec is None or key_spec is None or value_spec is None:
            debug_skip("missing_qkv")
            return None
        if attn_metadata.num_prefills != 0 or attn_metadata.num_decodes != 0:
            debug_skip("non_spec_batch")
            return None
        if attn_metadata.num_spec_decodes != 1 or self.num_spec != 15:
            debug_skip("unsupported_batch_or_num_spec")
            return None
        if attn_metadata.num_accepted_tokens_cpu is None:
            debug_skip("missing_accepted_cpu")
            return None
        accepted = int(attn_metadata.num_accepted_tokens_cpu[0].item())
        if accepted < 1 or accepted > 16:
            debug_skip("bad_accepted")
            return None
        state_indices = attn_metadata.spec_state_indices_tensor
        if state_indices is None or state_indices.shape != (1, 16):
            debug_skip("bad_state_indices")
            return None
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            debug_skip("bad_head_dim")
            return None
        if ssm_state.dim() != 4 or ssm_state.stride(-1) != 1:
            debug_skip("bad_ssm_state")
            return None

        rows = query_spec.shape[1]
        h_count = self.num_k_heads // self.tp_size
        hv_count = self.num_v_heads // self.tp_size
        if (
            rows != 16
            or query_spec.shape != (1, 16, h_count, 128)
            or key_spec.shape != (1, 16, h_count, 128)
            or value_spec.shape != (1, 16, hv_count, 128)
            or hv_count != h_count * 2
            or a_spec.shape != (16, hv_count)
            or b_spec.shape != (16, hv_count)
        ):
            debug_skip("bad_shape")
            return None

        commit_index = state_indices[:, accepted - 1].to(torch.long)
        root_state = ssm_state.index_select(0, commit_index).contiguous()
        if root_state.dtype != torch.float16:
            root_state = root_state.to(torch.float16)

        output = core_attn_out[:16].view(1, 16, hv_count, 128)
        state_out = torch.empty(
            (1, hv_count, 128, 128),
            dtype=torch.float16,
            device=query_spec.device,
        )
        block_v = 8
        _qwen_gdn_t16_commit1_unpaired_kernel[(triton.cdiv(128, block_v), hv_count)](
            query_spec.detach().contiguous(),
            key_spec.detach().contiguous(),
            value_spec.detach().contiguous(),
            a_spec.detach().contiguous(),
            b_spec.detach().contiguous(),
            self._qwen_gdn_t16_a_log_neg_exp(),
            self.dt_bias.detach().contiguous(),
            root_state,
            state_out,
            output,
            accepted,
            self.head_k_dim**-0.5,
            h_count,
            hv_count,
            128,
            128,
            128,
            block_v,
            num_warps=1,
            num_stages=3,
        )
        ssm_state.index_copy_(0, commit_index, state_out.to(ssm_state.dtype))
        logger.info_once(
            "Qwen GDN T16 commit1 unpaired Triton path active: rows=16 "
            "accepted=%s state_dtype=%s",
            accepted,
            ssm_state.dtype,
        )
        return output

    def _maybe_forward_qwen_gdn_flashinfer_mtp_prefill(
        self,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        a_raw: torch.Tensor,
        b_raw: torch.Tensor,
        initial_state: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.enable_qwen_gdn_flashinfer_mtp:
            return None
        if query is None or key is None or value is None:
            return None
        if attn_metadata.spec_sequence_masks is not None:
            return None
        if attn_metadata.num_prefills != 1 or attn_metadata.num_decodes != 0:
            return None
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            return None

        rows = attn_metadata.num_actual_tokens
        if rows <= 1 or rows > 64:
            return None
        num_v_heads = self.num_v_heads // self.tp_size
        if query.shape[0] != 1 or query.shape[1] != rows:
            return None
        if a_raw.shape[0] != rows or b_raw.shape[0] != rows:
            return None
        if initial_state.shape[0] != 1 or not initial_state.is_contiguous():
            return None

        output = core_attn_out[:rows].view(
            1, rows, num_v_heads, self.head_v_dim
        )
        a_mtp = a_raw.detach().contiguous().view(1, rows, num_v_heads)
        b_mtp = b_raw.detach().contiguous().view(1, rows, num_v_heads)

        try:
            if initial_state.dtype == torch.bfloat16:
                from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                    gated_delta_rule_mtp as flashinfer_gdn_mtp,
                )

                result = flashinfer_gdn_mtp(
                    A_log=self.A_log.detach(),
                    a=a_mtp,
                    dt_bias=self.dt_bias.detach(),
                    q=query.detach().contiguous(),
                    k=key.detach().contiguous(),
                    v=value.detach().contiguous(),
                    b=b_mtp,
                    initial_state_source=initial_state.detach(),
                    initial_state_indices=None,
                    intermediate_states_buffer=None,
                    disable_state_update=False,
                    use_qk_l2norm_in_kernel=False,
                    scale=self.head_k_dim**-0.5,
                    output=output,
                )
                if result is not output:
                    output.copy_(result)
            elif initial_state.dtype == torch.float32:
                from flashinfer.gdn_decode import gated_delta_rule_mtp

                state_indices = self._qwen_gdn_flashinfer_mtp_zero_index
                if state_indices is None or state_indices.device != query.device:
                    state_indices = torch.zeros(
                        1, dtype=torch.int32, device=query.device
                    )
                    self._qwen_gdn_flashinfer_mtp_zero_index = state_indices
                output, _ = gated_delta_rule_mtp(
                    q=query.detach().contiguous(),
                    k=key.detach().contiguous(),
                    v=value.detach().contiguous(),
                    initial_state=initial_state.detach(),
                    initial_state_indices=state_indices,
                    A_log=self.A_log.detach(),
                    a=a_mtp,
                    dt_bias=self.dt_bias.detach(),
                    b=b_mtp,
                    scale=self.head_k_dim**-0.5,
                    output=output,
                    intermediate_states_buffer=None,
                    disable_state_update=False,
                    use_qk_l2norm=False,
                )
            else:
                return None
        except Exception as exc:
            logger.warning_once(
                "Qwen GDN FlashInfer MTP prefill path failed; falling back to FLA: %s",
                exc,
            )
            return None

        logger.info_once(
            "Qwen GDN FlashInfer MTP prefill path active: rows=%s state_dtype=%s",
            rows,
            initial_state.dtype,
        )
        return output, initial_state

    def _maybe_forward_qwen_gdn_short_prefill_recurrent(
        self,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        a_raw: torch.Tensor,
        b_raw: torch.Tensor,
        initial_state: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.enable_qwen_gdn_short_prefill_recurrent:
            return None
        if query is None or key is None or value is None:
            return None
        if attn_metadata.spec_sequence_masks is not None:
            return None
        if attn_metadata.num_prefills != 1 or attn_metadata.num_decodes != 0:
            return None

        rows = attn_metadata.num_actual_tokens
        if rows <= 1 or rows > self.qwen_gdn_short_prefill_max_tokens:
            return None
        if query.shape[0] != 1 or query.shape[1] != rows:
            return None
        if initial_state.shape[0] != 1:
            return None

        out, per_token_state = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a_raw,
            b=b_raw,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=initial_state,
            inplace_final_state=False,
            cu_seqlens=attn_metadata.non_spec_query_start_loc,
            use_qk_l2norm_in_kernel=False,
        )
        final_state = per_token_state.narrow(0, rows - 1, 1).contiguous()
        logger.info_once(
            "Qwen GDN short-prefill recurrent path active: rows=%s state_dtype=%s",
            rows,
            initial_state.dtype,
        )
        return out, final_state

    def _maybe_qwen_gdn_fused_conv_prep(
        self,
        mixed_qkv: torch.Tensor | None,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        attn_metadata: GDNAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.enable_qwen_gdn_fused_conv_prep:
            return None
        if mixed_qkv is None or state_indices is None:
            return None
        if attn_metadata.spec_sequence_masks is not None:
            return None
        if attn_metadata.num_prefills != 1 or attn_metadata.num_decodes != 0:
            return None

        rows = attn_metadata.num_actual_tokens
        state_len = conv_weights.shape[1] - 1
        if rows <= state_len or rows > self.qwen_gdn_short_prefill_max_tokens:
            return None
        if state_indices.numel() != 1:
            if self.layer_idx == 0 and rows <= self.qwen_gdn_short_prefill_max_tokens:
                logger.info_once(
                    "Qwen GDN fused conv+prep skip: state_indices=%s",
                    tuple(state_indices.shape),
                )
            return None
        expected_dim = 2 * (self.num_k_heads // self.tp_size) * self.head_k_dim
        expected_dim += (self.num_v_heads // self.tp_size) * self.head_v_dim
        if mixed_qkv.shape != (rows, expected_dim):
            if self.layer_idx == 0 and rows <= self.qwen_gdn_short_prefill_max_tokens:
                logger.info_once(
                    "Qwen GDN fused conv+prep skip: shape=%s rows=%s expected_dim=%s",
                    tuple(mixed_qkv.shape),
                    rows,
                    expected_dim,
                )
            return None
        if not mixed_qkv.is_cuda:
            return None
        if conv_state.dim() != 3 or conv_weights.shape[1] != self.conv_kernel_size:
            if self.layer_idx == 0 and rows <= self.qwen_gdn_short_prefill_max_tokens:
                logger.info_once(
                    "Qwen GDN fused conv+prep skip: conv_state=%s conv_weights=%s",
                    tuple(conv_state.shape),
                    tuple(conv_weights.shape),
                )
            return None
        if self.activation not in ("silu", "swish"):
            return None

        result = qwen_gdn_short_prefill_conv_prep(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            conv_state=conv_state,
            conv_weights=conv_weights,
            conv_bias=self.conv1d.bias,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        logger.info_once(
            "Qwen GDN fused conv+prep path active: rows=%s state_dtype=%s",
            rows,
            conv_state.dtype,
        )
        return result

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        self._maybe_log_dflash_gdn_row_trace(attn_metadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        use_dflash_ddtree_branch_gdn = self._use_dflash_ddtree_branch_gdn(
            attn_metadata
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                a_spec = a
                b_spec = b
            else:
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
        else:
            a_spec = None
            b_spec = None

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None and use_dflash_ddtree_branch_gdn:
            assert mixed_qkv_spec is not None
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert a_spec is not None and b_spec is not None
            core_attn_out_spec = self._forward_dflash_ddtree_branch_gdn_spec(
                mixed_qkv_spec=mixed_qkv_spec,
                a_spec=a_spec,
                b_spec=b_spec,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
            )
        elif spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        fused_conv_prep_non_spec = None
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            if spec_sequence_masks is not None:
                a_non_spec_for_conv = a.index_select(0, non_spec_token_indx)
                b_non_spec_for_conv = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec_for_conv = a
                b_non_spec_for_conv = b
            fused_conv_prep_non_spec = self._maybe_qwen_gdn_fused_conv_prep(
                mixed_qkv=mixed_qkv_non_spec,
                a=a_non_spec_for_conv,
                b=b_non_spec_for_conv,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=non_spec_state_indices_tensor,
                has_initial_state=has_initial_state,
                attn_metadata=attn_metadata,
            )
            if fused_conv_prep_non_spec is None:
                mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                # - "cache_indices" updates the conv_state cache in positions
                #   pointed to by "state_indices_tensor"
                mixed_qkv_non_spec = causal_conv1d_fn(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if use_dflash_ddtree_branch_gdn:
            query_spec, key_spec, value_spec = None, None, None
        else:
            query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if fused_conv_prep_non_spec is not None:
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = fused_conv_prep_non_spec
            else:
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = fused_post_conv_prep(
                    conv_output=mixed_qkv_non_spec,
                    a=a_non_spec,
                    b=b_non_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=False,
                )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None and use_dflash_ddtree_branch_gdn:
            last_recurrent_state = None
        elif spec_sequence_masks is not None:
            assert a_spec is not None and b_spec is not None
            core_attn_out_spec = (
                self._maybe_forward_qwen_gdn_t16_commit1_unpaired_spec(
                    query_spec=query_spec,
                    key_spec=key_spec,
                    value_spec=value_spec,
                    a_spec=a_spec,
                    b_spec=b_spec,
                    ssm_state=ssm_state,
                    core_attn_out=core_attn_out,
                    attn_metadata=attn_metadata,
                )
            )
            if core_attn_out_spec is None:
                core_attn_out_spec = self._maybe_forward_qwen_gdn_flashinfer_mtp_spec(
                    query_spec=query_spec,
                    key_spec=key_spec,
                    value_spec=value_spec,
                    a_spec=a_spec,
                    b_spec=b_spec,
                    ssm_state=ssm_state,
                    core_attn_out=core_attn_out,
                    attn_metadata=attn_metadata,
                )
            if core_attn_out_spec is not None:
                last_recurrent_state = None
            else:
                core_attn_out_spec, last_recurrent_state = (
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log,
                        a=a,
                        b=b,
                        dt_bias=self.dt_bias,
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                            : attn_metadata.num_spec_decodes
                            + 1  # type: ignore[attr-defined]
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]
            assert has_initial_state is not None
            initial_state[~has_initial_state, ...] = 0  # type: ignore[operator]
            mtp_prefill_result = (
                self._maybe_forward_qwen_gdn_flashinfer_mtp_prefill(
                    query=query_non_spec,
                    key=key_non_spec,
                    value=value_non_spec,
                    a_raw=a_non_spec,
                    b_raw=b_non_spec,
                    initial_state=initial_state,
                    core_attn_out=core_attn_out,
                    attn_metadata=attn_metadata,
                )
            )
            if mtp_prefill_result is not None:
                core_attn_out_non_spec, last_recurrent_state = mtp_prefill_result
            else:
                short_prefill_result = (
                    self._maybe_forward_qwen_gdn_short_prefill_recurrent(
                        query=query_non_spec,
                        key=key_non_spec,
                        value=value_non_spec,
                        a_raw=a_non_spec,
                        b_raw=b_non_spec,
                        initial_state=initial_state,
                        attn_metadata=attn_metadata,
                    )
                )
                if short_prefill_result is not None:
                    core_attn_out_non_spec, last_recurrent_state = (
                        short_prefill_result
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = self.chunk_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=initial_state,
                        output_final_state=True,
                        cu_seqlens=non_spec_query_start_loc,
                        chunk_indices=attn_metadata.chunk_indices,
                        chunk_offsets=attn_metadata.chunk_offsets,
                        use_qk_l2norm_in_kernel=False,
                    )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_fast(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return


def gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``fast_kernel=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``fast_kernel=True`` (AITER Triton fast path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if fast_kernel:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    fast_kernel: bool,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
