# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# fmt: off
"""Reference for test_mamba2_fused_decode.py: the Inductor kernel that vLLM's
torch.compile (VLLM_COMPILE, v0.30, torch 2.14, triton 3.8) generates on B200 for
MambaMixer2's Mixer2RMSNormGated.forward_native (8 groups, TP4 -> 2 local
groups of 1024) followed by the static per-tensor QuantFP8.forward_native of
the FP8 out_proj input. Copied verbatim from the compile cache of a TP4 server
(only the DeviceProperties index was set to 0); launched through Inductor's own
CachingAutotuner, as the compiled graph does."""

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*fp8e4nv', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_silu_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 167776256}, 'kernel_num_gb': 0.1048617, 'kernel_flop': 0, 'backend_hash': 'B49614C0BBA23CA71245E046F2A6ABFCFD211E8FB0FFF5563191062F952B4CD3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_silu_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 2)
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x3), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 1024*x0 + 4640*x1), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (((r0_2 + 1024*x3) % 2048)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (((r0_2 + 1024*x0 + 2048*x1) % (2048*ks0))), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr3 + (0))
    tmp29 = tl.broadcast_to(tmp28, [1, 1])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp1 * tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp8
    tmp19 = tl.full([1, 1], 1024.0, tl.float32)
    tmp20 = (tmp14 / tmp19)
    tmp21 = tl.full([1, 1], 1e-05, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp15 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp30 = (tmp6 / tmp29)
    tmp31 = tmp27 * tmp30
    tmp32 = tl.full([1, 1], -448.0, tl.float32)
    tmp33 = tl.maximum(tmp31, tmp32, tl.PropagateNan.ALL)
    tmp34 = tl.full([1, 1], 448.0, tl.float32)
    tmp35 = tl.minimum(tmp33, tmp34, tl.PropagateNan.ALL)
    tmp36 = tmp35.to(tl.float8e4nv)
    tl.store(out_ptr1 + (((r0_2 + 1024*x3) % (2048*ks0))), tmp36, xmask)
