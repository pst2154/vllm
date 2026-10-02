# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer builder: fused multi-step draft decode
(VLLM_FLASHINFER_FUSED_DRAFT_DECODE)."""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FIDecode,
    FlashInferDecodeKernel,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    FlashInferTrtllmAPIDecode,
    TRTLLMPrefill,
)
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _make_builder(
    monkeypatch, *, env: bool, trtllm_decode: bool, dcp_world_size: int = 1
) -> FlashInferMetadataBuilder:
    monkeypatch.setenv("VLLM_FLASHINFER_FUSED_DRAFT_DECODE", "1" if env else "0")
    # Non-gated model config (no HF token needed); only its attention shape
    # is used.
    vllm_config = create_vllm_config(model_name="Qwen/Qwen3-0.6B", max_model_len=1024)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "can_use_trtllm_attention",
        lambda *args, **kwargs: trtllm_decode,
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )
    if dcp_world_size > 1:
        monkeypatch.setattr(
            flashinfer_backend,
            "get_dcp_group",
            lambda: SimpleNamespace(world_size=dcp_world_size, rank_in_group=0),
        )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        return FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("env", "trtllm_decode", "dcp_world_size", "expected"),
    [
        (False, True, 1, False),  # default off
        (True, True, 1, True),
        (True, False, 1, False),  # native FlashInfer decode plans on host
        (True, True, 2, False),  # DCP local seq lens are not advanced
    ],
)
def test_fused_draft_decode_gating(
    monkeypatch, env, trtllm_decode, dcp_world_size, expected
):
    builder = _make_builder(
        monkeypatch,
        env=env,
        trtllm_decode=trtllm_decode,
        dcp_world_size=dcp_world_size,
    )
    assert builder.use_trtllm_decode_attention == trtllm_decode
    assert builder.supports_draft_decode_metadata_update is expected


def _metadata(decode, prefill=None, use_cascade=False) -> FlashInferMetadata:
    return FlashInferMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        q_data_type_prefill=torch.bfloat16,
        q_data_type_decode=torch.bfloat16,
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        causal=True,
        prefill=prefill,
        decode=decode,
        use_cascade=use_cascade,
        cascade_wrapper=None,
    )


def _trtllm_decode() -> FlashInferTrtllmAPIDecode:
    return FlashInferTrtllmAPIDecode(
        kernel=FlashInferDecodeKernel.TRTLLM_GEN,
        block_tables=torch.zeros((2, 4), dtype=torch.int32),
        seq_lens=torch.tensor([5, 9], dtype=torch.int32),
        max_seq_len=16,
    )


def test_update_draft_decode_metadata_is_noop_for_trtllm_decode():
    builder = object.__new__(FlashInferMetadataBuilder)
    decode = _trtllm_decode()
    seq_lens = decode.seq_lens
    block_tables = decode.block_tables
    metadata = _metadata(decode)

    builder.update_draft_decode_metadata(metadata)

    # The same persistent tensors stay referenced; the speculator advances
    # them in place between draft steps.
    assert metadata.decode is decode
    assert decode.seq_lens is seq_lens
    assert decode.block_tables is block_tables
    assert decode.max_seq_len == 16


def test_update_draft_decode_metadata_rejects_native_decode():
    builder = object.__new__(FlashInferMetadataBuilder)
    with pytest.raises(RuntimeError, match="trtllm-gen"):
        builder.update_draft_decode_metadata(_metadata(FIDecode(wrapper=None)))


def test_update_draft_decode_metadata_rejects_prefill():
    builder = object.__new__(FlashInferMetadataBuilder)
    prefill = TRTLLMPrefill(
        block_tables=torch.zeros((1, 4), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        cum_seq_lens_q=torch.tensor([0, 8], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 8], dtype=torch.int32),
        max_q_len=8,
        max_seq_len=8,
    )
    with pytest.raises(RuntimeError, match="prefill"):
        builder.update_draft_decode_metadata(_metadata(_trtllm_decode(), prefill))
