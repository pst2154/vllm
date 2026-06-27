# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch.nn as nn

from vllm.config import CompilationMode, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_vllm_config = replace(
        vllm_config,
        # DSpark's 512-wide attention head is not supported by FlashInfer with
        # an FP8 KV cache. The draft model has its own cache group, so keep the
        # target cache quantized while storing draft K/V in the model dtype.
        cache_config=replace(vllm_config.cache_config, cache_dtype="auto"),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=True,
            backend=speculative_config.attention_backend,
        ),
    )
    if draft_vllm_config.compilation_config.mode == CompilationMode.NONE:
        model_tag_context = nullcontext()
    else:
        from vllm.compilation.backends import set_model_tag

        model_tag_context = set_model_tag("dspark_head")

    with model_tag_context:
        dspark_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
        )

    assert get_pp_group().world_size == 1, "DSpark does not support pipeline parallel"
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    dspark_model.model.embed_tokens = target_inner.embed_tokens
    dspark_model.lm_head = target_language_model.lm_head
    return dspark_model
