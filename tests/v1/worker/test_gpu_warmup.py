# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.worker.gpu.warmup import (
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)


def test_mixed_prefill_decode_warmup_skips_bs1():
    model_runner = SimpleNamespace(
        is_pooling_model=False,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    execute_model = Mock()
    sample_tokens = Mock()

    executed = run_mixed_prefill_decode_warmup(
        model_runner,
        execute_model,
        sample_tokens,
        num_tokens=128,
    )

    assert not executed
    execute_model.assert_not_called()
    sample_tokens.assert_not_called()


def test_generic_warmup_skips_deepseek_v4_sparse_attention():
    backend = Mock()
    backend.get_name.return_value = "DEEPSEEK_SPARSE_SWA"
    model_runner = SimpleNamespace(
        attn_groups=[[SimpleNamespace(backend=backend)]],
    )
    execute_model = Mock()
    sample_tokens = Mock()

    warmup_kernels(model_runner, execute_model, sample_tokens)

    execute_model.assert_not_called()
    sample_tokens.assert_not_called()
