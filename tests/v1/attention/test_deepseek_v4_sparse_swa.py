# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.attention.backends.mla.sparse_swa import _get_decode_threshold


def test_decode_threshold_accounts_for_parallel_drafting():
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            parallel_drafting=True,
        )
    )

    assert _get_decode_threshold(vllm_config) == 11
