# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.lilicorr import (
    LiLiCorrHead,
    check_head_weight_coverage,
    parse_lilicorr_config,
)
from vllm.v1.worker.gpu.spec_decode.lilicorr.kernels import (
    lilicorr_greedy_path,
    lilicorr_topk_log_probs,
)
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

GEOMETRY = {
    "lilicorr_candidate_topk": 4,
    "lilicorr_hidden_size": 8,
    "lilicorr_num_layers": 2,
    "lilicorr_num_heads": 2,
    "lilicorr_mlp_ratio": 2.0,
    "lilicorr_factor_dim": 4,
    "lilicorr_vector_eps": 1e-6,
    "lilicorr_logit_scale": 10.0,
}


def make_config(**overrides):
    geometry = GEOMETRY | overrides
    geometry = {key: value for key, value in geometry.items() if value is not None}
    return SimpleNamespace(dflash_config=geometry)


def make_head() -> LiLiCorrHead:
    torch.manual_seed(0)
    head = LiLiCorrHead(
        model_hidden_size=16,
        block_size=5,
        rms_norm_eps=1e-6,
        config=parse_lilicorr_config(make_config()),
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0, 0.2)
    head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
    return head


@pytest.mark.parametrize("field", sorted(GEOMETRY))
def test_lilicorr_config_requires_every_geometry_field(field: str):
    with pytest.raises(ValueError, match=field):
        parse_lilicorr_config(make_config(**{field: None}))


@pytest.mark.parametrize("top_k", [3, 5, 6])
def test_lilicorr_config_rejects_non_power_of_two_top_k(top_k: int):
    with pytest.raises(ValueError, match="power of two"):
        parse_lilicorr_config(make_config(lilicorr_candidate_topk=top_k))


def test_lilicorr_head_scores_expected_lattice_shapes():
    head = make_head()
    batch_size = 3
    token_embeddings = torch.randn(batch_size, 4, 4, 16)
    candidate_log_probs = torch.randn(batch_size, 4, 4).log_softmax(dim=-1)
    pass_hidden = torch.randn(batch_size, 4, 16)
    anchor_hidden = torch.randn(batch_size, 16)
    anchor_valid = torch.ones(batch_size, dtype=torch.bool)

    start, pair = head.score(
        token_embeddings=token_embeddings,
        candidate_log_probs=candidate_log_probs,
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid,
    )

    assert start.shape == (batch_size, 4)
    assert pair.shape == (batch_size, 3, 4, 4)


def test_invalid_anchor_ignores_stale_anchor_values():
    head = make_head()
    inputs = {
        "token_embeddings": torch.randn(2, 4, 4, 16),
        "candidate_log_probs": torch.randn(2, 4, 4).log_softmax(dim=-1),
        "pass_hidden": torch.randn(2, 4, 16),
        "anchor_valid": torch.zeros(2, dtype=torch.bool),
    }
    first = head.score(anchor_hidden=torch.randn(2, 16), **inputs)
    second = head.score(anchor_hidden=torch.randn(2, 16) * 100, **inputs)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_lilicorr_head_weight_coverage_is_fail_closed():
    head = make_head()
    names = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    check_head_weight_coverage(head, names)

    with pytest.raises(ValueError, match="missing 1 head parameters"):
        check_head_weight_coverage(head, names - {"lilicorr.out_head.weight"})
    with pytest.raises(ValueError, match="1 unexpected head parameters"):
        check_head_weight_coverage(head, names | {"lilicorr.unknown.weight"})


def test_topk_candidates_are_normalized_over_the_full_vocabulary():
    torch.manual_seed(1)
    logits = torch.randn(7, 300)
    actual_values, actual_ids = lilicorr_topk_log_probs(logits, 4)
    expected_values, expected_ids = torch.log_softmax(logits, dim=-1).topk(4)

    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_ids, expected_ids)


def test_greedy_path_follows_conditioned_transitions():
    torch.manual_seed(2)
    batch_size, num_slots, top_k = 3, 4, 4
    start = torch.randn(batch_size, top_k)
    pair = torch.randn(batch_size, num_slots - 1, top_k, top_k)
    candidate_ids = torch.randint(0, 100, (batch_size, num_slots, top_k))
    actual = lilicorr_greedy_path(start, pair, candidate_ids)

    expected = torch.empty(batch_size, num_slots, dtype=torch.int64)
    for row in range(batch_size):
        current = int(start[row].argmax())
        expected[row, 0] = candidate_ids[row, 0, current]
        for slot in range(1, num_slots):
            current = int(pair[row, slot - 1, current].argmax())
            expected[row, slot] = candidate_ids[row, slot, current]

    torch.testing.assert_close(actual, expected)


def test_context_anchor_uses_last_non_rejected_row():
    speculator = LiLiCorrSpeculator.__new__(LiLiCorrSpeculator)
    speculator.max_num_reqs = 4
    speculator.anchor_hidden = torch.zeros(4, 3)
    speculator.anchor_valid = torch.zeros(4, dtype=torch.bool)
    hidden_states = torch.arange(18, dtype=torch.float32).view(6, 3)
    input_batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 6], dtype=torch.int32),
    )

    speculator._set_context_hidden_states(
        hidden_states, input_batch, torch.tensor([0, 2])
    )

    torch.testing.assert_close(speculator.anchor_hidden[0], hidden_states[1])
    torch.testing.assert_close(speculator.anchor_hidden[1], hidden_states[3])
    assert speculator.anchor_valid.tolist() == [True, True, False, False]
