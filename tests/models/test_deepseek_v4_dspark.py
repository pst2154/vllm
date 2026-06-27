# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import torch

from vllm.models.deepseek_v4.nvidia.dspark import DeepSeekV4DSpark


def test_dspark_loads_special_heads_outside_decoder_layer_loader():
    markov_w1 = mock.MagicMock()
    confidence_proj = mock.MagicMock()
    backbone = mock.MagicMock()
    backbone.named_parameters.return_value = [
        ("layers.63.markov_head.markov_w1.weight", markov_w1),
        ("layers.63.confidence_head.proj.weight", confidence_proj),
    ]
    backbone.load_weights.return_value = {"layers.61.main_norm.weight"}

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, expert_dtype="fp4"),
        model=backbone,
    )
    main_norm_weight = torch.ones(1)
    markov_weight = torch.ones(2, 1)
    confidence_weight = torch.ones(1, 3)

    loaded = DeepSeekV4DSpark.load_weights(
        model,
        [
            ("mtp.0.main_norm.weight", main_norm_weight),
            ("mtp.2.markov_head.markov_w1.weight", markov_weight),
            ("mtp.2.confidence_head.proj.weight", confidence_weight),
        ],
    )

    backbone.load_weights.assert_called_once_with(
        [("layers.61.main_norm.weight", main_norm_weight)]
    )
    markov_w1.weight_loader.assert_called_once_with(markov_w1, markov_weight)
    confidence_proj.weight_loader.assert_called_once_with(
        confidence_proj, confidence_weight
    )
    backbone.finalize_mega_moe_weights.assert_called_once_with()
    assert loaded == {
        "model.layers.61.main_norm.weight",
        "model.layers.63.markov_head.markov_w1.weight",
        "model.layers.63.confidence_head.proj.weight",
    }
