# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU check of the sharded draft argmax (VLLM_SPEC_DRAFT_SHARDED_ARGMAX)
against the stock greedy draft path, argmax(LogitsProcessor(lm_head, h)),
with a real vocab-parallel lm_head and vLLM's TP communicators, eagerly and
from a captured CUDA graph. Tie-heavy integer weights stress the
lowest-id tie-break across ranks."""

import pytest
import ray
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.spec_decode.sharded_argmax import sharded_draft_argmax

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

VOCAB_SIZE = 131072
HIDDEN_SIZE = 4096


@ray.remote(num_gpus=1, max_calls=1)
def sharded_draft_argmax_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        with set_current_vllm_config(VllmConfig()), torch.device(device):
            lm_head = ParallelLMHead(
                VOCAB_SIZE, HIDDEN_SIZE, params_dtype=torch.bfloat16
            )
            logits_processor = LogitsProcessor(VOCAB_SIZE)
        assert lm_head.tp_size == tp_size
        # Same full weight on every rank; the loader keeps this rank's shard.
        gen = torch.Generator(device=device).manual_seed(1234)
        weight = torch.randint(
            -2, 3, (VOCAB_SIZE, HIDDEN_SIZE), generator=gen, device=device
        ).to(torch.bfloat16)
        lm_head.weight_loader(lm_head.weight, weight)
        del weight

        def stock(hidden_states):
            return logits_processor(lm_head, hidden_states).argmax(dim=-1)

        def hidden(num_rows, seed):
            g = torch.Generator(device=device).manual_seed(seed)
            # Identical on all ranks, as for the drafter.
            return torch.randint(
                -1, 2, (num_rows, HIDDEN_SIZE), generator=g, device=device
            ).to(torch.bfloat16)

        for num_rows in (1, 2, 4, 8):
            for trial in range(10):
                hs = hidden(num_rows, trial + 7 * num_rows)
                got = sharded_draft_argmax(lm_head, logits_processor, hs)
                assert got.dtype == torch.int64
                assert torch.equal(got, stock(hs)), (num_rows, trial)

        num_rows = 4
        static = torch.zeros(num_rows, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        with graph_capture(device=device) as graph_capture_context:
            # Warm-up (Triton compile) on the capture stream.
            sharded_draft_argmax(lm_head, logits_processor, static)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                out = sharded_draft_argmax(lm_head, logits_processor, static)
        for trial in range(100):
            static.copy_(hidden(num_rows, 1000 + trial))
            graph.replay()
            assert torch.equal(out, stock(static)), trial
        torch.accelerator.synchronize()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_sharded_draft_argmax_matches_stock(monkeypatch: pytest.MonkeyPatch, tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, 1, sharded_draft_argmax_worker)
