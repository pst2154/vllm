# Qwen3.6 NVFP4 + DFlash on DGX Spark

This branch carries the best confirmed Qwen3.6 35B A3B NVFP4/DFlash path from
the DGX Spark optimization run.

## Best Confirmed Result

| Item | Value |
| --- | --- |
| Model | `Qwen3.6-35B-A3B-NVFP4` |
| Draft model | `Qwen3.6-35B-A3B-DFlash` |
| Hardware | DGX Spark / GB10 |
| vLLM path | TF5 container plus this fork's GDN fast path |
| Active optimization | `VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1` |
| Runtime recipe | DFlash `num_speculative_tokens=15`, fp16 GDN SSM cache, CUDA graph disabled |
| Benchmark gate | `pp=2048`, `tg=128`, `depth=0`, concurrency 1, no prompt cache, generation latency |
| Best 5-run TG128 mean | `97.20284519156971 tok/s` |
| Best 5-run values | `74.78382592759694`, `102.32446498702366`, `95.06421604875982`, `130.84421561667477`, `82.99750337779336` |
| Activation log | `Qwen GDN T16 commit1 unpaired Triton path active: rows=16 accepted=16 state_dtype=torch.float16` |

A later 5-hour offline loop did not beat this result. The post-loop serving
confirmation with 2 warmup and 5 measured runs scored `85.85016733500065 tok/s`,
so the `97.20284519156971 tok/s` run remains the score to beat.

## How To Run

The tested setup used the Spark TF5 image and mounted this fork's edited GDN
files into the container. Set the paths for your machine first:

```bash
export VLLM_REPO=/home/asteiner/Git_Repos/vllm
export MODEL_ROOT=/home/asteiner/models
export MODS_DIR=/home/asteiner/Git_Repos/spark-vllm-docker/mods
```

Start the server:

```bash
docker run -d \
  --name qwen36-t16-commit1-unpaired \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MODEL_ROOT}:/models:ro" \
  -v "${MODS_DIR}:/workspace/mods:ro" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1 \
  -v "${VLLM_REPO}/vllm/model_executor/layers/mamba/gdn_linear_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py:ro" \
  -v "${VLLM_REPO}/vllm/v1/attention/backends/gdn_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py:ro" \
  ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest \
  bash -lc 'set -euo pipefail
cd /workspace/mods/fix-qwen3.5-autoround && bash run.sh
cd /workspace/mods/fix-qwen3-coder-next && bash run.sh
exec vllm serve /models/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b \
  --language-model-only \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --dtype auto \
  --kv-cache-dtype auto \
  --load-format fastsafetensors \
  --attention-backend flash_attn \
  --enable-chunked-prefill \
  --trust-remote-code \
  --quantization compressed-tensors \
  --enable-prefix-caching \
  --moe-backend flashinfer_cutlass \
  --mamba-ssm-cache-dtype float16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --optimization-level 3 \
  --performance-mode throughput \
  --default-chat-template-kwargs "{\"preserve_thinking\":true}" \
  --override-generation-config "{\"temperature\":0.5,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0}" \
  --disable-uvicorn-access-log \
  --no-enable-log-requests \
  --compilation-config "{\"cudagraph_mode\":0}" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.6-35B-A3B-DFlash\",\"num_speculative_tokens\":15}"'
```

Wait for health:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Check that the fast path activated:

```bash
docker logs qwen36-t16-commit1-unpaired 2>&1 \
  | grep 'Qwen GDN T16 commit1 unpaired Triton path active'
```

## Benchmark

Run two warmup passes first:

```bash
/home/asteiner/Git_Repos/vllm/.venv/bin/llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b \
  --served-model-name qwen3.6-35b \
  --tokenizer /home/asteiner/models/Qwen3.6-35B-A3B-NVFP4 \
  --pp 2048 \
  --tg 128 \
  --depth 0 \
  --runs 2 \
  --no-cache \
  --latency-mode generation \
  --save-result qwen36_t16_warmup_tg128.json \
  --format json
```

Then run the measured 5-run gate:

```bash
/home/asteiner/Git_Repos/vllm/.venv/bin/llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b \
  --served-model-name qwen3.6-35b \
  --tokenizer /home/asteiner/models/Qwen3.6-35B-A3B-NVFP4 \
  --pp 2048 \
  --tg 128 \
  --depth 0 \
  --runs 5 \
  --no-cache \
  --latency-mode generation \
  --save-result qwen36_t16_live5_tg128.json \
  --format json
```

## Uploaded Changes

| Change | Files / flag | What it does | Evidence |
| --- | --- | --- | --- |
| T16 commit-one unpaired GDN recurrent kernel | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1` | Specializes the live DFlash verifier shape: one request, 16 verifier rows, DFlash k=15, 128x128 GDN state tiles. It computes every verifier output row exactly and commits only the accepted recurrent state row in one Triton launch. | Best offline GDN fixture: `16.408000946044922 us`; best served TG128: `97.20284519156971 tok/s`. |
| Unpaired value-head launch layout | `_qwen_gdn_t16_commit1_unpaired_kernel` | Launches one value head per program with `block_v=8`. This repeats q/k work for sibling value heads but reduces per-program state pressure enough to beat paired-HV variants on GB10. | Paired-HV and shared sibling-HV experiments were slower; the unpaired path remained the incumbent. |
| Accepted-token CPU metadata | `vllm/v1/attention/backends/gdn_attn.py` | Carries the already-known accepted-token count into GDN metadata so the fast path can choose the exact state row to commit without changing target verification semantics. | Required by the active T16 path; activation log reports the accepted row count. |
| fp16 GDN SSM cache recipe | Runtime flag `--mamba-ssm-cache-dtype float16` | Uses fp16 for the GDN recurrent cache while the kernel computes the recurrent update in fp32 and writes the committed row back to the cache dtype. This reduces state bandwidth for the short verifier path. | Used in the best serving run; offline max abs error stayed under the `0.02` gate. |
| Cached `-exp(A_log)` decay vector | `_qwen_gdn_t16_a_log_neg_exp()` | Precomputes and caches the static per-head decay coefficient used by the recurrent kernel. | Keeps the hot verifier launch focused on per-token recurrent work. |
| Strict guards and stock fallback | `_maybe_forward_qwen_gdn_t16_commit1_unpaired_spec()` | Enables the fast path only for the exact tested shape: no prefill rows, one speculative decode, `num_spec=15`, 16 verifier rows, 128-dim heads, and contiguous GDN state metadata. All other cases fall back to the stock path. | Avoids silently changing behavior outside the benchmarked Qwen/DFlash case. |
| DFlash k=15 serving recipe | `--speculative-config '{"method":"dflash",...,"num_speculative_tokens":15}'` | Uses the DFlash drafter at the draft width that produced the best live result with the T16 GDN verifier. | Best live result used DFlash k=15. |
| CUDA graph disabled for this recipe | `--compilation-config '{"cudagraph_mode":0}'` | Avoids CUDA graph capture for this Spark/TF5 path; the measured best was with compile enabled but CUDA graph mode set to none. | Used in both the best run and the post-loop confirmation. |
| Experimental Qwen GDN hooks, left off by default | `VLLM_QWEN_GDN_FLASHINFER_MTP`, `VLLM_QWEN_GDN_FUSED_CONV_PREP`, `VLLM_QWEN_GDN_FP8_PROJ`, `VLLM_QWEN_GDN_NVFP4_PROJ`, `VLLM_QWEN_GDN_TRITON_PROJ`, DDTree-related GDN flags | Keeps bounded experiment hooks in the fork, but they are not part of the best recipe unless explicitly enabled. | These lanes did not replace the T16 unpaired path in the confirmed score. |

## Notes

- Do not enable B12x on this host unless you explicitly accept the Spark hang
  risk from prior experiments.
- The current best is noisy. Compare candidates with the same benchmark gate and
  at least 2 warmup runs plus 5 measured runs.
- The fast path preserves exact target/GDN verification for emitted tokens; it
  changes how the short GDN verifier work is executed, not the acceptance rule.
